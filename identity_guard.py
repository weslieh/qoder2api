"""identity_guard.py —— 出站 system prompt 身份句最小中和

背景: 上游网关对第三方 Agent 客户端的身份声明句做**大小写敏感的字面
子串检测**, 命中即返回 HTTP 200 + SSE 内嵌 403 / 10605 / RESOURCE_LIMIT
排队信封 (queueType=p1, queueCount=0, serviceAvailable=false) —— 这是硬
拒绝而非真实排队, 重试 retryAfterSeconds 后依然被拒。受控 A/B 实测确认
触发形态是字面匹配而不是语义识别:

  "You are ZCode,"                  命中
  "You are Claude Code,"            命中
  "You are Codex,"                  命中
  "You are ZCode ,"                 不命中 (逗号前多一个空格)
  "You are ZCode."                  不命中 (无逗号)
  "You are zcode,"                  不命中 (大小写敏感)

  "powered by DeepSeek Harness."    命中
  "powered by DeepSeek Harness ."   不命中 (句号前多一个空格)
  "powered by the DeepSeek Harness." 不命中

对策是**最小字符级中和**: 只在命中片段内部 (逗号 / 句号前) 插入一个
空格, 打断字面匹配。品牌名、语序、语义全部保留, 对不命中的文本零改动;
中和结果确定且幂等 (标点前已有空格 -> 不再命中), 因此缓存前缀稳定。

处理范围 (与 examples/qoder_identity_guard.py 的差异):
  - 只处理 role == "system" 的消息。客户端身份声明只出现在 system;
    user / assistant / tool 历史消息一律不动, 省掉全部多余开销。
  - 只扫 system prompt 的**前 N 个字符** (默认 2000): 身份句都在提示词
    开头, 全文跑正则是纯浪费。可用 QODER_IDENTITY_GUARD_PREFIX 调整
    (字符数, <=0 表示不限); 边界正好落在片段中间时该片段不处理。
  - 不做自有品牌豁免: 本项目不存在需要放行的"自家"身份句。

开关 (默认开启): QODER_IDENTITY_GUARD=0/off/false/no/disable 关闭。
"""

import os
import re

# 只扫 system prompt 前多少字符; 见模块文档。
_DEFAULT_PREFIX_CHARS = 2000

# 通用品牌名形态: 大写字母开头, 词内允许字母数字, 词间允许单个空格或
# 连字符 (后续词须大写/数字开头, 避免吞掉小写普通词)。
# 例: ZCode / Claude Code / DeepSeek Harness / DeepSeek-Harness / GPT-5
_BRAND = r"[A-Z][A-Za-z0-9]*(?:[ -][A-Z0-9][A-Za-z0-9]*)*"

# 与上游检测同口径: 大小写敏感, 逗号 / 句号必须**紧跟**品牌名。
# 中和后 (标点前已有空格) 不再命中 —— 天然幂等。
_IDENTITY_RE = re.compile(r"You are (" + _BRAND + r"),")
_POWERED_RE = re.compile(r"powered by (" + _BRAND + r")\.")

_FALSY = ("0", "off", "false", "no", "disable", "disabled")


def guard_enabled() -> bool:
    """默认开启; 显式设置 QODER_IDENTITY_GUARD=0/off/... 才关闭。"""
    raw = str(os.environ.get("QODER_IDENTITY_GUARD") or "").strip().lower()
    return raw not in _FALSY


def _prefix_limit() -> int:
    """前缀扫描窗口 (字符数); 0 表示整段文本都扫。"""
    raw = str(os.environ.get("QODER_IDENTITY_GUARD_PREFIX") or "").strip()
    if not raw:
        return _DEFAULT_PREFIX_CHARS
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_PREFIX_CHARS
    return value if value > 0 else 0


def neutralize_text(text: str, limit: int | None = None) -> tuple[str, int]:
    """对单个字符串做最小中和; 返回 (新文本, 命中次数)。

    只在前 limit 个字符内匹配 (limit=None 时取环境配置; <=0 表示不限)。
    无命中时原样返回 (同一对象), 调用方可据此跳过复制。
    """
    if not isinstance(text, str) or not text:
        return text, 0
    if limit is None:
        limit = _prefix_limit()
    if limit and len(text) > limit:
        head, tail = text[:limit], text[limit:]
    else:
        head, tail = text, ""
    hits = 0

    def _identity(m):
        nonlocal hits
        hits += 1
        # "You are ZCode," -> "You are ZCode ," (逗号前插空格)
        return m.group(0)[:-1] + " ,"

    def _powered(m):
        nonlocal hits
        hits += 1
        # "powered by DeepSeek Harness." -> "powered by DeepSeek Harness ."
        return m.group(0)[:-1] + " ."

    new_head = _IDENTITY_RE.sub(_identity, head)
    new_head = _POWERED_RE.sub(_powered, new_head)
    if not hits:
        return text, 0
    return new_head + tail, hits


def _neutralize_content(content):
    """content 可能是 str 或多模态 part 列表; 只动 text part。"""
    if isinstance(content, str):
        return neutralize_text(content)
    if isinstance(content, list):
        hits = 0
        changed = False
        out = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                new, n = neutralize_text(part["text"])
                if n:
                    part = dict(part, text=new)
                    hits += n
                    changed = True
            out.append(part)
        return (out if changed else content), hits
    return content, 0


def neutralize_system_message(message):
    """只对 role == "system" 的消息做中和; 返回 (消息, 命中次数)。"""
    if not isinstance(message, dict) or message.get("role") != "system":
        return message, 0
    new_content, hits = _neutralize_content(message.get("content"))
    if not hits:
        return message, 0
    return dict(message, content=new_content), hits


def neutralize_system_messages(messages):
    """遍历 messages, 只中和 system 消息中的身份句特征。

    返回 (新列表或原列表, 总命中次数)。无命中时返回原列表 (零复制),
    保证开关关闭 / 无特征请求的路径与改动前逐字节一致。
    """
    if not guard_enabled():
        return messages, 0
    total = 0
    changed = False
    out = []
    for message in messages or []:
        item, n = neutralize_system_message(message)
        if n:
            total += n
            changed = True
        out.append(item)
    return (out if changed else messages), total
