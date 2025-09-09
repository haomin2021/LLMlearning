import pathlib
import re
from ollama import Client

client = Client(host="http://localhost:11434")

def detect_output_language(text: str) -> str:
    """
    极简语言检测：含有中日韩字符则用中文，否则用英文。
    够用且无需额外依赖；需要更准可以换成 langdetect 包。
    """
    return "Chinese" if re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', text) else "English"

def summarize_text(content: str, bullets: int = 10) -> str:
    # 1) 选定输出语言（对你目前英文原文会锁定为 English）
    out_lang = detect_output_language(content)  # "English" 或 "Chinese"

    # 2) 生成编号骨架（强约束）
    skeleton = "\n".join(f"{i}." for i in range(1, bullets + 1))

    # 3) 用英文下达硬性规则，避免模型“顺着中文说中文”
    prompt = f"""
            You are a precise academic summarizer.

            TASK:
            - Produce EXACTLY {bullets} bullet points.
            - Write STRICTLY in {out_lang}. Do NOT translate the language of the input.
            - Keep key conclusions, numbers, dates, names, terms.
            - If information is missing, write "(insufficient information)" for that bullet.
            - Output ONLY the numbered list (no title, no preface, no extra text).

            TEXT TO SUMMARIZE:
            {content}

            OUTPUT TEMPLATE (fill each line with one bullet):
            {skeleton}
            """

    resp = client.generate(
        model="llama3.1:8b",         # 确保已 pull
        prompt=prompt,
        options={"temperature": 0.2, "num_ctx": 8192}
    )["response"].strip()

    # 4) 后处理：只提取真正的列表项，保证恰好 bullets 条
    #    兼容 "1. ..." / "- ..." / "* ..." 三种前缀
    lines = re.findall(r'^\s*(?:\d+\.\s+|[-*]\s+)(.+)$', resp, flags=re.MULTILINE)
    items = [re.sub(r'\s+', ' ', x).strip() for x in lines if x.strip()]

    # 不足就补，超出就裁剪
    if len(items) < bullets:
        items += ["(insufficient information)"] * (bullets - len(items))
    else:
        items = items[:bullets]

    # 统一渲染为简单的无序列表（也可保留编号，看你喜好）
    return "\n".join(f"- {it}" for it in items)

if __name__ == "__main__":
    file_path = pathlib.Path(__file__).parent / "exampletext.txt"
    text_content = file_path.read_text(encoding="utf-8")
    summary = summarize_text(text_content, bullets=10)
    print("=== 总结 ===")
    print(summary)