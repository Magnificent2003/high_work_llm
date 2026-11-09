"""多图调用示例；模型与接口配置由项目根目录 .env 提供。"""
from pathlib import Path

from llm import VLMInvoker


# 两段提示词可独立修改；设为空字符串时不发送对应提示词。
system_prompt = ""
user_prompt = "请逐张分析这些图片，比较各张图片的异同。"

# 这里填写多个本地图片路径，也支持 URL 或 data URL。
project_dir = Path(__file__).resolve().parent
image_paths = [
    project_dir / "viz_preview/00000002.png",
    project_dir / "viz_preview/00000014.png",
]


def main():
    AI = VLMInvoker()
    text = AI.chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_paths=image_paths,
        label_images=True,
    )
    print(text)


if __name__ == "__main__":
    main()
