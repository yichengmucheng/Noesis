import argparse

from openai import OpenAI
from transformers.utils.versions import require_version
from PIL import Image
import json
import os
import sys
sys.path.append('../')  # 加入项目根目录（dots_ocr 模块所在位置）
sys.path.append('./')   # 加入当前 demo 目录（确保依赖能找到）

from dots_ocr.utils import dict_promptmode_to_prompt
from dots_ocr.model.inference import inference_with_vllm


parser = argparse.ArgumentParser()
parser.add_argument("--ip", type=str, default="localhost")
parser.add_argument("--port", type=str, default="7681")
parser.add_argument("--model_name", type=str, default="dotsocr-model")
parser.add_argument("--prompt_mode", type=str, default="prompt_layout_all_en")

args = parser.parse_args()

require_version("openai>=1.5.0", "To fix: pip install openai>=1.5.0")


def main():
    addr = f"http://{args.ip}:{args.port}/v1"
    image_path = "/mnt/mydisk1/zhixinguangyu/sxt_home/dots.ocr-master/demo/demo_image1.jpg"
    prompt = dict_promptmode_to_prompt[args.prompt_mode]
    image = Image.open(image_path)
    response = inference_with_vllm(
        image,
        prompt, 
        ip=args.ip,
        port=args.port,
        temperature=0.1,
        top_p=0.9,
        model_name=args.model_name,
    )
    print(f"response: {response}")

    # 5. 自动保存结果（核心新增：不修改原代码，只在脚本层保存）
    # 生成结果文件名（和图片同名，后缀改为 .json）
    image_dir = os.path.dirname(image_path)  # 结果保存在图片同目录
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    result_path = os.path.join(image_dir, f"{image_name}_ocr_result.json")
    
    # 保存为JSON格式（方便后续解析）
    try:
        # 尝试解析响应为JSON（适配现有输出格式）
        result_json = json.loads(response) if isinstance(response, str) else response
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2)
        print(f"\n💾 结果已保存到：{result_path}")
    except Exception as e:
        # 若不是JSON格式，保存为文本文件
        result_path = os.path.join(image_dir, f"{image_name}_ocr_result.txt")
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(str(response))
        print(f"\n💾 结果已保存到（文本格式）：{result_path}")


if __name__ == "__main__":
    main()
