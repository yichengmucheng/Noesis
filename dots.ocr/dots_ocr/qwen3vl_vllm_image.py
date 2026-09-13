# import time
# import cv2
# import base64
# import json
# from openai import OpenAI
 
# # 安装依赖库
# # pip install opencv-python==4.5.4.58 -i https://pypi.tuna.tsinghua.edu.cn/simple
# # pip install openai -i https://pypi.tuna.tsinghua.edu.cn/simple
 
# if __name__ == '__main__':
#     image_path = "/mnt/mydisk1/zhixinguangyu/sxt_home/DeepSeek-OCR-main/DeepSeek-OCR-master/DeepSeek-OCR-vllm/output/images/0_0.jpg" # 替换成你的图片地址
#     openai_api_key = "EMPTY"
#     openai_api_base = "http://127.0.0.1:7682/v1" # 替换成你的Qwen3-VL大模型部署地址
 
#     image = cv2.imread(image_path)
 
#     h, w, c = image.shape
#     resize_h = int(h / 3)
#     resize_w = int(w / 3)
#     image = cv2.resize(image, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST)
#     encoded_image_byte = cv2.imencode(".jpg", image)[1].tobytes()  # bytes类型
#     image_base64 = base64.b64encode(encoded_image_byte)
#     image_base64 = image_base64.decode("utf-8")  # str类型
 
#     try:
#         client = OpenAI(
#             api_key=openai_api_key,
#             base_url=openai_api_base,
#         )
#         t1 = time.time()
#         messages = [
#             # {
#             #     "role": "system",
#             #     "content": [{"type": "text", "text": "You are a helpful assistant."}]
#             # },
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                             "url": "data:image/jpeg;base64,%s" % str(image_base64)
#                         },
#                     },
#                     {"type": "text", "text": "OCR识别思维导图中的文本，按照人类的阅读顺序，用markdown表示节点之间的层级关系"},
#                     # {"type": "text", "text": "获得图片中所有人的坐标？"},
#                     # {"type": "text", "text": "图片中所有车辆的坐标？"},
#                 ],
#             }
#         ]
#         completion = client.chat.completions.create(
#             # model="Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
#             # model="Qwen/Qwen2.5-VL-3B-Instruct",
#             # model="Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
#             model="qwen3-vl-8b-model",
#             messages=messages,
#             temperature=0.7,
#             top_p=0.8,
#             max_tokens=1228,
#             extra_body={
#                 "repetition_penalty": 1.05,
#             },
#         )
#         t2 = time.time()
#         t = t2 - t1
#         content = completion.choices[0].message.content
#         print("耗时：", t)
#         print("content:", content)
 
#         client.close()
#     except Exception as e:
#         print(e)

import fitz  # PyMuPDF
from PIL import Image
import io
from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
import os
from pathlib import Path


class DocumentParser:
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.pptx', '.txt', '.jpg', '.png']
    
    def parse_pdf(self, file_path):
        """
        解析PDF整页渲染图像并保存到输出目录，返回PIL图像列表
        """
        doc = fitz.open(file_path)
        images = []
        out_dir = Path(__file__).parent / "output" / "pages" / Path(file_path).stem
        out_dir.mkdir(parents=True, exist_ok=True)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            images.append(img)
            try:
                img.convert("RGB").save(out_dir / f"page_{page_num}.jpg")
            except Exception:
                pass
        doc.close()
        return images
    def create_multimodal_message(self, text_content, images=None):
        """
        创建符合Qwen3-VL格式的多模态消息
        """
        content = []
        
        # 添加文本内容
        if text_content:
            content.append({"type": "text", "text": text_content})
        
        # 添加图像内容
        if images:
            for image in images:
                if isinstance(image, str):  # 图像路径
                    content.append({"type": "image", "image": image})
                else:  # PIL图像对象
                    # 这里需要将图像保存为临时文件或进行base64编码
                    content.append({"type": "image", "image": image})
        
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        return messages

    def extract_document_content(self, file_path):
        """
        根据文件类型提取内容
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        images = None
        if file_ext == '.pdf':
            images = self.parse_pdf(file_path)
        prompt = f"""
            OCR this Mind Map image to markdown.
            1. Constraints:
                - The output text must be the original text from the image, with no translation.
                - All layout elements must be sorted according to human reading order.
            2. Final Output: The entire output must be a single markdown string.
            ---输出示例---
            # EGR冷却器漏水

            ## 内部漏水

            - 冷却芯焊接质量问题
            - 冷却芯原始缺陷
            - 使用不当
                - 冷却液流量不足
                - 无冷却液（干烧）
                - EGR冷却器振动过大
                - 水压过大
                - 冷却芯被腐蚀

            ---

            编制：李廷强 校对：廖跃辉 审核：黎华文 批准：谭达辉
            """
        messages = self.create_multimodal_message(prompt, images)
        return messages

_MODEL = None
_PROCESSOR = None
_MODEL_PATH = '/mnt/mydisk1/zhixinguangyu/sxt_home/dots.ocr-master/weights/Qwen3-VL-8B-Instruct'

class Qwen3VLModel:
    @staticmethod
    def _ensure_loaded():
        global _MODEL, _PROCESSOR
        if _MODEL is None or _PROCESSOR is None:
            path = os.environ.get('QWEN3VL_MODEL_PATH', _MODEL_PATH)
            _MODEL = Qwen3VLForConditionalGeneration.from_pretrained(
                path, dtype="auto", device_map="auto", local_files_only=True
            )
            _PROCESSOR = AutoProcessor.from_pretrained(path, local_files_only=True)

    @staticmethod
    def inference(file_path):
        Qwen3VLModel._ensure_loaded()
        document_parser = DocumentParser()
        messages = document_parser.extract_document_content(file_path)
        inputs = _PROCESSOR.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        # Inference: Generation of the output
        generated_ids = _MODEL.generate(**inputs, max_length=None,max_new_tokens=10000)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = _PROCESSOR.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # 获取项目根目录
        BASE_DIR = Path(__file__).parent
        output_dir = BASE_DIR / "output"
        output_dir.mkdir(exist_ok=True)
        # 保存输出到文件
        output_file = output_dir / "qwen3vl_output.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_text[0])
            
        return output_text

def qwen3vl_parse(file_path):
    return Qwen3VLModel.inference(file_path)

if __name__ == "__main__":
    qwen3vl_parse('/mnt/mydisk1/zhixinguangyu/sxt_home/dots.ocr-master/demo/FTA/5.4 FTA-高压油管漏油 .pdf')
    
