import asyncio
import os
import sys
from pathlib import Path
import json
from fastapi import FastAPI, HTTPException, Body
import re
# 导入parse_md函数
from parser import ocr_parse

# 创建FastAPI应用
app = FastAPI(
    title="DotsOCR服务",
    description="文档OCR识别服务API",
    version="1.0.0"
)

# 获取项目根目录
BASE_DIR = Path(__file__).parent

# POST方法：调用parser.py的main函数处理文件
@app.post("/api/parse-file")
async def parse_file(file_path: str = Body(..., embed=True)):
    """
    直接调用parser.py中的parse_md函数处理指定路径的文件
    
    Args:
        file_path: 待处理文件的路径
        
    Returns:
        包含解析结果的JSON响应
    """
    try:
        # 1.替换路径前缀，得到宿主机真实路径
        file_path = file_path.replace('/app/data/inputs', '/mnt/mydisk1/zhixinguangyu/sxt_home/LightRAG_test/data/inputs')
        # 2.验证文件是否存在
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"文件不存在: {file_path}")
        
        # 3.ocr_parse函数返回拼接后的完整markdown内容
        full_content = ocr_parse(file_path)
        # 4.过滤图片
        # content = re.sub(r'!\[.*?\]\(data:image/(png|jpg|jpeg|gif|bmp|webp);base64,[A-Za-z0-9+/=\\s]+\)', '', full_content)

        # 5.将content内容输出到文档中（检查测试）
        output_dir = BASE_DIR / "filter_img_output"
        output_dir.mkdir(exist_ok=True)
        
        input_filename = os.path.basename(file_path)
        output_filename = f"{os.path.splitext(input_filename)[0]}_ocr_result.md"
        output_path = output_dir / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_content)
        print(f"OCR结果已保存到: {output_path}")

        # 6.返回包含解析结果的字典
        return {
            "status": "success",
            "content": full_content
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@app.get("/")
async def root():
    """根路径"""
    return {"message": "DotsOCR服务"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7797, reload=False, workers=4)
    print("服务启动成功")