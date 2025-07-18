import dashscope
from dashscope import TextEmbedding
import os

# 通过环境变量获取API-KEY
api_key = os.getenv('DASHSCOPE_API_KEY')
dashscope.api_key = api_key

# 单个文本嵌入
def get_single_text_embedding():
    resp = TextEmbedding.call(
        model=TextEmbedding.Models.text_embedding_v1,
        input="这是要转换为嵌入向量的文本"
    )
    print(resp)

# 批量文本嵌入
def get_batch_text_embedding():
    resp = TextEmbedding.call(
        model=TextEmbedding.Models.text_embedding_v1,
        input=["文本1", "文本2", "文本3"]
    )
    print(resp)

if __name__ == '__main__':
    get_single_text_embedding()
    get_batch_text_embedding()