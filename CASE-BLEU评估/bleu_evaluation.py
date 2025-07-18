import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import jieba

def calculate_bleu(text_a, text_b):
    """
    计算两个文本之间的BLEU分数
    
    参数:
        text_a (str): 第一个文本
        text_b (str): 第二个文本
    
    返回:
        float: BLEU分数 (0-1之间)
    """
    # 下载必要的NLTK数据
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    # 对中文文本进行分词
    tokens_a = list(jieba.cut(text_a))
    tokens_b = list(jieba.cut(text_b))
    
    # 打印分词结果，用于调试
    print(f"文本A分词结果: {tokens_a}")
    print(f"文本B分词结果: {tokens_b}")
    
    # 使用SmoothingFunction处理零计数的情况
    smoothing = SmoothingFunction().method1
    
    # 计算BLEU分数，使用较小的n-gram权重
    # 对于短文本，我们主要关注1-gram和2-gram
    weights = (0.5, 0.5, 0, 0)  # 只使用1-gram和2-gram，权重各0.5
    
    # 将text_b作为参考，text_a作为候选
    score = sentence_bleu([tokens_b], tokens_a, 
                         weights=weights,
                         smoothing_function=smoothing)
    
    return score

def main():
    # 示例使用
    text_a = "今天天气真不错"
    text_b = "今天天气很好"
    
    bleu_score = calculate_bleu(text_a, text_b)
    print(f"\n文本A: {text_a}")
    print(f"文本B: {text_b}")
    print(f"BLEU分数: {bleu_score:.4f}")

if __name__ == "__main__":
    main() 