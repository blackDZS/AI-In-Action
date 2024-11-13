# 基于大模型的RAG应用评估方法

### 评估指标
在RAG流程中，主要包括三个核心部分：问题（Query）、检索到的文档（Context）以及模型生成的答案（Answer）。在评估过程中，我们还需要真实答案（Ground Truth）作为基准。在RAG应用中，我们关注两个关键点：其一是检索到的文档（Context），其二是基于检索到的文档所生成的答案（Answer）。下图1展示了这两个部分设置的评估指标，其中左侧列出了与`Answer`相关的指标，右侧则呈现了与`Context`相关的指标。指标计算方法可以参考[RAGAS Metrics](https://docs.ragas.io/en/stable/concepts/metrics/index.html)

- **Context**
    - Context Precision: 根据Context和Ground Truth计算检索到的Context准确率
    - Context Recall: 根据Context和Ground Truth计算检索到的Context召回率
    - Context Entities Recall: 根据Context和Ground Truth计算检索到的Context 中Entities的召回率

- **Answer**
    - Answer Faithfulness: 根据Context和Answer计算Answer是否来源于Context
    - Answer Semantic Similarity: 根据Ground Truth和Answer计算Answer与Ground Truth的语义相似性(使用Embedding向量计算)
    - Answer Correctness: 根据Ground Truth和Answer计算Answer准确率(使用LLM判断)


>更多内容请参考：[RAG应用开发指南（二）：如何评估与优化RAG应用性能](https://blackdzs.github.io/p/rag%E5%BA%94%E7%94%A8%E5%BC%80%E5%8F%91%E6%8C%87%E5%8D%97%E4%BA%8C%E5%A6%82%E4%BD%95%E8%AF%84%E4%BC%B0%E4%B8%8E%E4%BC%98%E5%8C%96rag%E5%BA%94%E7%94%A8%E6%80%A7%E8%83%BD/)

