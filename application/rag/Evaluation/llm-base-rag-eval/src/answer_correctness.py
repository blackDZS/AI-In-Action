import json
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from typing import List, Dict

def format_prompt(output_schema: str, query: str, context: str, answer: str, ground_truth: str) -> ChatPromptTemplate:
    """Construct the prompt template."""
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            f"""
            Given a ground truth and an answer, classify statements into:
            TP, FP, FN based on {output_schema}.
            Do not return explanations, only the JSON classification.
            question: "{query}"
            answer: "{answer}"
            ground_truth: "{ground_truth}"
            classification:
            """
        ),
        HumanMessagePromptTemplate.from_template("{context}")
    ])


def get_confusion_matrix(prompt: ChatPromptTemplate, llm: ChatOpenAI, context: str) -> Dict:
    """Invoke the LLM with the prompt and extract the JSON response."""
    response = prompt | llm
    response_content = response.invoke(context).content
    cleaned_content = response_content.strip('```json').strip('```')
    return json.loads(cleaned_content)

def calculate_score(prediction: Dict) -> float:
    """Calculate the statement presence score."""
    tp, fp, fn = prediction.get("TP", []), prediction.get("FP", []), prediction.get("FN", [])
    score = len(tp) / (len(tp) + 0.5 * (len(fp) + len(fn))) if len(tp) > 0 else 0
    return score


def answer_correctness_score(query: str, context: List[str], answer: str, ground_truth: str, model_name: str) -> float:
    """Analyze answer correctness and return a score."""
    llm = ChatOpenAI(model=model_name)
    output_schema = """
    {"type": "object", "properties": {"TP": {"title": "Tp", "type": "array", "items": {"type": "object"}}, 
                                      "FP": {"title": "Fp", "type": "array", "items": {"type": "object"}}, 
                                      "FN": {"title": "Fn", "type": "array", "items": {"type": "object"}}}, 
     "required": ["TP", "FP", "FN"]}
    """
    formatted_prompt = format_prompt(output_schema, query, " ".join(context), answer, ground_truth)
    classification = get_confusion_matrix(formatted_prompt, llm, "\n".join(context))
    score = calculate_score(classification)
    return score


# Example usage:
if __name__ == "__main__":
    query = "介绍下艾菲尔铁塔"
    contexts = [
        "埃菲尔铁塔位于法国巴黎第七区，是世界著名建筑和巴黎城市地标。",
        "建成于1889年，以设计师居斯塔夫·埃菲尔命名。",
    ]
    ground_truth = """
    埃菲尔铁塔是巴黎和法国的文化象征，初名为“三百米塔”, 1889年为世界博览会而建。
    """
    answer = "埃菲尔铁塔位于法国巴黎第七区，初名为三百米塔，建成于1889年。"
    model_name = "gpt-4o-mini"
    
    score = answer_correctness_score(query, contexts, answer, ground_truth, model_name)
    print("Answer Correctness:", score)
