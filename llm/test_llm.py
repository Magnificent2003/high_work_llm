from llm import LLMInvoker

results = [
    {
        "path": "./test/test_0001.jpg",
        "description": "图示桥梁中有16条裂缝，其中4条裂缝为短裂缝，4条裂缝长度适中，8条裂缝为长裂缝"
    },
    {
        "path": "./test/test_0002.jpg",
        "description": "图示桥梁中有12条裂缝，其中3条裂缝为短裂缝，6条裂缝长度适中，3条裂缝为长裂缝"
    },
]

def main():
    
    AI = LLMInvoker()
    text = AI.chat(f"作为一名桥梁工程领域的专家，给出运维决策：{results[0]['description']}")
    
    print(text)

if __name__ == "__main__":
    main()
