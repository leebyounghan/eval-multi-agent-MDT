import json

def read_file(dataname = "mdt", few_shot = "zero"):
    text_file = []

    with open(f"../data/{dataname}_test.jsonl") as files :
        for line in files :
            text_file.append(json.loads(line))

    return text_file

def load_qa_data(name, few_shot):
    if name == 'medqa':
        pass
    # elif name == 'medbullets':
    #     qa_dataset = read_file(name, few_shot)
    # elif name == 'jama':
    #     qa_dataset = read_file(name, few_shot)
    elif name == 'mdt':
        qa_dataset = read_file(name, few_shot)

    return qa_dataset
    