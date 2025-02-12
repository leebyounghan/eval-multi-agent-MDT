import os
import autogen
from autogen import AssistantAgent, UserProxyAgent, ConversableAgent
from openai import OpenAI
import json

def cot(query, seed):

    Decision_maker = ConversableAgent(
        name="Decision_maker",
        system_message="You are a medical expert who makes a decision. Please provide your decision as a short answer in the following format at the end of your response:\n[Rationale] rationale here\n\n[Answer] short answer here",
        llm_config={"config_list": [{"model": "gpt-4o", "seed": seed, "api_key": os.environ["OPENAI_API_KEY"]}]}, # "seed": 1, 
    )

    User = ConversableAgent(
        name="User",
        system_message="You are asking a medical question to medical expert.",
        llm_config={"config_list": [{"model": "gpt-4o", "seed": seed, "api_key": os.environ["OPENAI_API_KEY"]}]},
    )

    chat_result = User.initiate_chat(
        Decision_maker,
        message=query+"\n\nLet's think step by step.",
        summary_method="last_msg",
        max_turns=1,
    )
    decision = chat_result.chat_history[1]['content']
    return decision

def majority_vote(query, seed):

    experts = []
    for i in range(4):
        experts.append(ConversableAgent(
            name=f"Expert_{i+1}",
            system_message="You are a medical expert who makes a decision. Please provide your decision as a short answer in the following format at the end of your response:\n[Rationale] rationale here\n\n[Answer] short answer here",
            llm_config={"config_list": [{"model": "gpt-4o", "seed": seed+i, "api_key": os.environ["OPENAI_API_KEY"]}]}, # 
        ))

    User = ConversableAgent(
        name="User",
        system_message="You are asking a medical question to medical expert.",
        llm_config={"config_list": [{"model": "gpt-4o", "seed": seed, "api_key": os.environ["OPENAI_API_KEY"]}]},
    )

    expert_opinions = ""
    for i, expert in enumerate(experts):
        chat_result = User.initiate_chat(
            expert,
            message=query+"\n\nLet's think step by step.",
            summary_method="last_msg",
            max_turns=1,
        )
        expert_opinions += "[Expert "+str(i+1)+"]\n"+chat_result.chat_history[1]['content']+"\n\n"

    Summarizer = ConversableAgent(
        name="Summarizer",
        system_message="You are a medical expert who summarizes the opinions of other experts. Please provide the majority opinion as a short answer in the following format at the end of your response:\n[Rationale] rationale here\n\n[Answer] short answer here",
        llm_config={"config_list": [{"model": "gpt-4o", "seed": seed, "api_key": os.environ["OPENAI_API_KEY"]}]}, # "seed": 1, 
    )

    chat_result = User.initiate_chat(
        Summarizer,
        message=expert_opinions+"\n\nWhat is the majority opinion?",
        summary_method="last_msg",
        max_turns=1,
    )
    decision = chat_result.chat_history[1]['content']
    return expert_opinions, decision


def majority_vote_w_recruit(query, seed):
    # hiring phase
    User = ConversableAgent(
        name="User",
        system_message="You are asking a medical question to medical expert.",
        llm_config={"config_list": [{"model": "gpt-4o", "seed": seed, "api_key": os.environ["OPENAI_API_KEY"]}]},
    )

    Hiring_manager = ConversableAgent(
        name="Hiring_manager",
        system_message="You are a hiring manager to plan MDT meeting.",
        llm_config={"config_list": [{"model": "gpt-4o", "seed": seed, "api_key": os.environ["OPENAI_API_KEY"]}]},
    )

    chat_result = User.initiate_chat(
        Hiring_manager,
        message=query+"\n\nWho should be invited to the MDT meeting? Maximum 5 experts.",
        summary_method="reflection_with_llm",
        summary_args={"summary_prompt": "Just list the names of the experts who should be invited to the MDT meeting. Do not include any special characters such as '&' or'/'. List the names in the following format: 'Otolaryngologist|Medical Oncologist|Radiation Oncologist'."},
        max_turns=1,
    )

    experts = chat_result.summary.split('|')
    print(experts)
    expert_agents = []
    for i, expert in enumerate(experts):
        expert_agents.append(ConversableAgent(
            name='_'.join(expert.split('/')[0].split(' ')),
            system_message=f"You are a {expert} who makes a decision. Please provide your decision as a short answer in the following format at the end of your response:\n[Rationale] rationale here\n\n[Answer] short answer here",
            llm_config={"config_list": [{"model": "gpt-4o", "seed": seed+i, "api_key": os.environ["OPENAI_API_KEY"]}]},
        ))

    User = ConversableAgent(
        name="User",
        system_message="You are asking a medical question to medical expert.",
        llm_config={"config_list": [{"model": "gpt-4o", "seed": seed, "api_key": os.environ["OPENAI_API_KEY"]}]},
    )

    expert_opinions = ""
    for i, expert in enumerate(expert_agents):
        chat_result = User.initiate_chat(
            expert,
            message=query+"\n\nLet's think step by step.",
            summary_method="last_msg",
            max_turns=1,
        )
        expert_opinions += "["+chat_result.chat_history[1]['name']+"]\n"+chat_result.chat_history[1]['content']+"\n\n"

    Summarizer = ConversableAgent(
        name="Summarizer",
        system_message="You are a medical expert who summarizes the opinions of other experts. Please provide the majority opinion as a short answer in the following format at the end of your response:\n[Rationale] rationale here\n\n[Answer] short answer here",
        llm_config={"config_list": [{"model": "gpt-4o", "seed": seed, "api_key": os.environ["OPENAI_API_KEY"]}]}, # "seed": 1, 
    )

    chat_result = User.initiate_chat(
        Summarizer,
        message=expert_opinions+"\n\nWhat is the majority opinion?",
        summary_method="last_msg",
        max_turns=1,
    )
    decision = chat_result.chat_history[1]['content']
    return expert_opinions, decision

def sot(query, seed):
    Simulator = ConversableAgent(
        name="Simulator",
        system_message='''
        You are a medical expert. Imagine experts with differing opinions are participating in a MDT meeting.
        Create a simulation of a conversational debate between experts with differing opinions based on given patient information.
        The goal is for the medical experts to find consensus on a final answer through a series of questions and answers.

        Max turns is 10.

        # Steps
        1. Create 3-5 expert personas with different specialties and initial opinions
        2. Simulate a discussion where each expert:
           - Presents their initial assessment
           - Asks questions about aspects relevant to their specialty
           - Responds to other experts' concerns
           - Adjusts their position based on new information
        3. Guide the discussion toward consensus by:
           - Highlighting areas of agreement
           - Addressing key points of disagreement
           - Synthesizing different perspectives
        4. Conclude with a clear consensus recommendation or decision

        # Chat format
        [Expert_name]
        content

        [Expert_name] 
        content
        ...
        ''',
        llm_config={"config_list": [{"model": "gpt-4o", "seed": seed, "api_key": os.environ["OPENAI_API_KEY"]}]},
    )

    User = ConversableAgent(
        name="User",
        system_message="You are asking a medical question to medical expert.",
        llm_config={"config_list": [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}]},
    )

    chat_result = User.initiate_chat(
        Simulator,
        message=query+"\n\nLet's simulate a MDT meeting.",
        summary_method="last_msg",
        max_turns=1,
    )
    chat_history = chat_result.chat_history[1]['content']

    Summarizer = ConversableAgent(
        name="Summarizer",
        system_message="You are a medical expert who summarizes the opinions of other experts. Please provide the final decision from the meeting as a short answer in the following format at the end of your response:\n[Rationale] rationale here\n\n[Answer] short answer here",
        llm_config={"config_list": [{"model": "gpt-4o", "seed": seed, "api_key": os.environ["OPENAI_API_KEY"]}]}, # "seed": 1, 
    )

    chat_result = User.initiate_chat(
        Summarizer,
        message=chat_history+"\n\nWhat is the final decision from the meeting?",
        summary_method="last_msg",
        max_turns=1,
    )
    decision = chat_result.chat_history[1]['content']
    return chat_history, decision
    



def group_chat(query, turns, seed):

    expert_agents = []
    for i in range(4):
        expert_agents.append(ConversableAgent(
            name='Expert_'+str(i+1),
            system_message=f"You are an medical expert who is involved in the MDT to make a decision. Engage with other participants and be concise.",
            llm_config={"config_list": [{"model": "gpt-4o", "seed": seed+i, "api_key": os.environ["OPENAI_API_KEY"]}]},
        ))

    # meeting phase
    from autogen import GroupChat
    from autogen import GroupChatManager

    User = ConversableAgent(
        name="User",
        system_message="You are asking a medical question to medical expert.",
        llm_config={"config_list": [{"model": "gpt-4o", "seed": seed, "api_key": os.environ["OPENAI_API_KEY"]}]},
    )

    group_chat = GroupChat(
        agents=expert_agents,
        messages=[],
        max_round=turns,
    )

    group_chat_manager = GroupChatManager(
        groupchat=group_chat,
        llm_config={"config_list": [{"model": "gpt-4o", "seed": seed, "api_key": os.environ["OPENAI_API_KEY"]}]},
    )

    # Use last_message instead of reflection_with_llm to keep full chat history
    chat_result = User.initiate_chat(
        group_chat_manager,
        message=query+"\n\nLet's have a discussion.",
        summary_method="last_msg",
    )

    # Get full chat history from the group chat manager
    chat_history = "".join(["["+group_chat_manager.groupchat.messages[i]['name']+"]\n"+ group_chat_manager.groupchat.messages[i]['content'] + "\n\n" for i in range(1, len(group_chat_manager.groupchat.messages))])

    Summarizer = ConversableAgent(
        name="Summarizer",
        system_message="You are a medical expert who summarizes the opinions of other experts. Please provide the final decision from the meeting as a short answer in the following format at the end of your response:\n[Rationale] rationale here\n\n[Answer] short answer here",
        llm_config={"config_list": [{"model": "gpt-4o", "seed": seed, "api_key": os.environ["OPENAI_API_KEY"]}]}, # "seed": 1, 
    )

    chat_result = User.initiate_chat(
        Summarizer,
        message=chat_history+"\n\nWhat is the final decision from the meeting?",
        summary_method="last_msg",
        max_turns=1,
    )
    decision = chat_result.chat_history[1]['content']
    return chat_history, decision


def group_chat_w_recruit(query, turns, seed):
    # hiring phase
    User = ConversableAgent(
        name="User",
        system_message="You are asking a medical question to medical expert.",
        llm_config={"config_list": [{"model": "gpt-4o", "seed": seed, "api_key": os.environ["OPENAI_API_KEY"]}]},
    )

    Hiring_manager = ConversableAgent(
        name="Hiring_manager",
        system_message="You are a hiring manager to plan MDT meeting.",
        llm_config={"config_list": [{"model": "gpt-4o", "seed": seed, "api_key": os.environ["OPENAI_API_KEY"]}]},
    )

    chat_result = User.initiate_chat(
        Hiring_manager,
        message=query+"\n\nWho should be invited to the MDT meeting? Maximum 5 experts.",
        summary_method="reflection_with_llm",
        summary_args={"summary_prompt": "Just list the names of the experts who should be invited to the MDT meeting. Do not include any special characters such as '&' or'/'. List the names in the following format: 'Otolaryngologist|Medical Oncologist|Radiation Oncologist'."},
        max_turns=1,
    )

    experts = chat_result.summary.split('|')
    print(experts)
    expert_agents = []
    for i, expert in enumerate(experts):
        expert_agents.append(ConversableAgent(
            name='_'.join(expert.split('/')[0].split(' ')),
            system_message=f"You are an {expert} who is involved in the MDT to make a decision. Engage with other participants and be concise.",
            llm_config={"config_list": [{"model": "gpt-4o", "seed": seed+i, "api_key": os.environ["OPENAI_API_KEY"]}]},
        ))

    # expert_agents = []
    # for i in range(4):
    #     expert_agents.append(ConversableAgent(
    #         name='Expert_'+str(i+1),
    #         system_message=f"You are an medical expert who is involved in the MDT to make a decision. Engage with other participants and be concise.",
    #         llm_config={"config_list": [{"model": "gpt-4o", "seed": seed+i, "api_key": os.environ["OPENAI_API_KEY"]}]},
    #     ))

    # meeting phase
    from autogen import GroupChat
    from autogen import GroupChatManager

    User = ConversableAgent(
        name="User",
        system_message="You are asking a medical question to medical expert.",
        llm_config={"config_list": [{"model": "gpt-4o", "seed": seed, "api_key": os.environ["OPENAI_API_KEY"]}]},
    )

    group_chat = GroupChat(
        agents=expert_agents,
        messages=[],
        max_round=turns,
    )

    group_chat_manager = GroupChatManager(
        groupchat=group_chat,
        llm_config={"config_list": [{"model": "gpt-4o", "seed": seed, "api_key": os.environ["OPENAI_API_KEY"]}]},
    )

    # Use last_message instead of reflection_with_llm to keep full chat history
    chat_result = User.initiate_chat(
        group_chat_manager,
        message=query+"\n\nLet's have a discussion.",
        summary_method="last_msg",
    )

    # Get full chat history from the group chat manager
    chat_history = "".join(["["+group_chat_manager.groupchat.messages[i]['name']+"]\n"+ group_chat_manager.groupchat.messages[i]['content'] + "\n\n" for i in range(1, len(group_chat_manager.groupchat.messages))])

    Summarizer = ConversableAgent(
        name="Summarizer",
        system_message="You are a medical expert who summarizes the opinions of other experts. Please provide the final decision from the meeting as a short answer in the following format at the end of your response:\n[Rationale] rationale here\n\n[Answer] short answer here",
        llm_config={"config_list": [{"model": "gpt-4o", "seed": seed, "api_key": os.environ["OPENAI_API_KEY"]}]}, # "seed": 1, 
    )

    chat_result = User.initiate_chat(
        Summarizer,
        message=chat_history+"\n\nWhat is the final decision from the meeting?",
        summary_method="last_msg",
        max_turns=1,
    )
    decision = chat_result.chat_history[1]['content']
    return chat_history, decision



def group_chat_w_restricted_role(query, turns, seed):
    # hiring phase
    User = ConversableAgent(
        name="User",
        system_message="You are asking a medical question to medical expert.",
        llm_config={"config_list": [{"model": "gpt-4o", "seed": seed, "api_key": os.environ["OPENAI_API_KEY"]}]},
    )

    Hiring_manager = ConversableAgent(
        name="Hiring_manager",
        system_message="You are a hiring manager to plan MDT meeting.",
        llm_config={"config_list": [{"model": "gpt-4o", "seed": seed, "api_key": os.environ["OPENAI_API_KEY"]}]},
    )

    chat_result = User.initiate_chat(
        Hiring_manager,
        message=query+"\n\nWho should be invited to the MDT meeting? Maximum 2 experts.",
        summary_method="reflection_with_llm",
        summary_args={"summary_prompt": "Just list the names of the experts who should be invited to the MDT meeting. Do not include any special characters such as '&' or'/'. List the names in the following format: 'Otolaryngologist|Medical Oncologist|Radiation Oncologist'."},
        max_turns=1,
    )

    experts = chat_result.summary.split('|')
    experts = [expert+"_1" for expert in experts] + [expert+"_2" for expert in experts]

    expert_agents = []
    for i, expert in enumerate(experts):
        expert_agents.append(ConversableAgent(
            name='_'.join(expert.split('/')[0].split(' ')),
            system_message=f"You are an {expert} who is involved in the MDT to make a decision. Engage with other participants and be concise.",
            llm_config={"config_list": [{"model": "gpt-4o", "seed": seed+i, "api_key": os.environ["OPENAI_API_KEY"]}]},
        ))


    # meeting phase
    from autogen import GroupChat
    from autogen import GroupChatManager

    User = ConversableAgent(
        name="User",
        system_message="You are asking a medical question to medical expert.",
        llm_config={"config_list": [{"model": "gpt-4o", "seed": seed, "api_key": os.environ["OPENAI_API_KEY"]}]},
    )

    group_chat = GroupChat(
        agents=expert_agents,
        messages=[],
        max_round=turns,
    )

    group_chat_manager = GroupChatManager(
        groupchat=group_chat,
        llm_config={"config_list": [{"model": "gpt-4o", "seed": seed, "api_key": os.environ["OPENAI_API_KEY"]}]},
    )

    # Use last_message instead of reflection_with_llm to keep full chat history
    chat_result = User.initiate_chat(
        group_chat_manager,
        message=query+"\n\nLet's have a discussion.",
        summary_method="last_msg",
    )

    # Get full chat history from the group chat manager
    chat_history = "".join(["["+group_chat_manager.groupchat.messages[i]['name']+"]\n"+ group_chat_manager.groupchat.messages[i]['content'] + "\n\n" for i in range(1, len(group_chat_manager.groupchat.messages))])

    Summarizer = ConversableAgent(
        name="Summarizer",
        system_message="You are a medical expert who summarizes the opinions of other experts. Please provide the final decision from the meeting as a short answer in the following format at the end of your response:\n[Rationale] rationale here\n\n[Answer] short answer here",
        llm_config={"config_list": [{"model": "gpt-4o", "seed": seed, "api_key": os.environ["OPENAI_API_KEY"]}]}, # "seed": 1, 
    )

    chat_result = User.initiate_chat(
        Summarizer,
        message=chat_history+"\n\nWhat is the final decision from the meeting?",
        summary_method="last_msg",
        max_turns=1,
    )
    decision = chat_result.chat_history[1]['content']
    return chat_history, decision



def majority_vote_w_recruit_and_group_chat(query, turns, seed):
    # hiring phase
    User = ConversableAgent(
        name="User",
        system_message="You are asking a medical question to medical expert.",
        llm_config={"config_list": [{"model": "gpt-4o", "seed": seed, "api_key": os.environ["OPENAI_API_KEY"]}]},
    )

    Hiring_manager = ConversableAgent(
        name="Hiring_manager",
        system_message="You are a hiring manager to plan MDT meeting.",
        llm_config={"config_list": [{"model": "gpt-4o", "seed": seed, "api_key": os.environ["OPENAI_API_KEY"]}]},
    )

    chat_result = User.initiate_chat(
        Hiring_manager,
        message=query+"\n\nWho should be invited to the MDT meeting? Maximum 5 experts.",
        summary_method="reflection_with_llm",
        summary_args={"summary_prompt": "Just list the names of the experts who should be invited to the MDT meeting. Do not include any special characters such as '&' or'/'. List the names in the following format: 'Otolaryngologist|Medical Oncologist|Radiation Oncologist'."},
        max_turns=1,
    )

    experts = chat_result.summary.split('|')
    print(experts)
    expert_agents = []
    for i, expert in enumerate(experts):
        expert_agents.append(ConversableAgent(
            name='_'.join(expert.split('/')[0].split(' ')),
            system_message=f"You are a {expert} who makes a decision. Please provide your decision as a short answer in the following format at the end of your response:\n[Rationale] rationale here\n\n[Answer] short answer here",
            llm_config={"config_list": [{"model": "gpt-4o", "seed": seed+i, "api_key": os.environ["OPENAI_API_KEY"]}]},
        ))

    User = ConversableAgent(
        name="User",
        system_message="You are asking a medical question to medical expert.",
        llm_config={"config_list": [{"model": "gpt-4o", "seed": seed, "api_key": os.environ["OPENAI_API_KEY"]}]},
    )


    # Create the JSON structure as a Python dictionary
    messages = [
        {"content": f"{query}\n\nLet's think step by step..", "role": "user", "name": expert_agents[0].name}
    ]
    for i, expert in enumerate(expert_agents):
        chat_result = User.initiate_chat(
            expert,
            message=query+"\n\nLet's think step by step.",
            summary_method="last_msg",
            max_turns=1,
        )
        messages.append({"content": chat_result.chat_history[1]['content'], "role": "user", "name": chat_result.chat_history[1]['name']})
    messages.append({"content": "Okay. Let's have a discussion.", "role": "user", "name": expert_agents[0].name})

    for i, expert in enumerate(experts):
        expert_agents[i].update_system_message(f"You are an {expert} who is involved in the MDT to make a decision. Engage with other participants and be concise.")

    from autogen import GroupChat
    from autogen import GroupChatManager

    group_chat = GroupChat(
        agents=expert_agents,
        messages=[],
        max_round=turns,
    )

    group_chat_manager = GroupChatManager(
        groupchat=group_chat,
        llm_config={"config_list": [{"model": "gpt-4o", "seed": seed, "api_key": os.environ["OPENAI_API_KEY"]}]},
    )


    # Convert the dictionary to a JSON string
    import json
    previous_state = json.dumps(messages, ensure_ascii=False)
    last_agent, last_message = group_chat_manager.resume(messages=previous_state)


    # Use last_message instead of reflection_with_llm to keep full chat history
    chat_result = last_agent.initiate_chat(
        group_chat_manager,
        message=last_message,
        clear_history=False,
        summary_method="last_msg",
    )

    # Get full chat history from the group chat manager
    chat_history = "".join(["["+group_chat_manager.groupchat.messages[i]['name']+"]\n"+ group_chat_manager.groupchat.messages[i]['content'] + "\n\n" for i in range(1, len(group_chat_manager.groupchat.messages))])

    Summarizer = ConversableAgent(
        name="Summarizer",
        system_message="You are a medical expert who summarizes the opinions of other experts. Please provide the final decision from the meeting as a short answer in the following format at the end of your response:\n[Rationale] rationale here\n\n[Answer] short answer here",
        llm_config={"config_list": [{"model": "gpt-4o", "seed": seed, "api_key": os.environ["OPENAI_API_KEY"]}]}, # "seed": 1, 
    )

    chat_result = User.initiate_chat(
        Summarizer,
        message=chat_history+"\n\nWhat is the final decision from the meeting?",
        summary_method="last_msg",
        max_turns=1,
    )
    decision = chat_result.chat_history[1]['content']
    return chat_history, decision



def group_chat_w_recruit_w_initial_error(query, turns, error, seed):

    # hiring phase
    User = ConversableAgent(
        name="User",
        system_message="You are asking a medical question to medical expert.",
        llm_config={"config_list": [{"model": "gpt-4o", "seed": seed, "api_key": os.environ["OPENAI_API_KEY"]}]},
    )

    Hiring_manager = ConversableAgent(
        name="Hiring_manager",
        system_message="You are a hiring manager to plan MDT meeting.",
        llm_config={"config_list": [{"model": "gpt-4o", "seed": seed, "api_key": os.environ["OPENAI_API_KEY"]}]},
    )

    chat_result = User.initiate_chat(
        Hiring_manager,
        message=query+"\n\nWho should be invited to the MDT meeting? Maximum 5 experts.",
        summary_method="reflection_with_llm",
        summary_args={"summary_prompt": "Just list the names of the experts who should be invited to the MDT meeting. Do not include any special characters such as '&' or'/'. List the names in the following format: 'Otolaryngologist|Medical Oncologist|Radiation Oncologist'."},
        max_turns=1,
    )

    experts = chat_result.summary.split('|')
    experts = ['_'.join(expert.split('/')[0].split(' ')) for expert in experts]
    print(experts)
    expert_agents = []
    for i, expert in enumerate(experts):
        expert_agents.append(ConversableAgent(
            name=expert,
            system_message=f"You are an {expert} who is involved in the MDT to make a decision. Engage with other participants and be concise.",
            llm_config={"config_list": [{"model": "gpt-4o", "seed": seed+i, "api_key": os.environ["OPENAI_API_KEY"]}]},
        ))

    # meeting phase
    from autogen import GroupChat
    from autogen import GroupChatManager

    group_chat = GroupChat(
        agents=expert_agents,
        messages=[],
        max_round=turns,
    )

    group_chat_manager = GroupChatManager(
        groupchat=group_chat,
        llm_config={"config_list": [{"model": "gpt-4o", "seed": seed, "api_key": os.environ["OPENAI_API_KEY"]}]},
    )

    # Create the JSON structure as a Python dictionary
    messages = [
        {"content": f"{query}\n\nLet's have a discussion.", "role": "user", "name": experts[0]}
    ]
    if error == "gene_therapy":
        messages.append({"content": "This patient presents a highly complex and challenging case. Given the recent promising clinical outcomes associated with gene therapy for head and neck cancer, it is imperative that we consider its application for this patient.", "role": "user", "name": experts[0]})
    elif error == "CART":
        messages.append({"content": "This patient presents a highly complex and challenging case. Given the recent promising clinical outcomes associated with CAR-T therapy for head and neck cancer, it is imperative that we consider its application for this patient.", "role": "user", "name": experts[0]})        
    elif error == "transplantation":
        messages.append({"content": "This patient presents a highly complex and challenging case. Given the recent promising clinical outcomes associated with transplantation for head and neck cancer, it is imperative that we consider its application for this patient.", "role": "user", "name": experts[0]})        
    
    # Convert the dictionary to a JSON string
    import json
    previous_state = json.dumps(messages, ensure_ascii=False)
    last_agent, last_message = group_chat_manager.resume(messages=previous_state)


    # Use last_message instead of reflection_with_llm to keep full chat history
    chat_result = last_agent.initiate_chat(
        group_chat_manager,
        message=last_message,
        clear_history=False,
        summary_method="last_msg",
    )

    # Get full chat history from the group chat manager
    chat_history = "".join(["["+group_chat_manager.groupchat.messages[i]['name']+"]\n"+ group_chat_manager.groupchat.messages[i]['content'] + "\n\n" for i in range(1, len(group_chat_manager.groupchat.messages))])

    Summarizer = ConversableAgent(
        name="Summarizer",
        system_message="You are a medical expert who summarizes the opinions of other experts. Please provide the final decision from the meeting as a short answer in the following format at the end of your response:\n[Rationale] rationale here\n\n[Answer] short answer here",
        llm_config={"config_list": [{"model": "gpt-4o", "seed": seed, "api_key": os.environ["OPENAI_API_KEY"]}]}, # "seed": 1, 
    )

    chat_result = User.initiate_chat(
        Summarizer,
        message=chat_history+"\n\nWhat is the final decision from the meeting?",
        summary_method="last_msg",
        max_turns=1,
    )
    decision = chat_result.chat_history[1]['content']
    return chat_history, decision
