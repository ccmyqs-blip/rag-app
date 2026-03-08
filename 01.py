from langchain.agents import create_agent
from langchain_community.chat_models import ChatTongyi
from langchain_core.tools import tool


@tool
agent = create_agent(
    model = ChatTongyi(model = "qwen3-max"),
    tools = [],
    middleware=[]

)

for chunk in agent.stream({
        "messages":[{"role":"user","content":"查询天气"}]},
    stream_mode="values"
):
    latest_messages = chunk["messages"]
    if latest_messages.content:
        print(type(latest_messages),__name__,latest_messages.content)

    try:
        if latest_messages.tool_calls:
            print(f"工具调用：{[tc['name']for tc in latest_messages.tool_calls]}")
    except AttributeError as e:
        pass

