import os
import random
from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
from utils.config_handler import  agent_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger



user_ids = ["1001","1002"]
month_arr = ["2026-03"]
rag = RagSummarizeService()
external_data = {}

@tool(description="从向量数据库中检索参考资料")
def rag_summarize(query:str)->str:
    return rag.rag_summarize(query)

@tool(description="获取城市的天气")
def get_weather(city:str)->str:
    return f"城市{city}为晴天"

@tool(description="获取用户所在城市")
def get_user_location()->str:
    return  random.choice(["杭州"])

@tool(description="获取用户ID")
def get_user_id()->str:
    return random.choice(user_ids)

@tool(description="获取当前时间")
def get_current_month() -> str:
    return random.choice(month_arr)

def generate_external_data():
    if not external_data:
        external_data_path = get_abs_path(agent_conf["external_data_path"])
    if not os.path.exists(external_data_path):
        raise  FileNotFoundError(f"外部数据文件{external_data_path}不存在")
    with open(external_data_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            arr:list[str] = line.strip().split(",")
            user_id: str = arr[0].replace(" ","")
        if user_id not in external_data:
            external_data[user_id] = {}
        external_data[user_id] = {

        }
@tool(description="从外部系统获取用户使用记录，如果未检索到返回空字符串")
def fetch_external_data(user_id:str,month:str)->str:
    generate_external_data()

    try :
        return external_data[user_id][month]
    except KeyError:
        logger.warning(f"未能检索到{user_id}在{month}的使用数据")
    return ""

@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"

if __name__ == "__main__":
    print(fetch_external_data("1001","2025-01"))
