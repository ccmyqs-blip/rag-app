import os
import hashlib


from  utils.logger_handler import logger
from langchain_core .documents import Document
from langchain_community.document_loaders import TextLoader,PyPDFLoader

def get_file_mad5_hex(file_path:str): #获取文件md5的十六进制字符串
    if not os.path.exists(file_path):
        logger.error(f"md5计算：文件 {file_path} 不存在")
        return
    if not os.path.isfile(file_path):
        logger.error(f"md5计算：文件 {file_path} 不是文件")

    md5_obj = hashlib.md5()
    chunk_size = 4096  #4KB分片，避免文件过大爆内存
    try:
        with open(file_path, "rb") as f: #必须二进制读取
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)

            md5_hex = md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error(f"文件{file_path}计算md5失败，e")
        return None


def listdir_with_allowed_type(path:str,allowed_types:tuple[str]): #返回文件夹中受允许的文件列表
    files = []
    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type]{path}不是文件夹")
        return allowed_types
    for f in os.listdir(path):
        if f.endswith(allowed_types):
            files.append(os.path.join(path, f))
    return tuple(files)


def pdf_loader(file_path:str,password=None)->list[Document]:
    return PyPDFLoader(file_path,password).load()

def txt_loader(file_path :str)->list[Document]:
    return TextLoader(file_path,encoding="utf-8").load()