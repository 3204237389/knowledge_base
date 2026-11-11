import json
import logging
import time
import zipfile
from pathlib import Path

import requests
from openai import api_key

from config.mineru_config import mineru_config
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, FileProcessingError, PdfConversionError
from processor.import_processor.state import ImportGraphState


class NodePDFToMD(BaseNode):
    """
    PDF 转 Markdown 节点：PDF结构化解析
    """

    name = "node_pdf_to_md"

    def process(self, state: ImportGraphState):
        logging.info(f"{self.name}节点开始执行了...")

        pdf_path_obj,output_dir_obj = self._step_1_validate_paths(state)

        zip_url = self._step_2_upload_and_poll(pdf_path_obj)
        print(f"已经获取下载url:{zip_url}")

        md_path = self._step_3_download_and_extract(zip_url, output_dir_obj, pdf_path_obj.stem)

        with open(md_path, "r",encoding="utf-8") as md_file:
            md_content = md_file.read()

        state["md_content"] = md_content
        state["md_path"] = md_path
        
        return state

    def _step_1_validate_paths(self, state:ImportGraphState):
        """
        校验文件
        :param state:
        :return:
        """
        pdf_path = state["pdf_path"]
        if not pdf_path:
            raise StateFieldError(field_name="pdf_path",expected_type=str)
        file_dir = state.get("file_dir")
        if not file_dir:
            raise StateFieldError(field_name="file_dir",expected_type=str)

        pdf_path_obj = Path(pdf_path)
        output_dir_obj = Path(file_dir)

        if not pdf_path_obj.exists():
            raise FileProcessingError(message=f"输入文件不存在：{pdf_path}")
        if not output_dir_obj.exists():
            raise FileProcessingError(message=f"输出目录不存在：{file_dir}")

        return pdf_path_obj,output_dir_obj

    def _step_2_upload_and_poll(self, pdf_path_obj: Path):
        """
        上传并获取下载链接
        :param pdf_path_obj:
        :return:
        """
        logging.info(f"_step_2_upload_and_poll上传文件到mineru服务器...")

        api_token = mineru_config.api_token
        base_url = mineru_config.base_url
        if not api_token:
            raise FileProcessingError(message="api_token未设置")
        if not base_url:
            raise FileProcessingError(message="base_url未设置")

        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_token}"
        }
        date = {
            "files": [
                {"name": pdf_path_obj.name}
            ],
            "model_version": "vlm"
        }
        url = f"{base_url}/file-urls/batch"
        requests.post(url, headers=header, json=date)
        response = requests.post(url, headers=header, json=date)
        if response.status_code != 200:
            raise FileProcessingError(message=f"申请上传文件失败{response.text}")
        result = response.json()
        if result.get("code") != 0:
            raise FileProcessingError(message=f"申请上传文件失败{result.get('message')}")
        batch_id = result["data"]["batch_id"]
        singed_url = result["data"]["file_urls"][0]

        with open(pdf_path_obj, "rb") as pdf_file:
            res_upload = requests.put(singed_url, data=pdf_file)
            if res_upload.status_code != 200:
                raise PdfConversionError(f"文件上传失败：状态码：{res_upload.status_code}，响应结果：{res_upload}")
            self.logger.info(f"文件上传成功！")

        poll_url = f"{base_url}/extract-results/batch/{batch_id}"
        start_time = time.time()
        timeout_seconds = 600
        poll_interval = 3

        while True:
            and_time = time.time() - start_time
            if and_time > timeout_seconds:
                raise FileProcessingError(message="获得下载地址超时")
            try:
                res_poll = requests.get(url=poll_url, headers=header, timeout=10)
            except Exception as e:
                self.logger.error(f"轮询接口异常：{e}")
                time.sleep(poll_interval)
                continue

            if res_poll.status_code != 200:
                raise PdfConversionError(f"【任务轮询】HTTP请求失败，状态码：{res_poll.status_code}，响应内容：{res_poll}")

            poll_data = res_poll.json()

            if poll_data.get("code") != 0:
                raise PdfConversionError(f"【任务轮询】业务错误，返回数据：{poll_data}")

            extract_results = poll_data["data"]["extract_result"]
            extract_result = extract_results[0]
            extract_state = extract_result["state"]

            if extract_state == "done":
                ful_zip_url = extract_result["full_zip_url"]
                return ful_zip_url
            elif extract_state == "failed":
                err_msg = extract_state.get("err_msg", "未知错误，无具体信息")
                raise PdfConversionError(f"【任务轮询】解析任务失败！batch_id：{batch_id}，错误信息：{err_msg}")

            else:
                self.logger.info(
                    f"【任务轮询】处理中... 已耗时{int(and_time)}s，状态：{extract_state}， batch_id：{batch_id}")

    def _step_3_download_and_extract(self, zip_url: str, output_dir_obj: Path, pdf_stem: str) -> str:
        logging.info(f"_step_3_download_and_extract下载解压并改名")

        response = requests.get(zip_url)
        if response.status_code != 200:
            raise FileProcessingError(message="获得下载文件失败：{response.text}")

        zip_save_path = output_dir_obj / f"{pdf_stem}_result.zip"
        with open(zip_save_path, 'wb') as f:
            f.write(response.content)

        extract_target_dir = output_dir_obj / pdf_stem
        extract_target_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_save_path, "r") as zip_file_obj:
            zip_file_obj.extractall(extract_target_dir)
        self.logger.info(f"【ZIP解压】ZIP解压完成，解压目录：{extract_target_dir}")

        self.logger.info(f"【MD重命名】找到MinerU生成的full.md文件")
        target_md_file = extract_target_dir / "full.md"
        self.logger.info(f"【MD重命名】开始将full.md文件进行重命名")
        new_md_path = target_md_file.with_name(f"{pdf_stem}.md")
        target_md_file.rename(new_md_path)
        self.logger.info(f"【MD重命名】重命名成功，文件名：{pdf_stem}.md")

        return str(new_md_path.absolute())
