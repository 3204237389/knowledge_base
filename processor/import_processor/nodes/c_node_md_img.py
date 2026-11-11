import base64
import json
import logging
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Tuple, List, Dict, Deque

from Tools.scripts.generate_token import update_file
from aiohttp.web_fileresponse import content_type
from minio import Minio
from minio.deleteobjects import DeleteObject
from multipart import file_path

from config.lm_config import lm_config
from config.minio_config import minio_config
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, FileProcessingError
from processor.import_processor.state import ImportGraphState
from utils.llm_utils import get_llm_client
from utils.minio_utils import get_minio_client


class NodeMDImg(BaseNode):
    """
    MarkDown图片处理节点：多模态图片理解
    """

    name = "node_md_img"

    def process(self, state: ImportGraphState):
        logging.info(f"{self.name}节点开始执行了...")

        md_content, md_path_obj, images_dir = self._step_1_get_content(state)
        print("md_content:{md_content},md_path_obj:{md_path_obj},images_dir:{images_dir}")
        target_images = self._step2_scan_images(md_content, images_dir)
        print(f"target_images:{target_images}")

        summaries = self._step_3_generate_summaries(md_path_obj.stem, target_images)

        new_md_content = self._step_4_upload_and_replace(md_path_obj.stem, target_images, summaries, md_content)

        new_md_file_name = self._step_5_backup_new_md_file(state['md_path'], new_md_content)

        state["md_content"] = new_md_content
        state["md_path"] = new_md_file_name
        return state

    def _step_1_get_content(self, state):
        md_path = state.get("md_path")
        if not md_path:
            raise StateFieldError(field_name='md_path', expected_type=str)
        md_path_obj = Path(md_path)

        if not md_path_obj.exists():
            raise FileProcessingError(message=f"输入文件不存在:{md_path}")

        md_content = state.get("md_content")

        if not md_content:
            with open(md_path, "r", encoding="utf-8") as md_file:
                md_content = md_file.read()

        images_dir = md_path_obj.parent / "images"

        return md_content, md_path_obj, images_dir

    def _step2_scan_images(self, md_content, images_dir):
        target_images = []

        for image_file in os.listdir(images_dir):
            file_ext = os.path.splitext(image_file)[1].lower()

            if file_ext not in self.config.image_extensions:
                self.logger.warning(f"图片格式不支持，跳过：{image_file}")
                continue

            ima_path = images_dir / image_file
            context = self._find_image_in_md(md_content, image_file)

            if not context:
                self.logger.warning(f"图片未在MD中引用，跳过处理：{image_file}")
                continue

            target_images.append((image_file, ima_path, context))
        return target_images

    def _find_image_in_md(self, md_content: str, image_file: str, content_len: int = 100) -> Tuple[str, str]:
        """
            找到图片 并截取上下文
        """
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")
        match = pattern.search(md_content)
        if not match:
            return None
        start, end = match.span()

        pre_text = md_content[max(0, start - content_len):start]
        post_text = md_content[end:min(len(md_content), end + content_len)]
        return pre_text, post_text

    def _step_3_generate_summaries(self, doc_stem: str, target_images: List[Tuple[str, str, Tuple[str, str]]]) -> Dict[
        str, str]:
        summaries = {}
        request_deque = deque()

        for img_file, image_path, context in target_images:
            self._apply_api_rate_limit(request_deque, max_requests=10)
            summaries[img_file] = self._summarize_image(image_path, root_folder=doc_stem, image_content=context)

        return summaries

    def _apply_api_rate_limit(
            self,
            request_times: Deque[float],
            max_requests: int,
            window_seconds: int = 60
    ) -> None:
        """
        通用滑动窗口API速率限制器（抽离为公共工具）
        核心逻辑：维护请求时间戳双端队列，窗口内请求数超上限则自动等待，防止触发第三方API限流
        :param request_times: 存储请求时间戳的双端队列，需外部初始化（全局/单例），跨调用复用
        :param max_requests: 速率限制窗口内的最大允许请求次数
        :param window_seconds: 速率限制滑动窗口时长，默认60秒（1分钟）
        :return: None，超出限制时会阻塞等待
        """
        current_time = time.time()

        while request_times and current_time - request_times[0] >= window_seconds:
            request_times.popleft()

        if len(request_times) >= max_requests:
            sleep_duration = window_seconds - (current_time - request_times[0])
            if sleep_duration > 0:
                logging.getLogger().info(
                    f"触发API速率限制，窗口{window_seconds}秒内最多{max_requests}次，需等待：{sleep_duration:.2f} 秒")
                time.sleep(sleep_duration)
                current_time = time.time()
                while request_times and current_time - request_times[0] >= window_seconds:
                    request_times.popleft()

        request_times.append(current_time)
        logging.getLogger().info(f"API请求时间戳已记录，当前{window_seconds}秒窗口内请求数：{len(request_times)}")

    def _summarize_image(self, image_path: str, root_folder: str, image_content: Tuple[str, str]) -> str:
        with open(image_path, "rb") as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        vl_ai = get_llm_client(lm_config.vl_model)

        message = [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""这是"{root_folder}"文件中的一张图片，图片上文部分为"{image_content[0]}"，下文部分为"{image_content[1]}"，请用中文简要总结这张图片的内容，用于 Markdown 图片标题。尽量不要超过十个字"""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                }
            ]
        }]
        response = vl_ai.invoke(message)

        return response.content.strip().replace("\n", "")

    def _step_4_upload_and_replace(self, doc_stem: str, target_images: List[Tuple[str, str, Tuple[str, str]]],
                                   summaries: Dict[str, str], md_content: str):
        minio_client = get_minio_client()
        minio_img_dir = minio_config.img_dir
        upload_dir = f"{minio_img_dir}/{doc_stem}"
        print(f"将图片存入{upload_dir}目录下")

        self.clean_minio_directory(minio_client, upload_dir)

        urls = self.upload_images_bath(minio_client, upload_dir, target_images)

        image_info = self.merge_summary_and_url(summaries, urls)

        md_content = self.process_md_file(md_content, image_info)

        return md_content

    def clean_minio_directory(self, minio_client: Minio, prefix: str) -> None:
        try:
            objects_to_delete = minio_client.list_objects(minio_config.bucket_name, prefix=prefix, recursive=True)
            delete_list = [DeleteObject(obj.object_name) for obj in objects_to_delete]
            if delete_list:
                errors = minio_client.remove_objects(minio_config.bucket_name, delete_list)
                for error in errors:
                    self.logger.error(f"删除失败：{error}")
        except Exception as e:
            self.logger.error(f"清理minio目录失败：{e}")

    def upload_images_bath(self, minio_client: Minio, upload_dir: str,
                           target_images: List[Tuple[str, str, Tuple[str, str]]]):
        urls = {}
        for img_file, img_path, _ in target_images:
            object_name = f"{upload_dir}/{img_file}"
            print(f"上传文件,{object_name}")
            urls[img_file] = self.uplaod_to_minio(minio_client, img_path, object_name)
        return urls
    def merge_summary_and_url(self, summaries: Dict[str, str], urls: Dict[str, str]) -> Dict[str, Tuple[str, str]]:
        image_info = {}
        for image_file, summary in summaries.items():
            if url := urls.get(image_file):
                image_info[image_file] = (summary, url)
        return image_info

    def process_md_file(self, md_content: str, image_info: Dict[str, Tuple[str, str]]) -> str:
        for image_file, (summary, url) in image_info.items():
            pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")
            md_content = pattern.sub(lambda m: f"![{summary}]({url})", md_content)
        return md_content

    def uplaod_to_minio(self, minio_client: Minio, img_path: str, object_name: str) -> str:
        ifSuccess = minio_client.fput_object(
            bucket_name=minio_config.bucket_name,
            object_name=object_name,
            file_path=img_path,
            content_type=f"image/{os.path.splitext(img_path)[1][1:]}"
        )
        url = f"http://{minio_config.endpoint}/{minio_config.bucket_name}/{object_name}"
        return url

    def _step_5_backup_new_md_file(self, origin_md_path: str, md_content: str) -> str:
        """
        步骤5：将处理后的MD内容保存为新文件（原文件不变，避免数据丢失）
        新文件命名规则：原文件名 + _new.md（如test.md → test_new.md）
        :param origin_md_path: 原始MD文件完整路径
        :param md_content: 处理后的新MD内容
        :return: 新MD文件的完整路径
        """
        new_md_file_name = os.path.splitext(origin_md_path)[0] + "_new.md"

        with open(new_md_file_name, "w", encoding="utf-8") as f:
            f.write(md_content)

        self.logger.info(f"处理后MD文件已保存，新文件路径：{new_md_file_name}")

        return new_md_file_name
