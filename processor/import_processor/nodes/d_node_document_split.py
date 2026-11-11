import json
import re
from pathlib import Path
from typing import Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from shapely.speedups import available
from tvm_ffi import List

from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.import_config import get_config
from processor.import_processor.state import ImportGraphState


class NodeDocumentSplit(BaseNode):
    """
    文档切分节点：智能文档切片
    """

    name = "node_document_split"

    def process(self, state: ImportGraphState):

        content,file_title = self._step_1_get_inputs(state)

        sections ,title_count,lines_count = self._step2_split_by_title(content, file_title)



        sections = self._step_3_handle_no_title(content, sections, title_count, file_title)

        sections = self._step_4_refine_chunks(sections)
        self._step_5_print_stats(lines_count, sections)

        self._step_6_backup(state, sections)

        state["chunks"] = sections
        return state

    def _step_1_get_inputs(self, state):
        print("node_document_split：步骤1：参数处理")
        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(field_name="file_title", message="文件标题不能为空", expected_type=str)
        content = state.get("md_content")
        if not content:
            raise StateFieldError(field_name="md_content", message="文件内容不能为空", expected_type=str)

        content = content.replace("\r\n", "\n").replace("\r", "\n")
        return content,file_title

    def _step2_split_by_title(self, content, file_title):
        print("node_document_split：步骤2：标题切（初切）")
        sections:List[Dict[str,str]] = []
        title_count = 0
        lines = content.split("\n")
        current_lines = []
        in_code_block = False
        current_title = ""

        title_pattern = r'\s*#{1,6}\s.+'

        def _flush_section():
            if not current_lines:
                return
            sections.append({
                "title": current_title,
                "content": "\n".join(current_lines),
                "parent_title": "",
                "file_title": file_title,
            })

        for line in lines:
            striped_line = line.strip()

            if striped_line.startswith("~~~") or striped_line.startswith("```"):
                in_code_block = not in_code_block
                current_lines.append(line)
                continue

            if (not in_code_block) and (re.match(title_pattern, line)):
                _flush_section()
                current_title = striped_line
                current_lines = [current_title]
                title_count += 1

            else:
                current_lines.append(line)

        _flush_section()
        print(f"标题初切：sections:{sections}")
        return sections,title_count,len(lines)

    def _step_3_handle_no_title(self, content, sections, title_count, file_title):
        print("node_document_split：步骤3：无标题兜底")
        if title_count ==  0:
            return [{"title":"无标题","content":content,"file_title":file_title}]
        return sections

    def _step_4_refine_chunks(self, sections):
        print("node_document_split：步骤4：块精细化处理（长切短和）")

        refined_split = []
        for sec in sections:
            refined_split.extend(self.split_long_section(sec))

        final_sections = self.merge_short_sections(refined_split)

        for sec in final_sections:
            if not sec.get("parent_title"):
                sec["parent_title"] = sec.get("title") or ""

        return final_sections

    def _step_5_print_stats(self, lines_count, sections):
        print("node_document_split：步骤5：打印日志")
        chunk_num = len(sections)
        self.logger.info("-" * 50 + " 文档切分统计信息 " + "-" * 50)
        self.logger.info(f"MD原始文本总行数：{lines_count}")
        self.logger.info(f"最终生成Chunk数量：{chunk_num}")

    def _step_6_backup(self, state, sections):
        print("node_document_split：步骤6：备份")

        path = f'{Path(state.get("md_path")).parent}/{state.get("file_title")}_new_chunks.json'
        with open(path,"w",encoding="utf-8") as f:
            json.dump(
                sections,
                f,
                ensure_ascii=False,
                indent=2,
            )

    def split_long_section(self, section: Dict[str,str]) -> List[Dict[str,str]]:
        print("node_document_split：骤4的方法1：长切")
        content = section.get("content")
        content_len = len(content)
        if content_len <= get_config().max_content_length:
            return [section]

        title = section.get("title")
        prefix = f"{title}\n\n" if title else ""
        available_len = get_config().max_content_length - len(prefix)

        body = content
        if title and body.lstrip().startswith(title):
            body = body[body.find(title) + len(title):].lstrip()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=available_len,
            chunk_overlap=0,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
        )

        sub_sections = []
        for index, chunk in enumerate(splitter.split_text(body), start=1):
            test = chunk.lstrip()
            if not test:
                continue
            full_test = (prefix + test).lstrip()

            sub_sections.append({
                "title": f"{title} - {index}" if title else f"chunk - {index}",
                "content": full_test,
                "parent_title": title,
                "patent":index,
                "file_title": section.get("file_title"),
            })

        return sub_sections
    def merge_short_sections(self, sections):
        print("node_document_split：骤4的方法2：短和")

        if not sections:
            self.logger.debug("待合并Chunk列表为空，直接返回")
            return []

        merged_sections = []
        current_chunk = None

        for sec in sections:
            if current_chunk is None:
                current_chunk = sec
                continue

            is_current_short = len(current_chunk["content"]) < self.config.min_content_length
            is_same_parent = current_chunk.get("parent_title") == sec.get("parent_title")

            if is_current_short and is_same_parent:
                parent_title = sec.get("parent_title", "")
                next_content = sec["content"]
                if parent_title and next_content.startswith(parent_title):
                    next_content = next_content[len(parent_title):].lstrip()
                current_chunk["content"] += "\n\n" + next_content
                if "part" in sec:
                    current_chunk["part"] = sec["part"]
                self.logger.debug(
                    f"合并短Chunk：{current_chunk.get('parent_title')} → 累计长度{len(current_chunk['content'])}")
            else:
                merged_sections.append(current_chunk)
                current_chunk = sec

        if current_chunk is not None:
            merged_sections.append(current_chunk)

        self.logger.debug(f"短Chunk合并完成：原{len(sections)}个 → 合并后{len(merged_sections)}个")
        return merged_sections
