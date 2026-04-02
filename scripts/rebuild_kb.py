from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.settings import get_settings
from tools.knowledge_base import KnowledgeBaseTool


def main() -> None:
    settings = get_settings()
    tool = KnowledgeBaseTool(settings=settings)
    tool.rebuild()

    unique_files = len({item.metadata.get("file_name") for item in tool.documents})
    print("知识库重建完成")
    print(f"后端: {settings.kb_backend}")
    print(f"向量库目录: {settings.qdrant_path_resolved}")
    print(f"Collection: {settings.qdrant_collection}")
    print(f"Embedding 后端: {settings.embedding_backend}")
    print(
        "BM25 后端: "
        + (
            f"Qdrant 官方 BM25 ({settings.qdrant_bm25_tokenizer})"
            if settings.use_qdrant_official_bm25
            else "本地 BM25"
        )
    )
    print(f"Chunk 数: {len(tool.documents)}")
    print(f"源文件数: {unique_files}")


if __name__ == "__main__":
    main()
