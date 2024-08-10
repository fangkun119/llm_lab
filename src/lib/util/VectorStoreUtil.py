import time
from enum import Enum
from typing import List, Optional

from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from itertools import islice
from abc import ABC, abstractmethod


class VectorStoreProp:
    def __init__(self, vendor_name, dump_sub_dir, dump_file):
        self._vendor_name = vendor_name
        self._dump_sub_dir = dump_sub_dir
        self._dump_file = dump_file

    @property
    def vendor_name(self):
        return self._vendor_name

    @property
    def dump_file(self):
        return self._dump_file

    @property
    def dump_sub_dir(self):
        return self._dump_sub_dir


class VectorStoreVendor(Enum):
    FAISS = 1,


prop_map = {
    VectorStoreVendor.FAISS: VectorStoreProp(
        vendor_name="FAISS",
        dump_sub_dir="faiss_dump",
        dump_file="index"),
}


class VectorStoreWrapper(ABC):
    def __init__(self, prop: VectorStoreProp):
        self._prop = prop
        self._vector_store = None

    @abstractmethod
    def init_from_docs(self, docs: List[Document], embedding: Embeddings):
        pass

    @abstractmethod
    def init_from_dump(self, embedding: Embeddings, base_dir: str):
        pass

    @abstractmethod
    def get_vector_store(self):
        pass

    @abstractmethod
    def trigger_dump(self, base_dir: str):
        pass


class FAISSWrapper(VectorStoreWrapper, ABC):
    def init_from_docs(self, docs: List[Document], embedding: Embeddings):
        chunk_size = 10
        doc_list_arr = [list(islice(docs, index, index + chunk_size)) for index in
                        range(0, len(docs), chunk_size)]
        for idx, doc_list in enumerate(doc_list_arr):
            print(f"load_chunk: {idx}")
            if self._vector_store is None:
                self._vector_store = FAISS.from_documents(documents=doc_list, embedding=embedding)
            else:
                time.sleep(1) # prevent triggering the rate limiter
                self._vector_store.add_documents(documents=doc_list)
        print("vector store load complete")

    def trigger_dump(self, base_dir: str):
        print(f"vector store dump triggered, dir: {base_dir}/{self._prop.dump_sub_dir}")
        self._vector_store.save_local(
            folder_path=f"{base_dir}/{self._prop.dump_sub_dir}",
            index_name=self._prop.dump_file)

    def init_from_dump(self, embedding: Embeddings, base_dir: str):
        print(f"load from {base_dir}/{self._prop.dump_sub_dir}")
        self._vector_store = FAISS.load_local(
            folder_path=f"{base_dir}/{self._prop.dump_sub_dir}",
            embeddings=embedding,
            index_name=self._prop.dump_file,
            allow_dangerous_deserialization=True)
        print(f"load compete")

    def get_vector_store(self):
        return self._vector_store


class VectorStoreUtil:
    @staticmethod
    def create_wrapper(vendor: VectorStoreVendor) -> Optional[VectorStoreWrapper]:
        if vendor == VectorStoreVendor.FAISS:
            return FAISSWrapper(prop=prop_map.get(vendor))
        else:
            return None
