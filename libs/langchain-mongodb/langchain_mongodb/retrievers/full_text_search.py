import warnings
from typing import Annotated, Any, Dict, List, Optional, Union

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
from pymongo.collection import Collection
from pymongo.errors import CollectionInvalid
from pymongo_search_utils import create_vector_search_index, update_vector_search_index

from langchain_mongodb.pipelines import text_search_stage
from langchain_mongodb.utils import _append_client_metadata, make_serializable


class MongoDBAtlasFullTextSearchRetriever(BaseRetriever):
    """Retriever performs full-text searches using Lucene's standard (BM25) analyzer."""

    collection: Collection
    """MongoDB Collection on an Atlas cluster"""
    search_index_name: str
    """Atlas Search Index name"""
    search_field: Union[str, List[str]]
    """Collection field that contains the text to be searched. It must be indexed"""
    k: Optional[int] = None
    """Number of documents to return. Default is no limit"""
    filter: Optional[Dict[str, Any]] = None
    """(Optional) List of MQL match expression comparing an indexed field"""
    include_scores: bool = True
    """If True, include scores that provide measure of relative relevance"""
    top_k: Annotated[
        Optional[int], Field(deprecated='top_k is deprecated, use "k" instead')
    ] = None
    _added_metadata: bool = False
    """Number of documents to return. Default is no limit"""

    def __init__(
        self,
        collection: Collection,
        search_field: Union[str, List[str]],
        search_index_name: str,
        filter: Optional[Dict[str, Any]] = None,
        include_scores: bool = True,
        k: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        dimensions: int = -1,
        auto_create_index: bool | None = None,
        auto_index_timeout: int = 15,
        vector_index_options: dict | None = None,
        relevance_score_fn: str | None = "cosine",
        embedding_key: str | None = "embedding",
        embedding: Optional[Union[Embeddings, str]] = None,
    ):
        super().__init__(
            collection=collection,
            search_field=search_field,
            search_index_name=search_index_name,
            filter=filter,
            include_scores=include_scores,
            k=k,
            metadata=metadata,
            tags=tags,
        )
        self._relevance_score_fn = relevance_score_fn
        self._embedding_key = embedding_key
        self._embedding = embedding

        if auto_create_index is False:
            return
        if auto_create_index is None and dimensions == -1:
            return

        # Bail if the index is already created.
        if any(
            [
                ix["name"] == search_index_name
                for ix in self.collection.list_search_indexes()
            ]
        ):
            return
        self.create_vector_search_index(
            dimensions,
            wait_until_complete=auto_index_timeout,
            vector_index_options=vector_index_options,
        )

    def create_vector_search_index(
        self,
        dimensions: int = -1,
        filters: Optional[List[str]] = None,
        update: bool = False,
        wait_until_complete: Optional[float] = None,
        vector_index_options: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Creates a MongoDB Atlas vectorSearch index for the retriever

        Note**: This method may fail as it requires a MongoDB Atlas with these
        `pre-requisites <https://www.mongodb.com/docs/atlas/atlas-vector-search/create-index/#prerequisites>`.
        Currently, vector and full-text search index operations need to be
        performed manually on the Atlas UI for shared M0 clusters.

        Args:
            dimensions (Optional[int]): Number of dimensions in embedding. Should be `-1` if embedding is an instance of `AutoEmbeddings`.
                Otherwise if the value is not provided, it will be inferred using the embedding model.
            filters (Optional[List[Dict[str, str]]], optional): additional filters
            for index definition.
                Defaults to None.
            update (Optional[bool]): Updates existing vectorSearch index.
                 Defaults to False.
            wait_until_complete (Optional[float]): If given, a TimeoutError is raised
                if search index is not ready after this number of seconds.
                If not given, the default, operation will not wait.
            kwargs: (Optional): Keyword arguments supplying any additional options
                to SearchIndexModel.
        """
        try:
            self.collection.database.create_collection(self.collection.name)
        except CollectionInvalid:
            pass

        index_operation = (
            update_vector_search_index if update else create_vector_search_index
        )
        embedding_model = None
        assert self._embedding_key is not None
        path = self._embedding_key
        if dimensions == -1:
            dimensions = len(self._embedding.embed_query("foo"))

        index_operation(
            collection=self.collection,
            index_name=self.search_index_name,
            dimensions=dimensions,
            path=path,
            similarity=self._relevance_score_fn,
            filters=filters or [],
            vector_index_options=vector_index_options,
            wait_until_complete=wait_until_complete,
            auto_embedding_model=embedding_model,
            **kwargs,
        )  # type: ignore [operator]

    def close(self) -> None:
        """Close the resources used by the MongoDBAtlasFullTextSearchRetriever."""
        self.collection.database.client.close()

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        """Retrieve documents that are highest scoring / most similar  to query.

        Args:
            query: String to find relevant documents for
            run_manager: The callback handler to use
        Returns:
            List of relevant documents
        """
        is_top_k_set = False
        with warnings.catch_warnings():
            # Ignore warning raised by checking the value of top_k.
            warnings.simplefilter("ignore", DeprecationWarning)
            if self.top_k is not None:
                is_top_k_set = True
        default_k = self.k if not is_top_k_set else self.top_k
        pipeline = text_search_stage(  # type: ignore
            query=query,
            search_field=self.search_field,
            index_name=self.search_index_name,
            limit=kwargs.get("k", default_k),
            filter=self.filter,
            include_scores=self.include_scores,
        )

        if not self._added_metadata:
            _append_client_metadata(self.collection.database.client)
            self._added_metadata = True

        # Execution
        cursor = self.collection.aggregate(pipeline)  # type: ignore[arg-type]

        # Formatting
        docs = []
        for res in cursor:
            text = (
                res.pop(self.search_field)
                if isinstance(self.search_field, str)
                else res.pop(self.search_field[0])
            )
            make_serializable(res)
            docs.append(Document(page_content=text, metadata=res))
        return docs
