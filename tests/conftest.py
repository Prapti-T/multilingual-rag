"""Shared fixtures. DOCS are tiny hand-written *test fixtures*, not a corpus: they exist so the
plumbing (chunk -> embed -> Qdrant -> retrieve -> evaluate) can be exercised offline with the
hash embedder. Retrieval *quality* on real data is measured with `mlrag eval`, not here."""
import pytest

from mlrag.config import Config
from mlrag.embedders import HashEmbedder
from mlrag.generate import ExtractiveGenerator
from mlrag.ingest import ingest, load_meta
from mlrag.pipeline import RAGPipeline
from mlrag.retrieve import Retriever
from mlrag.sparse import Bm25Encoder
from mlrag.store import QdrantStore


def _doc(doc_id, lang, title, text, heading=""):
    return {"doc_id": doc_id, "topic": title, "title": title, "lang": lang,
            "url": f"https://example.test/{lang}/{title}", "sections": [{"heading": heading, "text": text}]}


DOCS = [
    _doc("ne:काठमाडौं", "ne", "काठमाडौं", "काठमाडौं नेपालको राजधानी हो। यो बागमती नदीको किनारमा अवस्थित छ।", "परिचय"),
    _doc("ne:सगरमाथा", "ne", "सगरमाथा", "सगरमाथा संसारको सबैभन्दा अग्लो हिमाल हो। यो नेपाल र चीनको सिमानामा पर्छ।", "परिचय"),
    _doc("ne:प्रकाश संश्लेषण", "ne", "प्रकाश संश्लेषण",
         "प्रकाश संश्लेषणमा बिरुवाले सूर्यको प्रकाश प्रयोग गरेर खाना बनाउँछ।", "परिचय"),
    _doc("en:Photosynthesis", "en", "Photosynthesis",
         "Photosynthesis is the process by which plants make food using sunlight. "
         "It takes place in the chloroplasts of leaf cells.", "Overview"),
    _doc("en:Mount Everest", "en", "Mount Everest",
         "Mount Everest is the highest mountain in the world. It lies on the border between Nepal and China.",
         "Overview"),
]


class Index:
    def __init__(self, tmp_path, **cfg_overrides):
        self.cfg = Config()
        self.cfg.qdrant.path = ":memory:"
        self.cfg.index.meta_dir = str(tmp_path / "index")
        for k, v in cfg_overrides.items():
            setattr(self.cfg.sparse, k, v)
        self.embedder = HashEmbedder()
        self.store = QdrantStore(self.cfg.qdrant.collection, path=":memory:")
        self.stats = ingest(DOCS, self.cfg, self.embedder, self.store)
        meta = load_meta(self.cfg)
        self.sparse = Bm25Encoder.from_dict(meta["sparse"])
        self.retriever = Retriever(self.store, self.embedder, self.sparse)
        self.pipeline = RAGPipeline(self.retriever, ExtractiveGenerator(), self.cfg)


@pytest.fixture
def index(tmp_path):
    return Index(tmp_path)


@pytest.fixture
def index_no_fold(tmp_path):
    return Index(tmp_path, fold_vowel_length=False)
