"""Recommend the three most similar security categories for a new control."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import DriverError, Neo4jError
from sentence_transformers import SentenceTransformer, util


MODEL_NAME = "BAAI/bge-m3"
ENV_PATH = Path(__file__).resolve().parent / ".env"
REQUIRED_ENV_VARS = (
    "NEO4J_URI",
    "NEO4J_USERNAME",
    "NEO4J_PASSWORD",
    "NEO4J_DATABASE",
)
RECOMMEND_OPTION = "1"
SAVE_OPTION = "2"
EXIT_OPTION = "0"

CATEGORY_QUERY = """
MATCH (d:Domain)-[:HAS_CATEGORY]->(c:Category)
RETURN
    d.id AS domain_id,
    d.name AS domain_name,
    c.id AS category_id,
    c.name AS category_name
ORDER BY c.id
"""


class ConfigurationError(Exception):
    """Raised when the local Neo4j configuration is incomplete."""


class CategoryDataError(Exception):
    """Raised when Category data cannot be used for recommendation."""


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str
    username: str
    password: str
    database: str


@dataclass(frozen=True)
class Category:
    domain_id: str
    domain_name: str
    category_id: str
    category_name: str

    @property
    def comparison_text(self) -> str:
        return f"{self.domain_name} {self.category_name}"


@dataclass(frozen=True)
class Recommendation:
    category: Category
    score: float


def load_config(env_path: Path = ENV_PATH) -> Neo4jConfig:
    """Load and validate Neo4j settings from algorithm/.env."""
    if not env_path.is_file():
        raise ConfigurationError(
            f"환경설정 파일을 찾을 수 없습니다: {env_path}\n"
            "algorithm/.env.example을 algorithm/.env로 복사한 뒤 "
            "Neo4j 연결 정보를 입력해주세요."
        )

    load_dotenv(dotenv_path=env_path, override=True)
    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name, "").strip()]
    if missing:
        raise ConfigurationError(
            "algorithm/.env에 다음 필수 환경변수를 입력해주세요: "
            + ", ".join(missing)
        )

    return Neo4jConfig(
        uri=os.environ["NEO4J_URI"].strip(),
        username=os.environ["NEO4J_USERNAME"].strip(),
        password=os.environ["NEO4J_PASSWORD"],
        database=os.environ["NEO4J_DATABASE"].strip(),
    )


def _read_categories(transaction: Any) -> list[Category]:
    records = transaction.run(CATEGORY_QUERY)
    categories: list[Category] = []

    for record in records:
        values = {
            "domain_id": record["domain_id"],
            "domain_name": record["domain_name"],
            "category_id": record["category_id"],
            "category_name": record["category_name"],
        }
        missing = [
            key
            for key, value in values.items()
            if value is None or not str(value).strip()
        ]
        if missing:
            raise CategoryDataError(
                "Category 조회 결과에 비어 있는 필드가 있습니다: " + ", ".join(missing)
            )

        categories.append(Category(**{key: str(value) for key, value in values.items()}))

    return categories


def fetch_categories(config: Neo4jConfig) -> list[Category]:
    """Read all categories from Neo4j without changing graph data."""
    driver = GraphDatabase.driver(
        config.uri,
        auth=(config.username, config.password),
    )
    try:
        driver.verify_connectivity()
        with driver.session(database=config.database) as session:
            categories = session.execute_read(_read_categories)
    finally:
        driver.close()

    if not categories:
        raise CategoryDataError(
            "Neo4j에서 Category를 찾지 못했습니다. 그래프 데이터가 구축되어 있는지 확인해주세요."
        )
    return categories


def recommend_categories(
    control_text: str,
    categories: Sequence[Category],
    model: SentenceTransformer,
    top_k: int = 3,
) -> list[Recommendation]:
    """Rank categories by cosine similarity to the supplied control text."""
    category_texts = [category.comparison_text for category in categories]
    control_embedding = model.encode(
        [control_text],
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    category_embeddings = model.encode(
        category_texts,
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    scores = util.cos_sim(control_embedding, category_embeddings)[0]
    score_values = scores.detach().cpu().tolist()

    ranked_indices = sorted(
        range(len(categories)),
        key=lambda index: score_values[index],
        reverse=True,
    )[: min(top_k, len(categories))]

    return [
        Recommendation(category=categories[index], score=float(score_values[index]))
        for index in ranked_indices
    ]


def read_control_text() -> str | None:
    """Prompt until a non-empty Control sentence is entered or input is cancelled."""
    while True:
        try:
            control_text = input("새로운 Control 문장을 입력하세요: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n입력이 취소되어 프로그램을 종료합니다.")
            return None

        if control_text:
            return control_text
        print("입력 문장이 비어 있습니다. Control 문장을 다시 입력해주세요.")


def read_menu_choice() -> str | None:
    """Display the prototype menu and return a valid user choice."""
    while True:
        print("\n=== SBERT Category 추천 메뉴 ===")
        print("1. 새로운 Control 문장 검사")
        print("2. Neo4j에 Control 항목 추가 (미구현)")
        print("0. 프로그램 종료")

        try:
            choice = input("메뉴를 선택하세요: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n입력이 취소되어 프로그램을 종료합니다.")
            return None

        if choice in {RECOMMEND_OPTION, SAVE_OPTION, EXIT_OPTION}:
            return choice
        print("0, 1, 2 중 하나를 입력해주세요.")


def print_recommendations(
    control_text: str,
    recommendations: Sequence[Recommendation],
) -> None:
    print("\n=== SBERT Category 추천 결과 ===\n")
    print("입력 Control:")
    print(f"{control_text}\n")

    for rank, recommendation in enumerate(recommendations, start=1):
        category = recommendation.category
        print(f"{rank}위 {category.category_id} {category.category_name}")
        print(f"유사도: {recommendation.score:.4f}\n")


def main() -> int:
    try:
        config = load_config()
    except ConfigurationError as error:
        print(f"설정 오류: {error}", file=sys.stderr)
        return 1

    model: SentenceTransformer | None = None

    while True:
        choice = read_menu_choice()
        if choice is None or choice == EXIT_OPTION:
            print("프로그램을 종료합니다.")
            return 0

        if choice == SAVE_OPTION:
            print(
                "Neo4j Control 항목 추가 기능은 현재 프로토타입에서 "
                "구현되지 않았습니다."
            )
            continue

        control_text = read_control_text()
        if control_text is None:
            return 0

        try:
            categories = fetch_categories(config)
        except CategoryDataError as error:
            print(f"데이터 오류: {error}", file=sys.stderr)
            return 1
        except (DriverError, Neo4jError) as error:
            print(
                "Neo4j 연결 또는 조회에 실패했습니다. Neo4j가 실행 중인지와 "
                "algorithm/.env의 URI, 계정, 비밀번호, 데이터베이스 이름을 "
                "확인해주세요.\n"
                f"상세 오류: {error}",
                file=sys.stderr,
            )
            return 1

        try:
            if model is None:
                model = SentenceTransformer(MODEL_NAME)
            recommendations = recommend_categories(control_text, categories, model)
        except Exception as error:
            print(
                "SBERT 모델을 불러오거나 유사도를 계산하지 못했습니다. "
                "패키지 설치 상태와 인터넷 연결(최초 모델 다운로드 시 필요)을 "
                "확인해주세요.\n"
                f"상세 오류: {error}",
                file=sys.stderr,
            )
            return 1

        print_recommendations(control_text, recommendations)


if __name__ == "__main__":
    raise SystemExit(main())
