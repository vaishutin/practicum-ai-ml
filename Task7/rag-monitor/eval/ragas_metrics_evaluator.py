import asyncio
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import Faithfulness, AnswerRelevancy


class RagasMetricsEvaluator:
    def __init__(self, model="gpt-4o-mini", emb_model="text-embedding-3-small"):
        c = AsyncOpenAI()
        llm = llm_factory(model, client=c, temperature=0.0, )
        emb = embedding_factory("openai", model=emb_model, client=c)
        self._f = Faithfulness(llm=llm)
        self._ar = AnswerRelevancy(llm=llm, embeddings=emb)
        self._model = model

    async def evaluate_one(self, question: str, answer: str, contexts: list[str]) -> dict:
        f, ar = await asyncio.gather(
            self._f.ascore(user_input=question, response=answer, retrieved_contexts=contexts),
            self._ar.ascore(user_input=question, response=answer),
        )
        return {"faithfulness": f.value, "answer_relevancy": ar.value}

    async def evaluate_batch(self, rows: list[dict], concurrency: int = 8) -> dict:
        """
        rows: [{"question_id","question","answer","contexts"}]
        """
        sem = asyncio.Semaphore(concurrency)

        async def run_row(r: dict) -> dict:
            async with sem:
                s = await self.evaluate_one(r["question"], r["answer"], r["contexts"])
                return {"question_id": r["question_id"], **s}

        scores = await asyncio.gather(*(run_row(r) for r in rows))

        f = [s["faithfulness"] for s in scores]
        ar = [s["answer_relevancy"] for s in scores]
        return {
            "model": self._model,
            "count": len(scores),
            "faithfulness_avg": sum(f) / len(f) if f else 0.0,
            "answer_relevancy_avg": sum(ar) / len(ar) if ar else 0.0,
            "per_item": scores,
        }
