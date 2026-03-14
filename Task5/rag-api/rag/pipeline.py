import os
from typing import Any, Dict

import logging
from .config import CONTEXT_QUESTION_TEMPLATE, INSTRUCTIONS, SAFE_INSTRUCTIONS, MODES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag-pipeline")

# ------------------------
# Глобальное состояние: режимы и промпты
# ------------------------
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # можно поменять
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# some configs
logger.info(
    "configs:  OPENAI_MODEL=%s, LLM_TEMPERATURE=%.1f, LLM_MAX_TOKENS=%d", OPENAI_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS)

class RagPipeline:
    def __init__(self, retriever: Any, tokenizer: Any, openai_client, current_mode: str = "cot"):
        self._retriever = retriever
        self._tokenizer = tokenizer
        self._openai_client = openai_client
        # ============== конфиги RAG ===========================
        self.current_mode = current_mode
        self.safe_prompt = True
        self.retriever_chunks_filter = False
        self.retriever_text_cleaner = False
        # ============== конфиги RAG ===========================

    def run_rag_pipeline(self,
                         question: str,
                         mode: str,
                         safe_prompt: bool,
                         ) -> Dict[str, Any]:
        """
        Главный RAG-пайплайн:
          1) Векторный поиск (сборка context - чанка с данными)
             a) принимает текст запроса от бота;
             b) строит эмбеддинг (bge-m3);
             c) делает запрос к Chroma;
          2) Сборка промпта выбранного режима;
          3) Вызов LLM (OpenAI);
          4) Возврат уже готового ответа пользователю.
        """
        if mode not in MODES:
            raise ValueError(f"Неизвестный режим: {mode}")

        # 1) Векторный поиск (сборка context - чанка с данными)
        context = self._retriever.retrieve_context(question)

        # 2) промпт из двух частей: a) инструкции, b) контекст + вопрос
        instructions = self.build_instructions(current_mode=mode, safe_prompt=safe_prompt)
        context_with_question = self.build_context_with_question(context, question)

        total_input_tokens = self.count_tokens(instructions) + self.count_tokens(context_with_question)
        logger.info("Final LLM input prompt (instruction and context_with_question) is built, tokens=%d, mode=%s)",
                    total_input_tokens, mode)
        logger.info("Final instructions:\n %s", instructions)
        logger.info("Final context_with_question:\n %s", context_with_question)

        # 3) LLM
        logger.info("Final prompt is sent to LLM.")
        answer = self.call_llm(instructions, context_with_question)
        logger.info("Got LLM's answer: \n%s", answer)

        # 4) вернуть ответ
        return {
            "answer": answer,
            # "mode": mode,
            # "used_top_k": RAG_TOP_K,
            # "context" : context,
            # можно добавить context для дебага (или убрать в прод)
            # "debug_prompt": prompt,
        }


    @staticmethod
    def build_instructions(current_mode: str, safe_prompt: bool) -> str:
        base = INSTRUCTIONS[current_mode]
        return base.format(safe_instructions=SAFE_INSTRUCTIONS) if safe_prompt else base.format(safe_instructions="")


    @staticmethod
    def build_context_with_question(context: str, question: str) -> str:
        """Собирает данные по контексту + вопрос пользователя вместе."""
        return CONTEXT_QUESTION_TEMPLATE.format(context=context, question=question)


    def count_tokens(self, text: str) -> int:
        return len(
            self._tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        )


    def call_llm(self, instruction: str, context_question_data: str) -> str:
        """Вызов LLM (OpenAI) с собранным промптом, разделенным на инструкцию и контекст + вопрос"""
        response = self._openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            messages=[
                {
                    "role": "system",
                    "content": instruction,
                },
                {
                    "role": "user",
                    "content": context_question_data,
                },
            ],
        )
        return response.choices[0].message.content