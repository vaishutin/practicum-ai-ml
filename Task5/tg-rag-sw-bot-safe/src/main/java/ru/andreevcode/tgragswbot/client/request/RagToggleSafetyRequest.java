package ru.andreevcode.tgragswbot.client.request;

public record RagToggleSafetyRequest(boolean safe_prompt,
                                     boolean retriever_chunks_filter,
                                     boolean retriever_text_cleaner) {
}
