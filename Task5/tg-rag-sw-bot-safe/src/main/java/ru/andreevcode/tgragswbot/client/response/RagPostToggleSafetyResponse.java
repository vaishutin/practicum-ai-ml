package ru.andreevcode.tgragswbot.client.response;

import ru.andreevcode.tgragswbot.client.ValidatableResponse;

public record RagPostToggleSafetyResponse(Boolean safe_prompt,
                                          Boolean retriever_chunks_filter,
                                          Boolean retriever_text_cleaner) implements ValidatableResponse {
    @Override
    public boolean isValid() {
        return safe_prompt != null && retriever_chunks_filter != null && retriever_text_cleaner != null;
    }
}
