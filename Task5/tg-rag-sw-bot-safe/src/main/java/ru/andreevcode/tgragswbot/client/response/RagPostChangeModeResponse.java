package ru.andreevcode.tgragswbot.client.response;

import ru.andreevcode.tgragswbot.client.ValidatableResponse;

public record RagPostChangeModeResponse(String current_mode) implements ValidatableResponse {
    @Override
    public boolean isValid() {
        return current_mode != null;
    }
}
