package ru.andreevcode.tgragswbot.service;

import org.springframework.stereotype.Service;
import ru.andreevcode.tgragswbot.client.RagApiClient;
import ru.andreevcode.tgragswbot.client.response.RagPostToggleSafetyResponse;

@Service
public class RagService {
    private final RagApiClient client;

    public RagService(RagApiClient client) {
        this.client = client;
    }

    public String ask(String userText) {
        return client.query(userText);
    }

    public String changeModeTo(String mode) {
        return client.changeModeTo(mode);
    }

    public RagPostToggleSafetyResponse toggleSafePromptModeTo(boolean newSafePromptMode, boolean retrieverChunksFilter,
                                                              boolean retrieverTextCleaner) {
        return client.toggleSafetyTo(newSafePromptMode, retrieverChunksFilter, retrieverTextCleaner);
    }
}