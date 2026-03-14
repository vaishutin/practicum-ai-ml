package ru.andreevcode.tgragswbot.service;

import org.springframework.stereotype.Service;
import ru.andreevcode.tgragswbot.client.RagApiClient;

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
}