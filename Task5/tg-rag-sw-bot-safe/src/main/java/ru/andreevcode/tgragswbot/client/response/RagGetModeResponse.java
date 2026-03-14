package ru.andreevcode.tgragswbot.client.response;

import java.util.List;

// Да, поля в snake_case — Jackson нормально это съест “как есть”, потому что имена совпадают 1:1.
public record RagGetModeResponse(List<String> available_modes, String current_mode) {}
