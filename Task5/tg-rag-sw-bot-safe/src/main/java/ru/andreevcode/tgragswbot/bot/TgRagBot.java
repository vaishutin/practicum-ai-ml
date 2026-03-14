package ru.andreevcode.tgragswbot.bot;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import org.telegram.telegrambots.bots.TelegramLongPollingBot;
import org.telegram.telegrambots.meta.api.methods.send.SendMessage;
import org.telegram.telegrambots.meta.api.objects.Update;
import org.telegram.telegrambots.meta.exceptions.TelegramApiException;
import ru.andreevcode.tgragswbot.client.RagApiException;
import ru.andreevcode.tgragswbot.config.AppProperties;
import ru.andreevcode.tgragswbot.service.RagService;
import ru.andreevcode.tgragswbot.util.MessageChunks;

import java.util.Set;

@Component
public class TgRagBot extends TelegramLongPollingBot {
    private static final int MESSAGE_SHRINK_MAX_CHARS = 200;
    private static final Set<String> ALLOWED_MODES = Set.of("zero_shot", "few_shot", "cot");
    private static final Logger log = LoggerFactory.getLogger(TgRagBot.class);

    private final AppProperties props;
    private final RagService ragService;

    public TgRagBot(AppProperties props, RagService ragService) {
        super(props.getTelegram().getToken());
        this.props = props;
        this.ragService = ragService;
    }

    @Override
    public String getBotUsername() {
        return props.getTelegram().getUsername();
    }

    @Override
    public void onUpdateReceived(Update update) {
        if (update == null || !update.hasMessage() || !update.getMessage().hasText()) return;

        var chatId = String.valueOf(update.getMessage().getChatId());
        var text = update.getMessage().getText().trim();
        if (text.isEmpty()) return;

        // change RAG mode
        if (text.startsWith("/change_mode")) {
            log.info("Got command: {}", text);
            handleChangeMode(chatId, text);
            return;
        }
        // toggle safety settings
        if (text.startsWith("/toggle_safety")) {
            log.info("Got command: {}", text);
            handleToggleSafety(chatId, text);
            return;
        }

        var shrunkMessage = shrink(text);
        // query
        String reply;
        try {
            log.info("Incoming query='{}', chatId={} is sending to RAG-API", shrunkMessage, chatId);
            reply = ragService.ask(text);
            log.info("Got RAG reply='{}', chatId={}", reply, chatId);
        } catch (RagApiException e) {
            log.warn("Failed to process message chatId={}, kind={}, status={}, body='{}': {}",
                    chatId, e.getKind(), e.getHttpStatus(), e.getResponseBody(), e.getMessage());
            reply = "Не удалось получить ответ от RAG.\n" + e.getMessage();
        }

        for (String chunk : MessageChunks.split(reply, props.getTelegram().getMaxMessageChars())) {
            try {
                execute(SendMessage.builder()
                        .chatId(chatId)
                        .text(chunk)
                        .build());
            } catch (Exception sendEx) {
                log.error("Failed to send message chatId={}: {}", chatId, sendEx.toString(), sendEx);
                break;
            }
        }
    }

    private void handleToggleSafety(String chatId, String text) {
        String[] parts = text.split("\\s+");

        if (parts.length != 4) {
            sendText(chatId, "Использование: /toggle_safety " +
                    "{prompt true/false} {chunks_filter true/false} {text_cleaner true/false}");
            return;
        }

        boolean newSafePromptMode = Boolean.parseBoolean(parts[1]);
        boolean newRetrieverChunksFilter = Boolean.parseBoolean(parts[2]);
        boolean newRetrieverCleaner = Boolean.parseBoolean(parts[3]);
        try {
            var resp = ragService.toggleSafePromptModeTo(
                    newSafePromptMode,
                    newRetrieverChunksFilter,
                    newRetrieverCleaner
            );
            String message = String.format(
                    "Настройки безопасности: safe_prompt=%s, chunks_filter=%s, text_cleaner=%s",
                    resp.safe_prompt() ? "ВКЛ" : "ВЫКЛ",
                    resp.retriever_chunks_filter() ? "ВКЛ" : "ВЫКЛ",
                    resp.retriever_text_cleaner() ? "ВКЛ" : "ВЫКЛ"
            );

            sendText(chatId, message);
            log.info("{}, настройки безопасности: safe_prompt={}, retriever_chunks_filter={}, " +
                    "retriever_text_cleaner={}", chatId, resp.safe_prompt(), resp.retriever_chunks_filter(),
                    resp.retriever_text_cleaner());
        } catch (Exception e){
            log.error("Failed to toggle safety settings: {}", e.getMessage(), e);
            sendText(chatId, "Не удалось изменить режим безопасных инструкций " + e.getMessage());
        }
    }

    private void handleChangeMode(String chatId, String text) {
        String[] parts = text.split("\\s+");

        if (parts.length != 2) {
            sendText(chatId, "Использование: /change_mode zero_shot | few_shot | cot");
            return;
        }

        String newMode = parts[1];

        if (!ALLOWED_MODES.contains(newMode)) {
            sendText(chatId, "Неизвестный режим: " + newMode);
            log.warn("Unknown safety_prompt to change: {}", newMode);
            return;
        }

        try {
            var currentMode = ragService.changeModeTo(newMode);
            var resultCommandMessage = "Текущий режим RAG изменен на: ";
            sendText(chatId, resultCommandMessage + currentMode);
            log.info("{}, {} {}", chatId,resultCommandMessage, currentMode);
        } catch (Exception e) {
            log.error("Failed to change RAG mode: {}", e.getMessage(), e);
            sendText(chatId, "Не удалось сменить режим RAG: " + e.getMessage());
        }
    }

    public void sendText(String chatId, String what) {
        var sm = buildBasicSendMessage(chatId, what).build();
        try {
            this.execute(sm);
        } catch (TelegramApiException e) {
            throw new RuntimeException(e);
        }
    }

    private SendMessage.SendMessageBuilder buildBasicSendMessage(String chatId, String what) {
        return SendMessage.builder()
                .chatId(chatId)
                .text(what);
    }

    private static String shrink(String s) {
        if (s == null) {
            return "";
        }
        return s.length() <= MESSAGE_SHRINK_MAX_CHARS ? s : s.substring(0, MESSAGE_SHRINK_MAX_CHARS) + "...";
    }
}