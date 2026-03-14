package ru.andreevcode.tgragswbot.config;

import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Profile;
import org.telegram.telegrambots.meta.TelegramBotsApi;
import org.telegram.telegrambots.updatesreceivers.DefaultBotSession;
import ru.andreevcode.tgragswbot.bot.TgRagBot;

@Profile("!test")
@Configuration
public class TelegramBotConfig {
    private static final Logger log = LoggerFactory.getLogger(TelegramBotConfig.class);

    private final TgRagBot bot;

    public TelegramBotConfig(TgRagBot bot) {
        this.bot = bot;
    }

    @PostConstruct
    public void register() throws Exception {
        TelegramBotsApi botsApi = new TelegramBotsApi(DefaultBotSession.class);
        botsApi.registerBot(bot);
        log.info("Telegram bot registered: @{}", bot.getBotUsername());
    }
}