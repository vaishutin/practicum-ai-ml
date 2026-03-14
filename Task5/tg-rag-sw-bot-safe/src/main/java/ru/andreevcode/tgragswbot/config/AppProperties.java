package ru.andreevcode.tgragswbot.config;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;

@Getter
@Setter
@ConfigurationProperties(prefix = "app")
public class AppProperties {

    private final Telegram telegram = new Telegram();
    private final Rag rag = new Rag();

    @Setter
    @Getter
    public static class Telegram {
        private String username;
        private String token;
        private int maxMessageChars = 4000;
    }

    @Getter
    @Setter
    public static class Rag {
        private String baseUrl;
        private final Timeout timeout = new Timeout();

        @Setter
        @Getter
        public static class Timeout {
            private int connectMs = 3000;
            private int readMs = 15000;

        }
    }
}
