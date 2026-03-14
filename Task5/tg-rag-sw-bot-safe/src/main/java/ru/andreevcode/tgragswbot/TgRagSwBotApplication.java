package ru.andreevcode.tgragswbot;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import ru.andreevcode.tgragswbot.config.AppProperties;

@SpringBootApplication
@EnableConfigurationProperties(AppProperties.class)
public class TgRagSwBotApplication {
    public static void main(String[] args) {
        SpringApplication.run(TgRagSwBotApplication.class, args);
    }
}
