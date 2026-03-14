package ru.andreevcode.tgragswbot.config;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.web.client.RestClient;

@Configuration
public class RagClientConfig {
    private static final Logger log = LoggerFactory.getLogger(RagClientConfig.class);

    @Bean
    public RestClient.Builder restClientBuilder() {
        return RestClient.builder();
    }

    @Bean
    public RestClient ragRestClient(RestClient.Builder restClientBuilder, AppProperties props) {
        var baseUrl = props.getRag().getBaseUrl();

        SimpleClientHttpRequestFactory rf = new SimpleClientHttpRequestFactory();
        rf.setConnectTimeout(props.getRag().getTimeout().getConnectMs());
        rf.setReadTimeout(props.getRag().getTimeout().getReadMs());

        log.info("RAG baseUrl={}, connectMs={}, readMs={}",
                baseUrl, props.getRag().getTimeout().getConnectMs(), props.getRag().getTimeout().getReadMs());

        return restClientBuilder
                .baseUrl(baseUrl)
                .requestFactory(rf)
                .build();
    }
}