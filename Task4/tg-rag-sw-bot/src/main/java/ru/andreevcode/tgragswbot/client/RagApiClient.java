package ru.andreevcode.tgragswbot.client;


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.http.converter.HttpMessageConversionException;
import org.springframework.stereotype.Component;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientResponseException;

@Component
public class RagApiClient {
    private static final Logger log = LoggerFactory.getLogger(RagApiClient.class);

    private final RestClient restClient;

    public RagApiClient(RestClient ragRestClient) {
        this.restClient = ragRestClient;
    }

    public String query(String userText) {
        try {
            String mode = fetchCurrentMode(); // <-- новое

            RagQueryRequest req = new RagQueryRequest(mode, userText);

            RagQueryResponse resp = restClient.post()
                    .uri("/api/rag/query")
                    .contentType(MediaType.APPLICATION_JSON)
                    .accept(MediaType.APPLICATION_JSON)
                    .body(req)
                    .retrieve()
                    .body(RagQueryResponse.class);

            if (resp == null || resp.answer() == null) {
                throw new RagApiException(
                        RagApiException.Kind.INVALID_RESPONSE,
                        "RAG вернул пустой ответ (нет поля answer)",
                        null, null, null
                );
            }
            return resp.answer();

        } catch (RestClientResponseException e) {
            // HTTP статус != 2xx
            String body = e.getResponseBodyAsString();
            int status = e.getStatusCode().value();
            throw new RagApiException(
                    RagApiException.Kind.HTTP_STATUS,
                    "RAG вернул HTTP " + status,
                    status,
                    body,
                    e
            );

        } catch (ResourceAccessException e) {
            // таймауты / DNS / connect refused / read timeout и т.п.
            throw new RagApiException(
                    RagApiException.Kind.TIMEOUT_OR_IO,
                    "Таймаут или проблема сети при запросе к RAG",
                    null,
                    null,
                    e
            );

        } catch (HttpMessageConversionException e) {
            // JSON не распарсился в RagQueryResponse
            throw new RagApiException(
                    RagApiException.Kind.INVALID_JSON,
                    "Не удалось распарсить JSON от RAG",
                    null,
                    null,
                    e
            );

        } catch (Exception e) {
            // на всякий случай, чтобы не терять причину
            throw new RagApiException(
                    RagApiException.Kind.UNKNOWN,
                    "Неизвестная ошибка при запросе к RAG",
                    null,
                    null,
                    e
            );
        }
    }

    private String fetchCurrentMode() {
        RagGetModeResponse modeResp = restClient.get()
                .uri("/api/rag/mode")
                .retrieve()
                .body(RagGetModeResponse.class);

        if (modeResp == null || modeResp.current_mode() == null || modeResp.current_mode().isBlank()) {
            throw new RagApiException(
                    RagApiException.Kind.INVALID_RESPONSE,
                    "RAG вернул пустой current_mode",
                    null, null, null
            );
        }
        log.info("RAG mode: {}", modeResp.current_mode());

        return modeResp.current_mode();
    }

    public String changeModeTo(String mode) {
        try {
            RagPostModeResponse resp = restClient.post()
                    .uri("/api/rag/mode")
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(new RagChangeModeRequest(mode))
                    .retrieve()
                    .body(RagPostModeResponse.class);

            if (resp == null || resp.current_mode() == null) {
                throw new RagApiException(
                        RagApiException.Kind.INVALID_RESPONSE,
                        "Нет подтверждения смены режима (нет поля current_mode)",
                        null, null, null
                );
            }
            return resp.current_mode();

        } catch (RestClientResponseException e) {
            throw new RagApiException(
                    RagApiException.Kind.HTTP_STATUS,
                    "Не удалось сменить режим RAG, HTTP " + e.getStatusCode().value(),
                    e.getStatusCode().value(),
                    shrink(e.getResponseBodyAsString(), 500),
                    e
            );

        } catch (ResourceAccessException e) {
            throw new RagApiException(
                    RagApiException.Kind.TIMEOUT_OR_IO,
                    "Таймаут или проблема сети при смене режима RAG",
                    null,
                    null,
                    e
            );
        }
    }

    private static String shrink(String s, int max) {
        if (s == null) return "";
        return s.length() <= max ? s : s.substring(0, max) + "...";
    }
}
