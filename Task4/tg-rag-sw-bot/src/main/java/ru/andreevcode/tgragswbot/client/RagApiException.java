package ru.andreevcode.tgragswbot.client;

import lombok.Getter;

@Getter
public class RagApiException extends RuntimeException {
    public enum Kind {
        TIMEOUT_OR_IO,
        HTTP_STATUS,
        INVALID_JSON,
        INVALID_RESPONSE,
        UNKNOWN
    }

    private final Kind kind;
    private final Integer httpStatus;      // nullable
    private final String responseBody;     // nullable (уже shrink)

    public RagApiException(Kind kind, String message, Integer httpStatus, String responseBody, Throwable cause) {
        super(message, cause);
        this.kind = kind;
        this.httpStatus = httpStatus;
        this.responseBody = responseBody;
    }
}