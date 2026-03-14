# ---- build stage ----
FROM gradle:8.14-jdk17 AS build
WORKDIR /workspace

COPY build.gradle.kts settings.gradle.kts ./
COPY src ./src

RUN gradle --no-daemon clean bootJar

# ---- runtime stage ----
FROM eclipse-temurin:17-jre
WORKDIR /app

COPY --from=build /workspace/build/libs/*.jar /app/tg-rag-sw-bot.jar

EXPOSE 8001
ENV JAVA_OPTS=""

ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS -jar /app/tg-rag-sw-bot.jar"]