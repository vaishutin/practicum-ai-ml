package ru.andreevcode.tgragswbot.util;

import java.util.ArrayList;
import java.util.List;

public final class MessageChunks {
    private MessageChunks() {}

    public static List<String> split(String text, int maxChars) {
        if (text == null) {
            return List.of("");
        }
        if (maxChars <= 0) {
            return List.of(text);
        }
        if (text.length() <= maxChars) {
            return List.of(text);
        }

        List<String> chunks = new ArrayList<>();
        int i = 0;

        while (i < text.length()) {
            int end = Math.min(i + maxChars, text.length());

            // стараемся резать по \n или пробелу (чтобы не ломать слова)
            int cut = bestCut(text, i, end);
            chunks.add(text.substring(i, cut));
            i = cut;
        }
        return chunks;
    }

    private static int bestCut(String s, int start, int end) {
        int nl = s.lastIndexOf('\n', end - 1);
        if (nl >= start + 1) {
            return nl + 1;
        }

        int sp = s.lastIndexOf(' ', end - 1);
        if (sp >= start + 1) {
            return sp + 1;
        }

        return end;
    }
}