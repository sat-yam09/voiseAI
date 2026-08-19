"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type SpeechResult = ArrayLike<{ transcript: string }> & { isFinal: boolean };
type RecognitionEvent = { results: ArrayLike<SpeechResult> };
type Recognition = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((event: RecognitionEvent) => void) | null;
  onerror: (() => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
};
type RecognitionConstructor = new () => Recognition;

declare global {
  interface Window {
    SpeechRecognition?: RecognitionConstructor;
    webkitSpeechRecognition?: RecognitionConstructor;
  }
}

export function useLiveTranscript() {
  const recognitionRef = useRef<Recognition | null>(null);
  const activeRef = useRef(false);
  const finalTextRef = useRef("");
  const [text, setText] = useState("");
  const supported = typeof window !== "undefined" && Boolean(window.SpeechRecognition || window.webkitSpeechRecognition);

  useEffect(() => {
    return () => {
      activeRef.current = false;
      recognitionRef.current?.stop();
    };
  }, []);

  const start = useCallback(() => {
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Recognition) return false;

    recognitionRef.current?.stop();
    const recognition = new Recognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-IN";
    finalTextRef.current = "";
    activeRef.current = true;
    setText("");
    recognition.onresult = (event) => {
      let interim = "";
      for (let index = 0; index < event.results.length; index += 1) {
        const result = event.results[index][0];
        if (event.results[index].isFinal) finalTextRef.current += `${result.transcript} `;
        else interim += result.transcript;
      }
      setText(`${finalTextRef.current}${interim}`.trim());
    };
    recognition.onerror = () => setText(finalTextRef.current.trim());
    recognition.onend = () => {
      if (activeRef.current) {
        try { recognition.start(); } catch { /* browser is already restarting */ }
      }
    };
    recognitionRef.current = recognition;
    try {
      recognition.start();
      return true;
    } catch {
      activeRef.current = false;
      return false;
    }
  }, []);

  const stop = useCallback(() => {
    activeRef.current = false;
    recognitionRef.current?.stop();
  }, []);

  const reset = useCallback(() => {
    stop();
    finalTextRef.current = "";
    setText("");
  }, [stop]);

  return { text, supported, start, stop, reset };
}
