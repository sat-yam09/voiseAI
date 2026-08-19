"use client";

import { useCallback, useRef, useState } from "react";

type RecorderState = "idle" | "recording" | "ready" | "error";

export function useAudioRecorder() {
  const recorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const audioRef = useRef<Blob | null>(null);
  const stopResolver = useRef<((blob: Blob | null) => void) | null>(null);
  const [state, setState] = useState<RecorderState>("idle");
  const [audio, setAudio] = useState<Blob | null>(null);
  const [error, setError] = useState<string | null>(null);

  const start = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Microphone access is not available in this browser.");
      setState("error");
      return false;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm";
      const nextRecorder = new MediaRecorder(stream, { mimeType });
      chunks.current = [];
      setAudio(null);
      setError(null);
      nextRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunks.current.push(event.data);
      };
      nextRecorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        const nextAudio = new Blob(chunks.current, { type: mimeType });
        audioRef.current = nextAudio;
        setAudio(nextAudio);
        setState("ready");
        stopResolver.current?.(nextAudio);
        stopResolver.current = null;
      };
      recorder.current = nextRecorder;
      nextRecorder.start(100);
      setState("recording");
      return true;
    } catch {
      setError("Microphone permission was not granted.");
      setState("error");
      return false;
    }
  }, []);

  const stop = useCallback(() => {
    return new Promise<Blob | null>((resolve) => {
      if (recorder.current?.state !== "recording") {
        resolve(audioRef.current);
        return;
      }
      stopResolver.current = resolve;
      recorder.current.stop();
    });
  }, []);

  const reset = useCallback(() => {
    if (recorder.current?.state === "recording") recorder.current.stop();
    recorder.current = null;
    chunks.current = [];
    audioRef.current = null;
    setAudio(null);
    setError(null);
    setState("idle");
  }, []);

  return { state, audio, error, getAudio: () => audioRef.current, start, stop, reset };
}
