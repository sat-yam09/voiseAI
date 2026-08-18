"use client";

import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { CSSProperties, FormEvent, useMemo, useState } from "react";
import { submitQuery } from "@/lib/api-client";
import type { QueryResponse } from "@/lib/contracts";
import { useAudioRecorder } from "@/hooks/use-audio-recorder";
import { useLiveTranscript } from "@/hooks/use-live-transcript";
import { ArrowIcon, KeyboardIcon, MicIcon, SparkIcon, StopIcon } from "./icons";
import styles from "./voice-rag-shell.module.css";

type Mode = "idle" | "voice" | "text";
type Pipeline = "idle" | "recording" | "processing" | "complete" | "error";

const transition = { duration: 0.26, ease: [0.16, 1, 0.3, 1] as const };

export function VoiceRagShell() {
  const shouldReduceMotion = useReducedMotion();
  const recorder = useAudioRecorder();
  const liveTranscript = useLiveTranscript();
  const [mode, setMode] = useState<Mode>("idle");
  const [pipeline, setPipeline] = useState<Pipeline>("idle");
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const status = useMemo(() => {
    if (pipeline === "recording") return "Listening";
    if (pipeline === "processing") return "Finding a grounded answer";
    if (pipeline === "complete") return "Answer ready";
    if (pipeline === "error") return "Something needs attention";
    return "Ask your knowledge base";
  }, [pipeline]);

  function clearResult() {
    setResult(null);
    setError(null);
    setPipeline("idle");
  }

  async function handleSubmit(event?: FormEvent) {
    event?.preventDefault();
    if (!query.trim() || pipeline === "processing") return;
    setError(null);
    setPipeline("processing");
    try {
      const nextResult = await submitQuery({ text: query });
      setResult(nextResult);
      setPipeline("complete");
      setMode("idle");
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "The query could not be completed.");
      setPipeline("error");
    }
  }

  async function handleVoiceStart() {
    clearResult();
    setMode("voice");
    liveTranscript.start();
    const started = await recorder.start();
    if (started) setPipeline("recording");
  }

  async function handleVoiceStop() {
    liveTranscript.stop();
    setPipeline("processing");
    const audio = await recorder.stop();
    if (!audio) return;
    try {
      const nextResult = await submitQuery({ audio });
      setResult(nextResult);
      setPipeline("complete");
      setMode("idle");
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "The voice query could not be completed.");
      setPipeline("error");
    }
  }

  function handleCancel() {
    recorder.reset();
    liveTranscript.reset();
    setMode("idle");
    setPipeline("idle");
    setError(null);
  }

  return (
    <main className={styles.page}>
      <header className={styles.header}>
        <div className={styles.wordmark}><span className={styles.wordmarkMark}><SparkIcon size={14} /></span>voice-rag</div>
        <div className={styles.headerMeta}><span className={styles.liveDot} /> local demo <span className={styles.headerDivider} /> v0.1</div>
      </header>

      <section className={`${styles.workspace} ${result ? styles.workspaceWithResult : ""}`} aria-live="polite">
        <AnimatePresence mode="wait" initial={false}>
          {!result && pipeline !== "error" ? (
            <motion.div key="welcome" initial={shouldReduceMotion ? false : { opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={shouldReduceMotion ? undefined : { opacity: 0, y: -8 }} transition={transition} className={styles.welcome}>
              <p className={styles.kicker}>A voice interface for retrieved knowledge</p>
              <h1>Ask clearly.<br /><em>Know why.</em></h1>
              <p className={styles.subhead}>Speak or type a question. Every answer is paired with the context that supports it.</p>
            </motion.div>
          ) : null}
        </AnimatePresence>

        {result ? <AnswerCard result={result} onReset={handleCancel} /> : null}

        {pipeline === "error" && error ? <div className={styles.errorMessage} role="alert"><span>{error}</span><button onClick={handleCancel}>Reset</button></div> : null}
      </section>

      <div className={styles.composerDock}>
        <AnimatePresence mode="wait" initial={false}>
          {mode === "idle" ? (
            <motion.div key="idle" layoutId="composer" className={styles.idleComposer} transition={transition}>
              <button className={styles.modeButton} onClick={handleVoiceStart} aria-label="Ask with voice"><span className={styles.modeIcon}><MicIcon /></span><span>Voice</span></button>
              <span className={styles.modeRule} />
              <button className={styles.modeButton} onClick={() => { clearResult(); setMode("text"); }} aria-label="Ask with text"><span className={styles.modeIcon}><KeyboardIcon /></span><span>Text</span></button>
            </motion.div>
          ) : mode === "voice" ? (
            <motion.div key="voice" layoutId="composer" className={`${styles.activeComposer} ${styles.voiceComposer}`} transition={transition}>
              <button className={styles.iconButton} onClick={handleCancel} aria-label="Cancel recording">×</button>
              <div className={styles.recordingBody}>
                <div className={styles.recordingLabel}><span className={styles.recordingPulse} />{status}</div>
                <div className={styles.liveTranscript} aria-live="polite" aria-label="Live transcript">
                  {liveTranscript.text ? <motion.span key={liveTranscript.text} initial={{ opacity: 0, filter: "blur(9px)", y: 3 }} animate={{ opacity: 1, filter: "blur(0px)", y: 0 }} transition={{ duration: 0.18, ease: [0.16, 1, 0.3, 1] }}>{liveTranscript.text}</motion.span> : <span className={styles.transcriptPlaceholder}>{liveTranscript.supported ? "Start speaking…" : "Live preview will appear after connection"}</span>}
                </div>
                <ListeningSignal active={recorder.state === "recording"} />
              </div>
              <button className={styles.sendButton} onClick={handleVoiceStop} aria-label="Send voice question"><StopIcon /></button>
            </motion.div>
          ) : (
            <motion.form key="text" layoutId="composer" className={`${styles.activeComposer} ${styles.textComposer}`} onSubmit={handleSubmit} transition={transition}>
              <button type="button" className={styles.iconButton} onClick={handleCancel} aria-label="Close text input">×</button>
              <input autoFocus value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Ask your knowledge base…" aria-label="Question" />
              <button className={styles.sendButton} type="submit" aria-label="Send question" disabled={!query.trim() || pipeline === "processing"}><ArrowIcon /></button>
            </motion.form>
          )}
        </AnimatePresence>
        <p className={styles.helperText}>{pipeline === "processing" ? "Searching the indexed knowledge base…" : "Your answer stays grounded in retrieved context."}</p>
      </div>

      <footer className={styles.footer}><span>MSMARCO-XI</span><span className={styles.footerDot}>·</span><span>Indic language ready</span></footer>
    </main>
  );
}

function ListeningSignal({ active }: { active: boolean }) {
  return <div className={`${styles.signalVisual} ${active ? styles.signalActive : ""}`} aria-hidden="true">
    <span className={styles.signalRing} />
    <span className={`${styles.signalRing} ${styles.signalRingSecond}`} />
    <svg className={styles.signalRibbon} viewBox="0 0 220 32" preserveAspectRatio="none">
      <path d="M0 16 C20 6 38 6 55 16 S89 26 110 16 S146 6 165 16 S200 26 220 16" />
      <path d="M0 16 C18 24 39 24 57 16 S91 8 111 16 S147 24 166 16 S201 8 220 16" />
    </svg>
    <div className={`${styles.waveform} ${active ? styles.waveformActive : ""}`}>{Array.from({ length: 24 }).map((_, index) => <i key={index} style={{ "--i": index } as CSSProperties} />)}</div>
  </div>;
}

function AnswerCard({ result, onReset }: { result: QueryResponse; onReset: () => void }) {
  return <motion.article className={styles.answerCard} initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={transition}>
    <div className={styles.answerTopline}><span className={`${styles.groundedBadge} ${result.grounded ? "" : styles.notGrounded}`}><span />{result.grounded ? "Grounded answer" : "Needs review"}</span><button className={styles.newQuestion} onClick={onReset}>New question <ArrowIcon size={14} /></button></div>
    <p className={styles.answerText}>{result.answer}</p>
    <div className={styles.sourceRow}>{result.sources.map((source) => <div className={styles.source} key={source.id}><span className={styles.sourceIndex}>0{result.sources.indexOf(source) + 1}</span><span>{source.label}</span></div>)}<span className={styles.latency}>{result.latency_ms.total} ms</span></div>
  </motion.article>;
}
