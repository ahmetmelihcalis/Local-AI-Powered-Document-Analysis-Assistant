import { DragEvent, FormEvent, useEffect, useRef, useState } from "react";
import {
  ChatSource,
  deleteDocument,
  DocumentResponse,
  getDocuments,
  getHealth,
  Language,
  sendQuestion,
  uploadDocument,
} from "./api";
import styles from "./App.module.css";

type ConnectionStatus = "checking" | "ready" | "error";
const SUPPORTED_DOCUMENT_EXTENSIONS = [".txt", ".md", ".pdf", ".docx"];

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: ChatSource[];
  durationMs?: number;
};

export default function App() {
  const [status, setStatus] = useState<ConnectionStatus>("checking");
  const [foundryAvailable, setFoundryAvailable] = useState(false);
  const [question, setQuestion] = useState("");
  const [language, setLanguage] = useState<Language>("tr");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isSending, setIsSending] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const [documents, setDocuments] = useState<DocumentResponse[]>([]);
  const [documentsLoading, setDocumentsLoading] = useState(true);
  const [documentsError, setDocumentsError] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [isDraggingFile, setIsDraggingFile] = useState(false);
  const [deletingDocumentId, setDeletingDocumentId] = useState<number | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    getHealth()
      .then((health) => {
        setStatus("ready");
        setFoundryAvailable(health.foundryLocal === "available");
      })
      .catch(() => setStatus("error"));
  }, []);

  useEffect(() => {
    getDocuments()
      .then(setDocuments)
      .catch(() => setDocumentsError(true))
      .finally(() => setDocumentsLoading(false));
  }, []);

  const statusText = {
    checking: "Checking connection",
    ready: "Local AI is ready",
    error: "Backend connection failed",
  }[status];

  function selectDocumentFile(file: File | null) {
    if (!file) {
      setSelectedFile(null);
      return;
    }

    const extension = `.${file.name.split(".").pop()?.toLowerCase()}`;
    if (!SUPPORTED_DOCUMENT_EXTENSIONS.includes(extension)) {
      setSelectedFile(null);
      setUploadError("Only TXT, Markdown, PDF and DOCX files are supported.");
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
      return;
    }

    setSelectedFile(file);
    setUploadError(null);
  }

  function handleFileDrop(event: DragEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsDraggingFile(false);
    selectDocumentFile(event.dataTransfer.files[0] ?? null);
  }

  async function handleDocumentUpload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!selectedFile || isUploading) {
      return;
    }

    setIsUploading(true);
    setUploadError(null);

    try {
      const uploadedDocument = await uploadDocument(selectedFile);
      setDocuments((current) => [uploadedDocument, ...current]);
      setDocumentsError(false);
      setSelectedFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } catch (error) {
      setUploadError(
        error instanceof Error ? error.message : "The document could not be uploaded.",
      );
    } finally {
      setIsUploading(false);
    }
  }

  async function handleDocumentDelete(documentId: number) {
    if (deletingDocumentId !== null) {
      return;
    }

    setDeletingDocumentId(documentId);
    setDeleteError(null);

    try {
      await deleteDocument(documentId);
      setDocuments((current) =>
        current.filter((document) => document.id !== documentId),
      );
    } catch (error) {
      setDeleteError(
        error instanceof Error ? error.message : "The document could not be deleted.",
      );
    } finally {
      setDeletingDocumentId(null);
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmedQuestion = question.trim();

    if (!trimmedQuestion || isSending || status !== "ready") {
      return;
    }

    setMessages((current) => [
      ...current,
      {
        id: crypto.randomUUID(),
        role: "user",
        content: trimmedQuestion,
      },
    ]);
    setQuestion("");
    setChatError(null);
    setIsSending(true);

    try {
      const response = await sendQuestion(trimmedQuestion, language);
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: response.answer,
          sources: response.sources,
          durationMs: response.durationMs,
        },
      ]);
    } catch (error) {
      setChatError(
        error instanceof Error ? error.message : "An unexpected error occurred.",
      );
    } finally {
      setIsSending(false);
    }
  }

  return (
    <div className={styles.appShell}>
      <header className={styles.header}>
        <div>
          <h1>Local RAG Assistant</h1>
        </div>

        <div className={styles.headerActions}>
          <label className={styles.languageSelect}>
            <span>Language</span>
            <select
              value={language}
              onChange={(event) => setLanguage(event.target.value as Language)}
              aria-label="Answer language"
            >
              <option value="tr">Turkish</option>
              <option value="en">English</option>
            </select>
          </label>
          {(status !== "ready" || !foundryAvailable) && (
            <div
              className={styles.status}
              data-state={status === "ready" ? "error" : status}
            >
              <span className={styles.statusDot} aria-hidden="true" />
              <span>
                {status === "ready"
                  ? "Foundry Local is unavailable"
                  : statusText}
              </span>
            </div>
          )}
        </div>
      </header>

      <main className={styles.workspace}>
        <aside className={styles.documentPanel} aria-label="Documents">
          <div className={styles.documentPanelHeader}>
            <h2>Documents</h2>
            <span>{documents.length}/20</span>
          </div>

          <form
            className={`${styles.uploadForm} ${isDraggingFile ? styles.dragging : ""}`}
            onSubmit={handleDocumentUpload}
            onDragEnter={(event) => {
              event.preventDefault();
              setIsDraggingFile(true);
            }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={(event) => {
              if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
                setIsDraggingFile(false);
              }
            }}
            onDrop={handleFileDrop}
          >
            <p className={styles.dropHint}>Drop a document here or choose a file</p>
            <input
              ref={fileInputRef}
              type="file"
              accept=".txt,.md,.pdf,.docx"
              aria-label="Choose a document"
              onChange={(event) =>
                selectDocumentFile(event.target.files?.[0] ?? null)
              }
              disabled={isUploading}
            />
            <button
              type="submit"
              disabled={
                !selectedFile ||
                isUploading ||
                status !== "ready" ||
                !foundryAvailable
              }
            >
              {isUploading ? "Uploading…" : "Upload"}
            </button>
          </form>
          {uploadError && (
            <p className={styles.documentError}>{uploadError}</p>
          )}
          {deleteError && (
            <p className={styles.documentError}>{deleteError}</p>
          )}

          {documentsLoading && (
            <p className={styles.documentMessage}>Loading documents…</p>
          )}
          {documentsError && (
            <p className={styles.documentError}>Documents could not be loaded.</p>
          )}
          {!documentsLoading && !documentsError && documents.length === 0 && (
            <p className={styles.documentMessage}>No documents uploaded yet.</p>
          )}

          {documents.length > 0 && (
            <ul className={styles.documentList}>
              {documents.map((document) => (
                <li className={styles.documentItem} key={document.id}>
                  <div className={styles.documentItemHeader}>
                    <strong title={document.originalName}>
                      {document.originalName}
                    </strong>
                    <button
                      type="button"
                      onClick={() => handleDocumentDelete(document.id)}
                      disabled={deletingDocumentId !== null}
                      aria-label={`Delete ${document.originalName}`}
                    >
                      {deletingDocumentId === document.id ? "Deleting…" : "Delete"}
                    </button>
                  </div>
                  <small className={styles.documentDetails}>
                    {document.fileType.toUpperCase()} · {document.chunkCount} chunks ·{" "}
                    {document.status}
                  </small>
                </li>
              ))}
            </ul>
          )}
        </aside>

        <section className={styles.chatPanel}>
          <section className={styles.messages} aria-live="polite">
          {messages.length === 0 && (
            <div className={styles.emptyState}>
              <div className={styles.emptyIcon} aria-hidden="true">AI</div>
              <h2>Ask questions about your documents</h2>
              <p>
                Answers are based only on information found in your uploaded
                documents and include the sources used.
              </p>
            </div>
          )}

          {messages.map((message) => (
            <article
              className={`${styles.message} ${styles[message.role]}`}
              key={message.id}
            >
              <span className={styles.messageRole}>
                {message.role === "user" ? "You" : "Assistant"}
              </span>
              <p>{message.content}</p>

              {message.sources && message.sources.length > 0 && (
                <details className={styles.sources}>
                  <summary>
                    Show {message.sources.length} source
                    {message.sources.length === 1 ? "" : "s"}
                  </summary>
                  <div className={styles.sourceList}>
                    {message.sources.map((source) => (
                      <article className={styles.sourceCard} key={`${message.id}-${source.documentId}-${source.excerpt}`}>
                        <div className={styles.sourceHeader}>
                          <strong>{source.fileName}</strong>
                          <span>{Math.round(source.score * 100)}% similarity</span>
                        </div>
                        {(source.page !== null || source.section) && (
                          <small>
                            {source.page !== null ? `Page ${source.page}` : ""}
                            {source.page !== null && source.section ? " · " : ""}
                            {source.section ?? ""}
                          </small>
                        )}
                        <p>{source.excerpt}</p>
                      </article>
                    ))}
                  </div>
                </details>
              )}

              {message.durationMs !== undefined && (
                <small className={styles.duration}>
                  Answered in {(message.durationMs / 1000).toFixed(1)} seconds
                </small>
              )}
            </article>
          ))}

          {isSending && (
            <div className={`${styles.message} ${styles.assistant}`}>
              <span className={styles.messageRole}>Assistant</span>
              <p className={styles.thinking}>Searching the documents…</p>
            </div>
          )}
          </section>

          <div className={styles.composerArea}>
          {chatError && <p className={styles.errorMessage}>{chatError}</p>}
          <form className={styles.composer} onSubmit={handleSubmit}>
            <label className={styles.visuallyHidden} htmlFor="question">
              Ask your documents a question
            </label>
            <textarea
              id="question"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask a question about your documents…"
              rows={2}
              maxLength={2000}
              disabled={isSending}
            />
            <button
              type="submit"
              disabled={!question.trim() || isSending || status !== "ready" || !foundryAvailable}
            >
              {isSending ? "Answering" : "Send"}
            </button>
          </form>
          <p className={styles.disclaimer}>
            The local model answers only from the selected document passages.
          </p>
          </div>
        </section>
      </main>
    </div>
  );
}
