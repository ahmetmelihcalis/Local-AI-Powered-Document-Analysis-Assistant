import { DragEvent, FormEvent, useEffect, useRef, useState } from "react";
import {
  ChatSource,
  deleteDocument,
  DocumentResponse,
  getDocuments,
  getHealth,
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

const LEGAL_EXAMPLE_QUESTIONS = [
  "What is an AI system?",
  "When must a personal data breach be communicated to the data subject?",
  "When must a controller notify the supervisory authority of a personal data breach?",
];

const GENERAL_EXAMPLE_QUESTION = "What is this document about?";

function legalReference(source: ChatSource): string | null {
  if (!source.article) {
    return null;
  }

  return [
    source.article,
    source.paragraph ? `(${source.paragraph})` : "",
    source.point ? `(${source.point})` : "",
    source.subpoint ? `(${source.subpoint})` : "",
  ].join("");
}

export default function App() {
  const [status, setStatus] = useState<ConnectionStatus>("checking");
  const [foundryAvailable, setFoundryAvailable] = useState(false);
  const [question, setQuestion] = useState("");
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

  async function askQuestion(questionToAsk: string) {
    const trimmedQuestion = questionToAsk.trim();

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
      const response = await sendQuestion(trimmedQuestion);
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

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void askQuestion(question);
  }

  return (
    <div className={styles.appShell}>
      <header className={styles.header}>
        <a className={styles.headerBrand} href="/" aria-label="Go to home">
          <span>Local AI-Powered</span>
          <h1>Document Analysis Assistant</h1>
          <p>Optimized for EU AI Act and GDPR documents</p>
        </a>

        <div className={styles.headerActions}>
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
              className={styles.fileInput}
              id="document-file"
              type="file"
              accept=".txt,.md,.pdf,.docx"
              aria-label="Choose a document"
              onChange={(event) =>
                selectDocumentFile(event.target.files?.[0] ?? null)
              }
              disabled={isUploading}
            />
            <div className={styles.fileSelector}>
              <label className={styles.fileSelectButton} htmlFor="document-file">
                Choose file
              </label>
              <span className={styles.fileName} title={selectedFile?.name}>
                {selectedFile?.name ?? "No file chosen"}
              </span>
            </div>
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
                      title={`Delete ${document.originalName}`}
                    >
                      {deletingDocumentId === document.id ? (
                        "…"
                      ) : (
                        <svg viewBox="0 0 24 24" aria-hidden="true">
                          <path d="M4 7h16M10 11v6M14 11v6M9 7l1-2h4l1 2M6 7l1 13h10l1-13" />
                        </svg>
                      )}
                    </button>
                  </div>
                  <small className={styles.documentDetails}>
                    <span>
                      {document.fileType.toUpperCase()} · {document.chunkCount} chunks
                    </span>
                    <span data-status={document.status}>{document.status}</span>
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
              <h2>Ask Questions About Your Documents</h2>
              <p>
                Answers are grounded in your uploaded documents. Supporting
                sources are shown when available.
              </p>
              <div className={styles.exampleQuestions}>
                <span>Try a legal example</span>
                <div>
                  {LEGAL_EXAMPLE_QUESTIONS.map((example) => (
                    <button
                      key={example}
                      type="button"
                      onClick={() => void askQuestion(example)}
                      disabled={status !== "ready" || !foundryAvailable}
                    >
                      {example}
                    </button>
                  ))}
                </div>
                <span className={styles.generalExampleLabel}>
                  Or ask about any uploaded document
                </span>
                <div>
                  <button
                    type="button"
                    onClick={() => void askQuestion(GENERAL_EXAMPLE_QUESTION)}
                    disabled={status !== "ready" || !foundryAvailable}
                  >
                    {GENERAL_EXAMPLE_QUESTION}
                  </button>
                </div>
              </div>
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
                    {message.sources.map((source) => {
                      const clause = legalReference(source);
                      return (
                        <article
                          className={styles.sourceCard}
                          key={`${message.id}-${source.documentId}-${source.excerpt}`}
                        >
                          <div className={styles.sourceHeader}>
                            <strong>{source.fileName}</strong>
                            <span>
                              Source relevance: {Math.round(source.score * 100)}%
                            </span>
                          </div>
                          {(source.page !== null || source.section || clause) && (
                            <small>
                              {clause ?? ""}
                              {clause && (source.page !== null || source.section)
                                ? " · "
                                : ""}
                              {source.page !== null ? `Page ${source.page}` : ""}
                              {source.page !== null && source.section ? " · " : ""}
                              {source.section ?? ""}
                            </small>
                          )}
                          <p>{source.excerpt}</p>
                        </article>
                      );
                    })}
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
              rows={1}
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
          {messages.length > 0 && (
            <p className={styles.disclaimer}>
              The local model answers only from the selected document passages.
            </p>
          )}
          </div>
        </section>
      </main>
    </div>
  );
}
