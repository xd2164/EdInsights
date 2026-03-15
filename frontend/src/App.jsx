import { useRef, useEffect } from "react";
import useChat from "./hooks/useChat";
import ChatMessage from "./components/ChatMessage";
import ChatInput from "./components/ChatInput";
import PredictionCard from "./components/PredictionCard";

export default function App() {
  const { messages, loading, send } = useChat();
  const bottomRef = useRef(null);
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  return (
    <div style={{ height: "100vh", display: "flex", flexDirection: "column", background: "#0f0f12" }}>
      <header style={{ padding: "14px 20px", borderBottom: "1px solid #2a2a32", background: "#16161b" }}>
        <h1 style={{ margin: 0, fontSize: 18, fontWeight: 700, color: "#e4e4e8", fontFamily: "Source Serif 4" }}>
          Literature Prediction · Community Citation Model
        </h1>
        <p style={{ margin: "4px 0 0 0", fontSize: 12, color: "#888" }}>
          Chat for citation predictions from the CCM demo
        </p>
      </header>
      <div style={{ flex: 1, overflowY: "auto", padding: "20px 24px" }}>
        {messages.map((msg) => (
          <div key={msg.id}>
            <ChatMessage message={msg} />
            {msg.predictions && msg.predictions.length > 0 && (
              <div style={{ marginLeft: 0, marginBottom: 20, maxWidth: 520 }}>
                <div style={{ fontSize: 11, color: "#888", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>Predicted literature (CCM)</div>
                {msg.predictions.map((p, i) => (
                  <PredictionCard key={i} item={p} />
                ))}
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div style={{ display: "flex", justifyContent: "flex-start", marginBottom: 16 }}>
            <div style={{ padding: "12px 16px", borderRadius: 16, background: "#1e1e24", border: "1px solid #2a2a32", color: "#888", fontSize: 14 }}>Running CCM prediction…</div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
      <ChatInput onSend={send} disabled={loading} />
    </div>
  );
}
