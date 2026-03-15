export default function ChatMessage({ message }) {
  const isUser = message.role === "user";
  return (
    <div style={{ display: "flex", justifyContent: isUser ? "flex-end" : "flex-start", marginBottom: 16 }}>
      <div
        style={{
          maxWidth: "85%",
          padding: "12px 16px",
          borderRadius: 16,
          background: isUser ? "#2563eb" : "#1e1e24",
          border: isUser ? "none" : "1px solid #2a2a32",
          color: isUser ? "#fff" : "#e4e4e8",
          fontFamily: "Source Serif 4",
          fontSize: 15,
          lineHeight: 1.55,
        }}
      >
        <div style={{ whiteSpace: "pre-wrap", wordBreak: "break-word" }}>{message.text}</div>
      </div>
    </div>
  );
}
