import { useState } from "react";

export default function ChatInput({ onSend, disabled }) {
  const [value, setValue] = useState("");
  const handleSubmit = (e) => {
    e.preventDefault();
    if (!value.trim() || disabled) return;
    onSend(value.trim());
    setValue("");
  };
  return (
    <form onSubmit={handleSubmit} style={{ display: "flex", gap: 10, padding: "12px 16px", background: "#16161b", borderTop: "1px solid #2a2a32" }}>
      <input
        type="text"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="Ask for citation predictions or top papers…"
        disabled={disabled}
        style={{
          flex: 1,
          padding: "12px 16px",
          borderRadius: 12,
          border: "1px solid #2a2a32",
          background: "#1c1c22",
          color: "#e4e4e8",
          fontSize: 15,
          fontFamily: "inherit",
          outline: "none",
        }}
      />
      <button type="submit" disabled={disabled || !value.trim()} style={{ padding: "12px 20px", borderRadius: 12, border: "none", background: "#2563eb", color: "#fff", fontWeight: 600, fontSize: 14, cursor: disabled ? "not-allowed" : "pointer", opacity: disabled || !value.trim() ? 0.6 : 1 }}>
        Send
      </button>
    </form>
  );
}
