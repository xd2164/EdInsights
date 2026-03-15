import { useState, useCallback } from "react";
import { sendMessage as apiSendMessage } from "../api";

export function useChat() {
  const [messages, setMessages] = useState([
    {
      id: "welcome",
      role: "assistant",
      text: "I'm the literature prediction chatbot powered by the Community Citation Model (CCM). Ask for 'citation predictions', 'top papers', or 'predict literature' to see papers ranked by predicted future citations from the demo dataset.",
      predictions: null,
    },
  ]);
  const [loading, setLoading] = useState(false);

  const send = useCallback(async (text) => {
    if (!text?.trim()) return;
    const userMsg = { id: Date.now().toString(), role: "user", text: text.trim() };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);
    const history = messages.map((m) => ({ role: m.role, content: m.text }));
    const out = await apiSendMessage(text.trim(), history);
    setLoading(false);
    if (out) {
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          text: out.reply || out.message || out.text || "No reply.",
          predictions: out.predictions ?? out.papers ?? null,
        },
      ]);
    } else {
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          text: "Start the CCM API first: from the repo root run 'python -m uvicorn api.main:app --port 8000' (after pip install -e repro/libs/geocitmodel and api/requirements.txt). Then refresh.",
          predictions: null,
        },
      ]);
    }
  }, [messages]);

  return { messages, loading, send };
}
