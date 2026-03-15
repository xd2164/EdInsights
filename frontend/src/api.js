const BASE = "";

export async function sendMessage(message, history = []) {
  try {
    const res = await fetch(`${BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history }),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  } catch (err) {
    console.error(err);
    return null;
  }
}

export async function getPredictions(query) {
  try {
    const res = await fetch(`${BASE}/api/predictions?q=${encodeURIComponent(query)}`);
    if (!res.ok) return null;
    return res.json();
  } catch (_) {
    return null;
  }
}
