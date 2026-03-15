export default function PredictionCard({ item }) {
  const title = item.title ?? item.name ?? "—";
  const authors = item.authors ?? item.author ?? "";
  const year = item.year ?? item.published ?? "";
  const score = item.score ?? item.relevance;
  const url = item.url ?? item.link ?? "#";
  return (
    <a
      href={url}
      target="_blank"
      rel="noopener noreferrer"
      style={{
        display: "block",
        padding: "12px 14px",
        background: "#16161b",
        border: "1px solid #2a2a32",
        borderRadius: 10,
        color: "#e4e4e8",
        textDecoration: "none",
        marginBottom: 8,
      }}
    >
      <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 4 }}>{title}</div>
      {(authors || year) && <div style={{ fontSize: 12, color: "#888" }}>{[authors, year].filter(Boolean).join(" · ")}</div>}
      {score != null && <div style={{ fontSize: 11, color: "#7c9cff", fontWeight: 600, marginTop: 4 }}>Predicted citations: {typeof score === "number" ? score : score}</div>}
    </a>
  );
}
