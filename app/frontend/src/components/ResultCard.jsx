function scoreColor(score) {
  const s = Math.max(0, Math.min(1, score));
  const hue = Math.round(120 * s);
  return {
    background: `hsl(${hue}, 65%, 50%)`,
    color: "#fff",
  };
}

function formatAuthors(authors) {
  if (!authors || authors.length === 0) return "(no authors)";
  if (authors.length > 4) {
    return authors.slice(0, 4).join(", ") + `, and ${authors.length - 4} more`;
  }
  return authors.join(", ");
}

export default function ResultCard({
  paper,
  rank,
  isStarred,
  onToggleStar,
  combined = false,
}) {
  const doi = paper.doi || "";
  const score = combined
    ? paper.rrf_score || 0
    : paper.rerank_score || 0;

  return (
    <div className="result-card">
      <div className="result-rank-col">
        <div className="result-rank">{rank}</div>
        <span className="score-badge" style={scoreColor(combined ? score / 0.082 : score)}>
          {combined ? score.toFixed(4) : score.toFixed(3)}
        </span>
      </div>
      <div className="result-body">
        <div className="result-title-row">
          <button
            className={`star-btn ${isStarred ? "active" : ""}`}
            onClick={onToggleStar}
            aria-label={isStarred ? "Remove from collection" : "Add to collection"}
          >
            {isStarred ? "★" : "☆"}
          </button>
          <div className="result-title">
            {doi ? (
              <a
                href={`https://doi.org/${doi}`}
                target="_blank"
                rel="noopener noreferrer"
              >
                {paper.title || "(no title)"}
              </a>
            ) : (
              paper.title || "(no title)"
            )}
          </div>
        </div>

        <div className="result-authors">{formatAuthors(paper.authors)}</div>

        {(paper.venue || paper.year) && (
          <div className="result-meta-line">
            <span className="result-venue-year">
              {paper.venue} {paper.year}
            </span>
          </div>
        )}

        {combined && paper.rrf_modes && (
          <div className="rrf-chips">
            {Object.entries(paper.rrf_modes).map(([mode, modeRank]) => (
              <span key={mode} className="rrf-chip">
                {mode} #{modeRank}
              </span>
            ))}
          </div>
        )}

        {paper.keywords?.length > 0 && (
          <div className="result-keywords">
            {paper.keywords.map((kw, i) => (
              <span key={i} className="keyword-chip">
                {kw}
              </span>
            ))}
          </div>
        )}

        {paper.abstract && (
          <div className="result-abstract">{paper.abstract}</div>
        )}
      </div>
    </div>
  );
}
