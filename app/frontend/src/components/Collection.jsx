import { useRef } from "react";
import { exportBib } from "../api";

export default function Collection({
  collection,
  onRemove,
  onUpdateNote,
  open,
  onToggle,
}) {
  const count = collection.length;
  const debounceTimers = useRef({});

  const handleNoteChange = (doi, value) => {
    clearTimeout(debounceTimers.current[doi]);
    debounceTimers.current[doi] = setTimeout(() => {
      onUpdateNote(doi, value);
    }, 400);
  };

  const handleExport = async () => {
    try {
      await exportBib(collection);
    } catch (e) {
      console.error("Export failed:", e);
    }
  };

  return (
    <>
      {/* Toggle tab on the right edge */}
      <button className="drawer-tab" onClick={onToggle}>
        <span className="drawer-tab-icon">★</span>
        {count > 0 && <span className="drawer-tab-badge">{count}</span>}
      </button>

      {/* Backdrop */}
      {open && <div className="drawer-backdrop" onClick={onToggle} />}

      {/* Drawer panel */}
      <div className={`drawer ${open ? "open" : ""}`}>
        <div className="drawer-header">
          <h2>
            My Collection
            {count > 0 && (
              <span className="drawer-count">
                {count} paper{count !== 1 ? "s" : ""}
              </span>
            )}
          </h2>
          <button className="drawer-close" onClick={onToggle}>
            ×
          </button>
        </div>

        <div className="drawer-body">
          {count === 0 ? (
            <div className="collection-empty">
              Click the ☆ star on any search result to add it here.
            </div>
          ) : (
            <>
              {collection.map((entry) => {
                const doi = entry.doi || entry.title;
                const authors = entry.authors || [];
                const authorsStr =
                  authors.length > 3
                    ? authors.slice(0, 3).join(", ") +
                      `, +${authors.length - 3}`
                    : authors.join(", ") || "(no authors)";

                return (
                  <div className="collection-card" key={doi}>
                    <button
                      className="collection-remove"
                      onClick={() => onRemove(doi)}
                      aria-label="Remove"
                    >
                      ×
                    </button>

                    <div className="collection-card-title">
                      {entry.doi ? (
                        <a
                          href={`https://doi.org/${entry.doi}`}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          {entry.title || "(no title)"}
                        </a>
                      ) : (
                        entry.title || "(no title)"
                      )}
                    </div>

                    <div className="collection-card-meta">
                      {authorsStr}
                      {(entry.venue || entry.year) &&
                        ` · ${entry.venue || ""} ${entry.year || ""}`}
                    </div>

                    <div className="collection-note">
                      <textarea
                        rows={1}
                        placeholder="Add a note..."
                        defaultValue={entry.user_note || ""}
                        onChange={(e) =>
                          handleNoteChange(doi, e.target.value)
                        }
                        onInput={(e) => {
                          e.target.style.height = "auto";
                          e.target.style.height = e.target.scrollHeight + "px";
                        }}
                        ref={(el) => {
                          if (el && el.value) {
                            el.style.height = "auto";
                            el.style.height = el.scrollHeight + "px";
                          }
                        }}
                      />
                    </div>
                  </div>
                );
              })}

              <div className="collection-actions">
                <button className="export-btn" onClick={handleExport}>
                  Export .bib
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
}
