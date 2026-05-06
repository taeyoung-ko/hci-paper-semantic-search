import { useRef, useState } from "react";
import { exportBib } from "../api";
import ResultCard from "./ResultCard";

const keyOf = (entry) => entry.doi || entry.title;

export default function Collection({ collection, onRemove, onUpdateNote }) {
  const count = collection.length;
  const debounceTimers = useRef({});
  const [selected, setSelected] = useState(() => new Set());

  const handleNoteChange = (doi, value) => {
    clearTimeout(debounceTimers.current[doi]);
    debounceTimers.current[doi] = setTimeout(() => {
      onUpdateNote(doi, value);
    }, 400);
  };

  const toggleSelect = (key) => {
    setSelected((prev) => {
      const next = new Set(prev);
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    });
  };

  const selectAll = () => setSelected(new Set(collection.map(keyOf)));
  const deselectAll = () => setSelected(new Set());

  const exportTarget = selected.size > 0
    ? collection.filter((e) => selected.has(keyOf(e)))
    : collection;

  const handleExport = async () => {
    if (exportTarget.length === 0) return;
    try {
      await exportBib(exportTarget);
    } catch (e) {
      console.error("Export failed:", e);
    }
  };

  return (
    <div className="collection-page">
      <div className="collection-header">
        <h2>
          My Collection
          {count > 0 && (
            <span className="collection-count">
              {count} paper{count !== 1 ? "s" : ""}
            </span>
          )}
        </h2>
        {count > 0 && (
          <div className="collection-toolbar">
            <button onClick={selectAll}>Select all</button>
            <button onClick={deselectAll} disabled={selected.size === 0}>
              Deselect all
            </button>
            <button className="export-btn" onClick={handleExport}>
              {selected.size > 0
                ? `Export selected (${selected.size}) .bib`
                : `Export all (${count}) .bib`}
            </button>
          </div>
        )}
      </div>

      <div className="collection-body">
        {count === 0 ? (
          <div className="collection-empty">
            Click the ☆ star on any search result to add it here.
          </div>
        ) : (
          collection.map((entry, i) => {
            const k = keyOf(entry);
            return (
              <ResultCard
                key={k}
                paper={entry}
                rank={i + 1}
                isStarred={true}
                onToggleStar={() => onRemove(k)}
                selectable={true}
                isSelected={selected.has(k)}
                onToggleSelect={() => toggleSelect(k)}
                hideScore={true}
              >
                <div className="collection-note">
                  <textarea
                    rows={1}
                    placeholder="Add a note..."
                    defaultValue={entry.user_note || ""}
                    onChange={(e) => handleNoteChange(k, e.target.value)}
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
              </ResultCard>
            );
          })
        )}
      </div>
    </div>
  );
}
