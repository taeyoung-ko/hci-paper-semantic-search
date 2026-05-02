import { useState, useCallback, useRef } from "react";

const COMPONENTS = [
  { key: "background", label: "Background", placeholder: "The research background or domain context your work sits in" },
  { key: "gap", label: "Research Gap", placeholder: "The limitation of prior work or research gap your study addresses" },
  { key: "solution", label: "Solution", placeholder: "The system, tool, or artifact you propose" },
  { key: "method", label: "Approach/Method", placeholder: "The methodology, technique, or approach you use, regardless of domain" },
  { key: "findings", label: "Findings", placeholder: "The experimental findings or results of your study" },
];

function AutoTextarea({ value, onChange, onKeyDown, placeholder, id }) {
  const ref = useRef(null);

  const handleInput = useCallback(
    (e) => {
      onChange(e);
      const el = ref.current;
      if (el) {
        el.style.height = "auto";
        el.style.height = el.scrollHeight + "px";
      }
    },
    [onChange]
  );

  return (
    <textarea
      ref={ref}
      id={id}
      rows={1}
      placeholder={placeholder}
      value={value}
      onChange={handleInput}
      onKeyDown={onKeyDown}
    />
  );
}

export function QueryBar({ queries, setQuery, onSearch, searching, beforeFields, children }) {
  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSearch();
    }
  };

  return (
    <div className="query-bar">
      {beforeFields}
      <div className="query-fields">
        {COMPONENTS.map(({ key, label, placeholder }) => (
          <div className="query-field" key={key}>
            <label htmlFor={`q-${key}`}>{label}</label>
            <AutoTextarea
              id={`q-${key}`}
              placeholder={placeholder}
              value={queries[key] || ""}
              onChange={(e) => setQuery(key, e.target.value)}
              onKeyDown={handleKey}
            />
          </div>
        ))}
      </div>
      {children}
      <button
        className="search-btn"
        onClick={onSearch}
        disabled={searching}
      >
        {searching ? "Searching..." : "Search"}
      </button>
    </div>
  );
}

export function VenueFilter({ venues, selectedVenues, toggleVenue }) {
  return (
    <div className="query-field venue-field">
      <label>Venues</label>
      <div className="venue-chips">
        {venues.map((v) => (
          <button
            type="button"
            key={v}
            className={`venue-chip ${selectedVenues.has(v) ? "active" : ""}`}
            onClick={() => toggleVenue(v)}
          >
            {v}
          </button>
        ))}
      </div>
    </div>
  );
}

export function AdvancedOptions({ retrieveK, setRetrieveK, rerankK, setRerankK }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="advanced-options">
      <div className="query-field">
        <label
          className="advanced-toggle"
          onClick={() => setOpen((p) => !p)}
        >
          <span className={`advanced-chevron ${open ? "open" : ""}`}>▶</span>
          Advanced
        </label>
        {open && (
          <div className="sliders-row">
            <div className="slider-group">
              <span className="options-label">Retrieval K</span>
              <input
                type="range"
                min={100}
                max={5000}
                step={100}
                value={retrieveK}
                onChange={(e) => setRetrieveK(Number(e.target.value))}
              />
              <div className="slider-val">{retrieveK}</div>
            </div>
            <div className="slider-group">
              <span className="options-label">Rerank K</span>
              <input
                type="range"
                min={10}
                max={500}
                step={10}
                value={rerankK}
                onChange={(e) => setRerankK(Number(e.target.value))}
              />
              <div className="slider-val">{rerankK}</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
