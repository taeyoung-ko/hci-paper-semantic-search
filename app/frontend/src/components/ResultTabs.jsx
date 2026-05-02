import { useState } from "react";
import ResultCard from "./ResultCard";

export default function ResultTabs({ data, collectedDois, onToggleStar }) {
  const { tabs, combined } = data;

  const allTabs = [];
  if (combined) {
    allTabs.push({ key: "__combined", label: combined.label, results: combined.results, isCombined: true });
  }
  tabs.forEach((tab) => {
    allTabs.push({ key: tab.mode, label: tab.label, results: tab.results, mode: tab.mode, query: tab.query, isCombined: false });
  });

  const [activeIdx, setActiveIdx] = useState(0);

  if (allTabs.length === 0) {
    return <div className="results-placeholder">No results found.</div>;
  }

  const activeTab = allTabs[activeIdx] || allTabs[0];

  return (
    <div>
      <div className="tabs-bar">
        {allTabs.map((tab, idx) => (
          <button
            key={tab.key}
            className={`tab-btn ${idx === activeIdx ? "active" : ""}`}
            onClick={() => setActiveIdx(idx)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div>
        {activeTab.results.map((paper, i) => (
          <ResultCard
            key={paper.doi || paper.title || i}
            paper={paper}
            rank={i + 1}
            combined={activeTab.isCombined}
            isStarred={collectedDois.has(paper.doi || paper.title)}
            onToggleStar={() =>
              onToggleStar(paper, activeTab.mode || "combined", activeTab.query || "")
            }
          />
        ))}
      </div>
    </div>
  );
}
