import { useState, useEffect, useCallback, useRef } from "react";
import { fetchFilterOptions, searchPapers } from "./api";
import { QueryBar, UnifiedQueryBar, VenueFilter, AdvancedOptions } from "./components/SearchForm";
import ResultTabs from "./components/ResultTabs";
import Collection from "./components/Collection";
import ManageData from "./components/AddVenue";

// ── localStorage helpers ──
const STORAGE_KEYS = {
  collection: "hcips_collection",
  queries: "hcips_queries",
  venues: "hcips_venues",
  retrieveK: "hcips_retrieve_k",
  rerankK: "hcips_rerank_k",
  searchResult: "hcips_search_result",
  unifiedQuery: "hcips_unified_query",
  unifiedResult: "hcips_unified_result",
};

function loadJSON(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

function saveJSON(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {}
}

export default function App() {
  const [filterOpts, setFilterOpts] = useState(null);
  const [searchResult, setSearchResult] = useState(() =>
    loadJSON(STORAGE_KEYS.searchResult, null)
  );
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState("");
  const [collection, setCollection] = useState(() =>
    loadJSON(STORAGE_KEYS.collection, [])
  );
  const [page, setPage] = useState("unified");

  // Search form state (lifted here so QueryBar and SearchOptions can share)
  const [queries, setQueries] = useState(() =>
    loadJSON(STORAGE_KEYS.queries, {})
  );
  const [selectedVenues, setSelectedVenues] = useState(new Set());
  const [retrieveK, setRetrieveK] = useState(() =>
    loadJSON(STORAGE_KEYS.retrieveK, 1000)
  );
  const [rerankK, setRerankK] = useState(() =>
    loadJSON(STORAGE_KEYS.rerankK, 100)
  );

  // Unified search state (separate from component-based search)
  const [unifiedQuery, setUnifiedQuery] = useState(() =>
    loadJSON(STORAGE_KEYS.unifiedQuery, "")
  );
  const [unifiedResult, setUnifiedResult] = useState(() =>
    loadJSON(STORAGE_KEYS.unifiedResult, null)
  );
  const [unifiedSearching, setUnifiedSearching] = useState(false);

  const venuesInitialized = useRef(false);

  useEffect(() => {
    fetchFilterOptions()
      .then((opts) => {
        setFilterOpts(opts);
        const saved = loadJSON(STORAGE_KEYS.venues, null);
        if (saved && Array.isArray(saved)) {
          setSelectedVenues(new Set(saved));
        } else {
          setSelectedVenues(new Set(opts.venues));
        }
        venuesInitialized.current = true;
      })
      .catch(() => setError("Failed to connect to backend."));
  }, []);

  const refreshFilterOptions = useCallback(async () => {
    try {
      const opts = await fetchFilterOptions();
      setFilterOpts(opts);
      setSelectedVenues((prev) => {
        const newOnes = opts.venues.filter((v) => !prev.has(v));
        if (newOnes.length === 0) return prev;
        const next = new Set(prev);
        newOnes.forEach((v) => next.add(v));
        return next;
      });
    } catch {}
  }, []);

  const handleClear = useCallback(() => {
    if (!window.confirm("Clear search inputs, results, and starred collection?")) return;
    setQueries({});
    setUnifiedQuery("");
    setSearchResult(null);
    setUnifiedResult(null);
    setCollection([]);
    setError("");
  }, []);

  // Persist to localStorage on change
  useEffect(() => { saveJSON(STORAGE_KEYS.collection, collection); }, [collection]);
  useEffect(() => { saveJSON(STORAGE_KEYS.queries, queries); }, [queries]);
  useEffect(() => { saveJSON(STORAGE_KEYS.retrieveK, retrieveK); }, [retrieveK]);
  useEffect(() => { saveJSON(STORAGE_KEYS.rerankK, rerankK); }, [rerankK]);
  useEffect(() => { saveJSON(STORAGE_KEYS.searchResult, searchResult); }, [searchResult]);
  useEffect(() => { saveJSON(STORAGE_KEYS.unifiedQuery, unifiedQuery); }, [unifiedQuery]);
  useEffect(() => { saveJSON(STORAGE_KEYS.unifiedResult, unifiedResult); }, [unifiedResult]);
  useEffect(() => {
    if (venuesInitialized.current) {
      saveJSON(STORAGE_KEYS.venues, [...selectedVenues]);
    }
  }, [selectedVenues]);

  const setQuery = useCallback((key, val) => {
    setQueries((prev) => ({ ...prev, [key]: val }));
  }, []);

  const toggleVenue = useCallback((v) => {
    setSelectedVenues((prev) => {
      const next = new Set(prev);
      next.has(v) ? next.delete(v) : next.add(v);
      return next;
    });
  }, []);

  const handleSearch = useCallback(async () => {
    setSearching(true);
    setError("");
    setSearchResult(null);
    try {
      const result = await searchPapers({
        background: queries.background || "",
        gap: queries.gap || "",
        solution: queries.solution || "",
        method: queries.method || "",
        findings: queries.findings || "",
        venues: [...selectedVenues],
        retrieve_k: retrieveK,
        rerank_k: rerankK,
      });
      setSearchResult(result);
      if (result.errors?.length) {
        setError(result.errors.join(" | "));
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setSearching(false);
    }
  }, [queries, selectedVenues, retrieveK, rerankK]);

  const handleUnifiedSearch = useCallback(async () => {
    setUnifiedSearching(true);
    setError("");
    setUnifiedResult(null);
    try {
      const result = await searchPapers({
        topic: unifiedQuery,
        venues: [...selectedVenues],
        retrieve_k: retrieveK,
        rerank_k: rerankK,
      });
      setUnifiedResult(result);
      if (result.errors?.length) {
        setError(result.errors.join(" | "));
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setUnifiedSearching(false);
    }
  }, [unifiedQuery, selectedVenues, retrieveK, rerankK]);

  const toggleStar = useCallback((paper, mode, query) => {
    setCollection((prev) => {
      const doi = paper.doi || paper.title;
      const idx = prev.findIndex((e) => (e.doi || e.title) === doi);
      if (idx >= 0) {
        return prev.filter((_, i) => i !== idx);
      }
      return [
        ...prev,
        {
          ...paper,
          source: {
            mode,
            query,
            rerank_score: paper.rerank_score,
            added_at: new Date().toISOString(),
          },
          user_note: "",
        },
      ];
    });
  }, []);

  const updateNote = useCallback((doi, note) => {
    setCollection((prev) =>
      prev.map((e) =>
        (e.doi || e.title) === doi ? { ...e, user_note: note } : e
      )
    );
  }, []);

  const removeFromCollection = useCallback((doi) => {
    setCollection((prev) =>
      prev.filter((e) => (e.doi || e.title) !== doi)
    );
  }, []);

  const collectedDois = new Set(
    collection.map((e) => e.doi || e.title)
  );

  if (!filterOpts) {
    return (
      <div className="app-loading">
        {error ? (
          <p className="error-msg">{error}</p>
        ) : (
          <p>Connecting to backend...</p>
        )}
      </div>
    );
  }

  return (
    <div className="app">
      {/* ── Top nav ── */}
      <nav className="app-nav">
        <h1>HCI Paper Semantic Search</h1>
        <div className="nav-right">
          <button
            className="nav-clear"
            onClick={handleClear}
            title="Clear search inputs, results, and starred collection (filters and Manage Data are kept)"
          >
            Clear
          </button>
          <div className="nav-tabs">
            <button
              className={`nav-tab ${page === "unified" ? "active" : ""}`}
              onClick={() => setPage("unified")}
            >
              Similar Papers
            </button>
            <button
              className={`nav-tab ${page === "search" ? "active" : ""}`}
              onClick={() => setPage("search")}
            >
              Related Work
            </button>
            <button
              className={`nav-tab ${page === "collection" ? "active" : ""}`}
              onClick={() => setPage("collection")}
            >
              My Collection{collection.length > 0 ? ` (${collection.length})` : ""}
            </button>
            <button
              className={`nav-tab ${page === "manage" ? "active" : ""}`}
              onClick={() => setPage("manage")}
            >
              Manage Data
            </button>
          </div>
        </div>
      </nav>

      {(page === "unified" || page === "search") && filterOpts.venues.length === 0 && (
        <div className="no-data-message">
          <p>
            No paper data available yet.{" "}
            <a href="#" onClick={(e) => { e.preventDefault(); setPage("manage"); }}>
              Go to Manage Data
            </a>{" "}
            to collect conference papers.
          </p>
        </div>
      )}

      {page === "unified" && filterOpts.venues.length > 0 && (
        <>
          <header className="app-header">
            <UnifiedQueryBar
              query={unifiedQuery}
              setQuery={setUnifiedQuery}
              onSearch={handleUnifiedSearch}
              searching={unifiedSearching}
              beforeFields={
                <VenueFilter
                  venues={filterOpts.venues}
                  selectedVenues={selectedVenues}
                  toggleVenue={toggleVenue}
                />
              }
            >
              <AdvancedOptions
                retrieveK={retrieveK}
                setRetrieveK={setRetrieveK}
                rerankK={rerankK}
                setRerankK={setRerankK}
              />
            </UnifiedQueryBar>
          </header>

          <main className="app-results">
            {error && <div className="error-banner">{error}</div>}
            {unifiedSearching && (
              <div className="search-loading">
                <div className="spinner" />
                <p>Searching — this may take 10–30 seconds...</p>
              </div>
            )}
            {!unifiedSearching && unifiedResult && (
              <ResultTabs
                data={unifiedResult}
                collectedDois={collectedDois}
                onToggleStar={toggleStar}
              />
            )}
            {!unifiedSearching && !unifiedResult && !error && (
              <div className="results-placeholder">
                Type a query and click Search.
              </div>
            )}
          </main>
        </>
      )}

      {page === "search" && filterOpts.venues.length > 0 && (
        <>
          <header className="app-header">
            <QueryBar
              queries={queries}
              setQuery={setQuery}
              onSearch={handleSearch}
              searching={searching}
              beforeFields={
                <VenueFilter
                  venues={filterOpts.venues}
                  selectedVenues={selectedVenues}
                  toggleVenue={toggleVenue}
                />
              }
            >
              <AdvancedOptions
                retrieveK={retrieveK}
                setRetrieveK={setRetrieveK}
                rerankK={rerankK}
                setRerankK={setRerankK}
              />
            </QueryBar>
          </header>

          <main className="app-results">
            {error && <div className="error-banner">{error}</div>}
            {searching && (
              <div className="search-loading">
                <div className="spinner" />
                <p>Searching — this may take 10–30 seconds...</p>
              </div>
            )}
            {!searching && searchResult && (
              <ResultTabs
                data={searchResult}
                collectedDois={collectedDois}
                onToggleStar={toggleStar}
              />
            )}
            {!searching && !searchResult && !error && (
              <div className="results-placeholder">
                Fill in at least one component box and click Search.
              </div>
            )}
          </main>
        </>
      )}

      {page === "collection" && (
        <Collection
          collection={collection}
          onRemove={removeFromCollection}
          onUpdateNote={updateNote}
        />
      )}

      <div style={{ display: page === "manage" ? "block" : "none" }}>
        <ManageData onCollectComplete={refreshFilterOptions} />
      </div>
    </div>
  );
}
