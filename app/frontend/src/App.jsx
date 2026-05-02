import { useState, useEffect, useCallback, useRef } from "react";
import { fetchFilterOptions, searchPapers } from "./api";
import { QueryBar, VenueFilter, AdvancedOptions } from "./components/SearchForm";
import ResultTabs from "./components/ResultTabs";
import Collection from "./components/Collection";

// ── localStorage helpers ──
const STORAGE_KEYS = {
  collection: "hcips_collection",
  queries: "hcips_queries",
  venues: "hcips_venues",
  retrieveK: "hcips_retrieve_k",
  rerankK: "hcips_rerank_k",
  searchResult: "hcips_search_result",
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
  const [drawerOpen, setDrawerOpen] = useState(false);

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

  // Persist to localStorage on change
  useEffect(() => { saveJSON(STORAGE_KEYS.collection, collection); }, [collection]);
  useEffect(() => { saveJSON(STORAGE_KEYS.queries, queries); }, [queries]);
  useEffect(() => { saveJSON(STORAGE_KEYS.retrieveK, retrieveK); }, [retrieveK]);
  useEffect(() => { saveJSON(STORAGE_KEYS.rerankK, rerankK); }, [rerankK]);
  useEffect(() => { saveJSON(STORAGE_KEYS.searchResult, searchResult); }, [searchResult]);
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
      {/* ── Top: Header + Options + Query Bar ── */}
      <header className="app-header">
        <h1>HCI Paper Semantic Search</h1>
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

      {/* ── Results (full width) ── */}
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

      {/* ── Right drawer: Collection ── */}
      <Collection
        collection={collection}
        onRemove={removeFromCollection}
        onUpdateNote={updateNote}
        open={drawerOpen}
        onToggle={() => setDrawerOpen((p) => !p)}
      />
    </div>
  );
}
