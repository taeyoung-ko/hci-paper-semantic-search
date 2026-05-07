import { useState, useEffect, useRef } from "react";
import DEFAULT_ENTRIES from "../defaults.json";

function buildDefaults() {
  return DEFAULT_ENTRIES.map((e) => ({
    venue: e.venue,
    year: e.year,
    stem_doi: e.stem_doi || "",
    pacm_issn: e.pacm_issn || "",
    pacm_track: e.pacm_track || "",
    selected: false,
  }));
}

function loadEntries() {
  const defaults = buildDefaults();
  const saved = localStorage.getItem("hcips_manage_entries");
  if (!saved) return defaults;
  let parsed;
  try { parsed = JSON.parse(saved); } catch { return defaults; }
  if (!Array.isArray(parsed) || parsed.length === 0) return defaults;
  const lookup = {};
  for (const d of DEFAULT_ENTRIES) {
    lookup[`${d.venue}|${d.year}`] = d;
  }
  for (const entry of parsed) {
    const key = `${entry.venue}|${entry.year}`;
    const def = lookup[key];
    if (def) {
      if (!entry.stem_doi && def.stem_doi) entry.stem_doi = def.stem_doi;
      if (!entry.pacm_issn && def.pacm_issn) entry.pacm_issn = def.pacm_issn;
      if (!entry.pacm_track && def.pacm_track) entry.pacm_track = def.pacm_track;
    }
    // Ensure fields exist
    if (!entry.pacm_issn) entry.pacm_issn = "";
    if (!entry.pacm_track) entry.pacm_track = "";
  }
  return parsed;
}

function ProgressBar({ progress, label, detail, indeterminate }) {
  return (
    <div className="progress-row">
      <div className="progress-labels">
        <span className="progress-label">{label}</span>
        {detail && <span className="progress-detail">{detail}</span>}
      </div>
      <div className="progress-track">
        <div
          className={`progress-fill ${indeterminate ? "indeterminate" : ""}`}
          style={indeterminate ? {} : { width: `${(progress || 0) * 100}%` }}
        />
      </div>
    </div>
  );
}

export default function ManageData({ onCollectComplete }) {
  const [email, setEmail] = useState("");
  const [entries, setEntries] = useState(loadEntries);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState("");
  const [skipExisting, setSkipExisting] = useState(
    () => localStorage.getItem("hcips_skip_existing") === "1"
  );
  const pollRef = useRef(null);
  const tempIdRef = useRef(-1);

  useEffect(() => {
    localStorage.setItem("hcips_skip_existing", skipExisting ? "1" : "0");
  }, [skipExisting]);

  useEffect(() => {
    localStorage.setItem("hcips_manage_entries", JSON.stringify(entries));
  }, [entries]);

  useEffect(() => {
    if (!running) return;
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch("/api/add-venue/status");
        const data = await res.json();
        setStatus(data);
        if (data.error) setError(data.error);
        if (data.done) {
          setRunning(false);
          clearInterval(pollRef.current);
          if (onCollectComplete) onCollectComplete();
        }
      } catch {}
    }, 1000);
    return () => clearInterval(pollRef.current);
  }, [running, onCollectComplete]);

  // Derived pivot
  const years = [...new Set(entries.map((e) => e.year))].sort((a, b) => {
    if (a < 0 && b < 0) return a - b;
    if (a < 0) return 1;
    if (b < 0) return -1;
    return b - a;
  });
  const venues = [];
  const seenV = new Set();
  for (const e of entries) {
    if (!seenV.has(e.venue)) { seenV.add(e.venue); venues.push(e.venue); }
  }
  const cellMap = {};
  entries.forEach((e, idx) => { cellMap[`${e.venue}|${e.year}`] = idx; });

  // Get venue-level PACM info (from first entry of that venue)
  const venuePacm = {};
  for (const e of entries) {
    if (!venuePacm[e.venue] && (e.pacm_issn || e.pacm_track)) {
      venuePacm[e.venue] = { issn: e.pacm_issn, track: e.pacm_track };
    }
  }

  // ── Mutations ──
  const updateField = (idx, field, value) => {
    setEntries((prev) => prev.map((e, i) => (i === idx ? { ...e, [field]: value } : e)));
  };
  const toggleSelect = (idx) => {
    setEntries((prev) => prev.map((e, i) => (i === idx ? { ...e, selected: !e.selected } : e)));
  };
  const renameVenue = (oldName, newName) => {
    setEntries((prev) => prev.map((e) => (e.venue === oldName ? { ...e, venue: newName } : e)));
  };
  const renameYear = (oldYear, newYear) => {
    setEntries((prev) => prev.map((e) => (e.year === oldYear ? { ...e, year: newYear } : e)));
  };
  const updateVenuePacm = (venue, field, value) => {
    setEntries((prev) => prev.map((e) => (e.venue === venue ? { ...e, [field]: value } : e)));
  };

  // Row operations
  const toggleVenueRow = (venue) => {
    setEntries((prev) => {
      const ve = prev.filter((e) => e.venue === venue && isCollectable(e, venue));
      const allSel = ve.length > 0 && ve.every((e) => e.selected);
      return prev.map((e) =>
        e.venue === venue && isCollectable(e, venue) ? { ...e, selected: !allSel } : e
      );
    });
  };
  const removeVenueRow = (venue) => {
    setEntries((prev) => prev.filter((e) => e.venue !== venue));
  };
  const addVenueRow = () => {
    const newEntries = years.map((y) => ({
      venue: "", year: y, stem_doi: "", pacm_issn: "", pacm_track: "", selected: false,
    }));
    setEntries((prev) => [...prev, ...newEntries]);
  };

  // Column operations
  const toggleYearCol = (year) => {
    setEntries((prev) => {
      const ye = prev.filter((e) => e.year === year && isCollectable(e, e.venue));
      const allSel = ye.length > 0 && ye.every((e) => e.selected);
      return prev.map((e) =>
        e.year === year && isCollectable(e, e.venue) ? { ...e, selected: !allSel } : e
      );
    });
  };
  const removeYearCol = (year) => {
    setEntries((prev) => prev.filter((e) => e.year !== year));
  };
  const addYearCol = () => {
    const tempYear = tempIdRef.current--;
    const activeVenues = venues.filter((v) => v.trim());
    const newEntries = activeVenues.map((v) => {
      const pacm = venuePacm[v] || {};
      return {
        venue: v, year: tempYear, stem_doi: "",
        pacm_issn: pacm.issn || "", pacm_track: pacm.track || "",
        selected: false,
      };
    });
    setEntries((prev) => [...prev, ...newEntries]);
  };

  // A cell is collectable if it has stem_doi OR its venue has pacm_issn
  function isCollectable(entry, venue) {
    if (entry.stem_doi && entry.stem_doi.trim()) return true;
    const pacm = venuePacm[venue];
    if (pacm && pacm.issn) return true;
    if (entry.pacm_issn && entry.pacm_issn.trim()) return true;
    return false;
  }

  // Bulk
  const selectAllCollectable = () => {
    setEntries((prev) => prev.map((e) => ({
      ...e, selected: isCollectable(e, e.venue) ? true : e.selected
    })));
  };
  const deselectAll = () => {
    setEntries((prev) => prev.map((e) => ({ ...e, selected: false })));
  };
  const resetToDefaults = () => {
    if (window.confirm("Reset all entries to defaults?")) setEntries(buildDefaults());
  };

  const selectedEntries = entries.filter(
    (e) => e.selected && e.venue.trim() && e.year > 0 && isCollectable(e, e.venue)
  );

  const handleCollect = async () => {
    if (!email.trim()) { setError("Email is required."); return; }
    if (selectedEntries.length === 0) { setError("Select at least one collectable cell."); return; }
    setRunning(true); setStatus(null); setError("");
    try {
      const res = await fetch("/api/add-venues", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: email.trim(),
          skip_existing: skipExisting,
          entries: selectedEntries.map((e) => ({
            venue: e.venue.trim(),
            year: parseInt(e.year),
            stem_doi: (e.stem_doi || "").trim(),
            pacm_issn: (e.pacm_issn || "").trim(),
            pacm_track: (e.pacm_track || "").trim(),
          })),
        }),
      });
      const data = await res.json();
      if (!data.ok) { setError(data.error || "Failed."); setRunning(false); }
    } catch (e) { setError(e.message); setRunning(false); }
  };

  const overallProgress = status?.overall_total > 0
    ? status.overall_current / status.overall_total : 0;

  return (
    <div className="manage-data">
      <div className="manage-header">
        <div className="manage-email">
          <label>Email</label>
          <input type="email" value={email}
            onChange={(e) => setEmail(e.target.value)} disabled={running} />
          <p className="email-help">
            Sent to Crossref &amp; OpenAlex as a polite-pool contact (User-Agent
            <code>mailto:</code>) for faster, more reliable API responses.
            Not stored on the server.
          </p>
        </div>
        <div className="manage-actions">
          <button onClick={selectAllCollectable} disabled={running}>Select all</button>
          <button onClick={deselectAll} disabled={running}>Deselect all</button>
          <button onClick={resetToDefaults} disabled={running}>Reset</button>
          <label className="skip-existing-toggle" title="If a venue/year already has any paper in the dataset, skip scanning it.">
            <input type="checkbox" checked={skipExisting}
              onChange={(e) => setSkipExisting(e.target.checked)} disabled={running} />
            Skip existing
          </label>
          <button className="collect-btn" onClick={handleCollect}
            disabled={running || selectedEntries.length === 0}>
            {running ? "Running..." : `Collect${selectedEntries.length > 0 ? ` (${selectedEntries.length})` : ""}`}
          </button>
        </div>
      </div>

      {status && (running || status.done) && (
        <div className="progress-area">
          <ProgressBar
            label={`Overall — ${status.overall_current} / ${status.overall_total}`}
            detail={status.overall_label} progress={overallProgress} indeterminate={false} />
          <ProgressBar
            label={status.step_label || ""} detail={status.step_detail || ""}
            progress={status.step_progress}
            indeterminate={status.step_progress === null && !status.done} />
          {status.done && !error && (
            <div className="progress-done">Complete. Switch to Search tab to use the new data.</div>
          )}
          {error && <div className="progress-error">{error}</div>}
        </div>
      )}
      {error && !status && <div className="progress-error" style={{ marginBottom: "1rem" }}>{error}</div>}

      <div className="pivot-wrap">
        <table className="pivot-table">
          <thead>
            <tr>
              <th className="th-venue">Venue</th>
              <th className="th-pacm">ISSN</th>
              <th className="th-pacm">Track</th>
              {years.map((y) => {
                const ye = entries.filter((e) => e.year === y && isCollectable(e, e.venue));
                const allSel = ye.length > 0 && ye.every((e) => e.selected);
                const someSel = ye.some((e) => e.selected);
                const isTemp = y < 0;
                return (
                  <th key={y} className="th-year">
                    <div className="th-year-inner">
                      <input type="checkbox" checked={allSel}
                        ref={(el) => { if (el) el.indeterminate = someSel && !allSel; }}
                        onChange={() => toggleYearCol(y)} disabled={running} />
                      <input type="number" className="year-input"
                        key={y} defaultValue={isTemp ? "" : y}
                        onBlur={(ev) => {
                          const nv = parseInt(ev.target.value);
                          if (nv && nv > 0 && nv !== y && !years.includes(nv)) renameYear(y, nv);
                          else if (!isTemp) ev.target.value = y;
                        }}
                        onKeyDown={(ev) => { if (ev.key === "Enter") ev.target.blur(); }}
                        disabled={running} />
                      <button className="col-rm" onClick={() => removeYearCol(y)} disabled={running}>×</button>
                    </div>
                  </th>
                );
              })}
              <th className="th-add-col">
                <button className="add-col-btn" onClick={addYearCol} disabled={running}>+ Add year</button>
              </th>
            </tr>
          </thead>
          <tbody>
            {venues.map((venue, vi) => {
              const ve = entries.filter((e) => e.venue === venue && isCollectable(e, venue));
              const allSel = ve.length > 0 && ve.every((e) => e.selected);
              const someSel = ve.some((e) => e.selected);
              const pacm = venuePacm[venue] || {};

              return (
                <tr key={vi}>
                  <td className="td-venue">
                    <div className="venue-inner">
                      <input type="checkbox" checked={allSel}
                        ref={(el) => { if (el) el.indeterminate = someSel && !allSel; }}
                        onChange={() => toggleVenueRow(venue)} disabled={running || ve.length === 0} />
                      <input type="text" className="venue-input" value={venue}
                        onChange={(ev) => renameVenue(venue, ev.target.value)}
                        disabled={running} />
                      <button className="row-rm" onClick={() => removeVenueRow(venue)} disabled={running}>×</button>
                    </div>
                  </td>
                  <td className="td-pacm">
                    <input type="text" className="pacm-input"
                      value={pacm.issn || ""}
                      onChange={(ev) => updateVenuePacm(venue, "pacm_issn", ev.target.value)}
                      disabled={running} />
                  </td>
                  <td className="td-pacm">
                    <input type="text" className="pacm-input"
                      value={pacm.track || ""}
                      onChange={(ev) => updateVenuePacm(venue, "pacm_track", ev.target.value)}
                      disabled={running} />
                  </td>
                  {years.map((y) => {
                    const idx = cellMap[`${venue}|${y}`];
                    const collectable = isCollectable(
                      idx !== undefined ? entries[idx] : { stem_doi: "" }, venue
                    );
                    if (idx === undefined) {
                      return (
                        <td key={y} className="td-cell td-empty">
                          <input type="checkbox" className="doi-check"
                            checked={false}
                            onChange={() => {
                              setEntries((prev) => [
                                ...prev,
                                { venue, year: y, stem_doi: "",
                                  pacm_issn: pacm.issn || "", pacm_track: pacm.track || "",
                                  selected: true },
                              ]);
                            }}
                            disabled={running} />
                          <input type="text" className="doi-input" value=""
                            onChange={(ev) => {
                              setEntries((prev) => [
                                ...prev,
                                { venue, year: y, stem_doi: ev.target.value,
                                  pacm_issn: pacm.issn || "", pacm_track: pacm.track || "",
                                  selected: false },
                              ]);
                            }}
                            disabled={running} />
                        </td>
                      );
                    }
                    const e = entries[idx];
                    const isEmpty = !e.stem_doi.trim();
                    return (
                      <td key={y} className={`td-cell ${e.selected ? "td-selected" : ""} ${isEmpty && !collectable ? "td-empty" : ""}`}>
                        <input type="checkbox" className="doi-check" checked={e.selected}
                          onChange={() => toggleSelect(idx)}
                          disabled={running} />
                        <input type="text" className="doi-input" value={e.stem_doi}
                          onChange={(ev) => updateField(idx, "stem_doi", ev.target.value)}
                          disabled={running} />
                      </td>
                    );
                  })}
                  <td className="td-spacer"></td>
                </tr>
              );
            })}
            <tr className="add-row-tr">
              <td className="td-add-row" colSpan={years.length + 4}>
                <button className="add-col-btn" onClick={addVenueRow} disabled={running}>+ Add venue</button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
