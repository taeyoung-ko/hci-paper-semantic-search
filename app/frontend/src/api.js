const API = "/api";

export async function fetchFilterOptions() {
  const res = await fetch(`${API}/filter-options`);
  if (!res.ok) throw new Error("Failed to load filter options");
  return res.json();
}

export async function searchPapers(params) {
  const res = await fetch(`${API}/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error("Search failed");
  return res.json();
}

export async function exportBib(collection) {
  const res = await fetch(`${API}/export-bib`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ collection }),
  });
  if (!res.ok) throw new Error("Export failed");
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `hcips_collection_${Date.now()}.bib`;
  a.click();
  URL.revokeObjectURL(url);
}
