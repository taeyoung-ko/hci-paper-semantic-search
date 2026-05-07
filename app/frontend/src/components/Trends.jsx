import { useState, useMemo, useRef, useEffect } from "react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend,
  Customized,
} from "recharts";
import { fetchTrends, fetchTopicPapers } from "../api";
import { VenueFilter } from "./SearchForm";
import ResultCard from "./ResultCard";

const COLORS = [
  "#4b6e48", "#7a5230", "#3a5a7a", "#7d3a3a", "#5a4778",
  "#3a6e6a", "#7d5a30", "#2d4d2a", "#a07e2e", "#3a527a",
  "#9a3a3a", "#5a8a5a", "#b08a3a", "#5a3a7a", "#3a7a8a",
  "#a04a5a", "#5a9a5a", "#a06e2e", "#7a4a8a", "#4a8a9a",
];

const colorFor = (i) => COLORS[i % COLORS.length];

export default function Trends({
  venues,
  selectedVenues,
  toggleVenue,
  yearMinAvailable,
  yearMaxAvailable,
  collectedDois,
  onToggleStar,
}) {
  const [yearMin, setYearMin] = useState(yearMinAvailable);
  const [yearMax, setYearMax] = useState(yearMaxAvailable);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [normalize, setNormalize] = useState(false);
  const [selectedTopicId, setSelectedTopicId] = useState(null);
  const [topicPapers, setTopicPapers] = useState([]);
  const [topicPapersLoading, setTopicPapersLoading] = useState(false);
  const [hoveredYear, setHoveredYear] = useState(null);
  const chartRef = useRef(null);
  const [chartHeight, setChartHeight] = useState(0);

  useEffect(() => {
    if (!chartRef.current) return;
    const el = chartRef.current;
    const update = () => {
      const w = el.getBoundingClientRect().width;
      if (w > 0) setChartHeight(Math.round(w * 0.75)); // 4:3
    };
    update();
    const obs = new ResizeObserver(update);
    obs.observe(el);
    return () => obs.disconnect();
  }, [data]);

  const handleRun = async (force = false) => {
    setLoading(true);
    setError("");
    try {
      const result = await fetchTrends({
        venues: [...selectedVenues],
        year_min: yearMin,
        year_max: yearMax,
        force,
      });
      if (!result.ok) {
        setError(result.error || "Failed.");
        setData(null);
      } else {
        setData(result);
        setSelectedTopicId(null);
        setTopicPapers([]);
      }
    } catch (e) {
      setError(e.message);
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  const chartData = useMemo(() => {
    if (!data?.over_time) return [];
    const yearMap = {};
    for (const row of data.over_time) {
      const y = row.year;
      if (!yearMap[y]) yearMap[y] = { year: y };
      yearMap[y][`t${row.topic_id}`] = row.frequency;
    }
    const totalsByYear = Object.fromEntries(
      (data.year_totals || []).map((y) => [y.year, y.total])
    );
    const rows = Object.values(yearMap).sort((a, b) => a.year - b.year);
    if (!normalize) return rows;
    return rows.map((row) => {
      const total = totalsByYear[row.year] || 1;
      const out = { year: row.year };
      for (const k of Object.keys(row)) {
        if (k === "year") continue;
        out[k] = (row[k] / total) * 100;
      }
      return out;
    });
  }, [data, normalize]);

  const visibleTopics = useMemo(() => data?.topics || [], [data]);

  const handleSelectTopic = async (id) => {
    if (selectedTopicId === id) {
      setSelectedTopicId(null);
      setTopicPapers([]);
      return;
    }
    setSelectedTopicId(id);
    setTopicPapersLoading(true);
    try {
      const res = await fetchTopicPapers({
        venues: [...selectedVenues],
        year_min: yearMin,
        year_max: yearMax,
        topic_id: id,
      });
      setTopicPapers(res.papers || []);
    } catch (e) {
      setTopicPapers([]);
    } finally {
      setTopicPapersLoading(false);
    }
  };

  return (
    <div className="trends-page">
      <div className="trends-filter-bar">
        <VenueFilter
          venues={venues}
          selectedVenues={selectedVenues}
          toggleVenue={toggleVenue}
        />
        <div className="trends-year-range">
          <label>Years</label>
          <div className="trends-year-inputs">
            <input
              type="number"
              value={yearMin}
              min={yearMinAvailable}
              max={yearMax}
              onChange={(e) => setYearMin(parseInt(e.target.value) || yearMinAvailable)}
            />
            <span className="trends-year-sep">–</span>
            <input
              type="number"
              value={yearMax}
              min={yearMin}
              max={yearMaxAvailable}
              onChange={(e) => setYearMax(parseInt(e.target.value) || yearMaxAvailable)}
            />
            <span className="trends-year-hint">
              (data: {yearMinAvailable}–{yearMaxAvailable})
            </span>
          </div>
        </div>
      </div>

      <div className="trends-header">
        <div className="trends-controls">
          <button
            className="trends-run-btn"
            onClick={() => handleRun(false)}
            disabled={loading || selectedVenues.size === 0}
          >
            {loading ? "Computing topics..." : "Compute trends"}
          </button>
          {data?.ok && (
            <button
              className="trends-recompute-btn"
              onClick={() => handleRun(true)}
              disabled={loading || selectedVenues.size === 0}
              title="Force recomputation (ignore cache)"
            >
              Recompute
            </button>
          )}
          <span className="trends-summary">
            {selectedVenues.size} venue
            {selectedVenues.size !== 1 ? "s" : ""} · {yearMin}–{yearMax}
          </span>
          {data?.ok && (
            <label className="trends-norm-toggle">
              <input
                type="checkbox"
                checked={normalize}
                onChange={(e) => setNormalize(e.target.checked)}
              />
              Normalize per year (%)
            </label>
          )}
        </div>
        {data?.ok && (
          <div className="trends-meta">
            {data.n_papers} papers · {data.topics.length} topics
            {data.n_outliers > 0 && ` · ${data.n_outliers} outliers (hidden)`}
          </div>
        )}
      </div>

      {error && <div className="error-banner">{error}</div>}

      {!data && !loading && !error && (
        <div className="results-placeholder">
          Pick venues / year range above and click <b>Compute trends</b>.
          BERTopic clusters papers by their embedding and shows topic
          frequency over time. First run for a given filter takes 30–90 s.
        </div>
      )}

      {loading && (
        <div className="search-loading">
          <div className="spinner" />
          <p>Running BERTopic — this can take 30–90 s on first run...</p>
        </div>
      )}

      {data?.ok && (
        <div className="trends-body">
          <div
            className="trends-chart"
            ref={chartRef}
            style={chartHeight ? { height: chartHeight } : undefined}
          >
            <div className="trends-chart-inner">
              <ResponsiveContainer width="100%" height="100%">
              <AreaChart
                data={chartData}
                onMouseMove={(s) => setHoveredYear(s?.activeLabel ?? null)}
                onMouseLeave={() => setHoveredYear(null)}
              >
                <CartesianGrid strokeDasharray="2 4" stroke="#d8d4c4" />
                <XAxis dataKey="year" stroke="#898989" />
                <YAxis
                  stroke="#898989"
                  tickFormatter={(v) => normalize ? `${v.toFixed(0)}%` : v}
                />
                <Legend wrapperStyle={{ display: "none" }} />
                {visibleTopics.map((topic, i) => (
                  <Area
                    key={topic.id}
                    type="monotone"
                    dataKey={`t${topic.id}`}
                    stackId="1"
                    stroke={colorFor(i)}
                    fill={colorFor(i)}
                    fillOpacity={0.78}
                    isAnimationActive={false}
                    activeDot={false}
                    dot={false}
                  />
                ))}
                <Customized
                  component={(chart) => {
                    if (hoveredYear == null) return null;
                    const items = chart?.formattedGraphicalItems;
                    if (!items || items.length === 0) return null;
                    const idx = chartData.findIndex((d) => d.year === hoveredYear);
                    if (idx < 0) return null;

                    const labels = [];
                    items.forEach((it, areaIdx) => {
                      const points = it?.props?.points;
                      const baseLine = it?.props?.baseLine;
                      if (!points || !points[idx]) return;
                      const topY = points[idx].y;
                      const cx = points[idx].x;
                      const topic = visibleTopics[areaIdx];
                      if (!topic) return;
                      const rawValue = chartData[idx]?.[`t${topic.id}`];
                      const value = typeof rawValue === "number" ? rawValue : 0;
                      if (!value) return;
                      const bottomY =
                        Array.isArray(baseLine) && baseLine[idx] && baseLine[idx].y != null
                          ? baseLine[idx].y
                          : (areaIdx > 0 && items[areaIdx - 1]?.props?.points?.[idx]?.y != null
                              ? items[areaIdx - 1].props.points[idx].y
                              : topY);
                      const cy = (topY + bottomY) / 2;

                      let goRight = areaIdx % 2 === 0;
                      if (idx === 0) goRight = true;
                      else if (idx === chartData.length - 1) goRight = false;
                      const dx = goRight ? 9 : -9;
                      const anchor = goRight ? "start" : "end";

                      const words =
                        topic.words.slice(0, 2).join(" · ") || `topic ${topic.id}`;
                      const display = normalize
                        ? `${Number(value).toFixed(1)}%`
                        : Math.round(value);
                      const color = colorFor(areaIdx);

                      labels.push(
                        <g key={topic.id} style={{ pointerEvents: "none" }}>
                          <circle
                            cx={cx}
                            cy={cy}
                            r={3.5}
                            fill={color}
                            stroke="#fff"
                            strokeWidth={1.5}
                          />
                          <text
                            x={cx + dx}
                            y={cy}
                            dy={3.5}
                            fontSize={10.5}
                            fontWeight={600}
                            fill="#2a2a2a"
                            stroke="#fff"
                            strokeWidth={3}
                            paintOrder="stroke"
                            textAnchor={anchor}
                          >
                            {`${words}: ${display}`}
                          </text>
                        </g>
                      );
                    });
                    return <g>{labels}</g>;
                  }}
                />
              </AreaChart>
            </ResponsiveContainer>
            </div>
          </div>

          <div
            className="trends-topics"
            style={chartHeight ? { height: chartHeight } : undefined}
          >
            <h3>Topics <span className="trends-topics-hint">(click to see papers)</span></h3>
            <ul>
              {data.topics.map((topic, i) => (
                <li
                  key={topic.id}
                  className={`trends-topic ${selectedTopicId === topic.id ? "selected" : ""}`}
                  onClick={() => handleSelectTopic(topic.id)}
                >
                  <span
                    className="trends-topic-swatch"
                    style={{ background: colorFor(i) }}
                  />
                  <div className="trends-topic-info">
                    <div className="trends-topic-words">
                      {topic.words.slice(0, 6).join(" · ") || "(no terms)"}
                    </div>
                    <div className="trends-topic-count">
                      {topic.count} paper{topic.count !== 1 ? "s" : ""}
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {data?.ok && selectedTopicId !== null && (
        <div className="trends-topic-papers">
          {(() => {
            const topic = data.topics.find((t) => t.id === selectedTopicId);
            return (
              <h3>
                Papers in:{" "}
                <span className="trends-topic-papers-words">
                  {topic ? topic.words.slice(0, 6).join(" · ") : `Topic ${selectedTopicId}`}
                </span>{" "}
                <span className="trends-topic-papers-count">
                  ({topic ? topic.count : topicPapers.length})
                </span>
              </h3>
            );
          })()}
          {topicPapersLoading ? (
            <div className="search-loading">
              <div className="spinner" />
            </div>
          ) : topicPapers.length === 0 ? (
            <div className="results-placeholder">No papers found.</div>
          ) : (
            <div>
              {topicPapers.map((p, i) => (
                <ResultCard
                  key={p.doi || p.title || i}
                  paper={p}
                  rank={i + 1}
                  isStarred={collectedDois?.has(p.doi || p.title)}
                  onToggleStar={() => onToggleStar?.(p, "trends", "")}
                  hideScore={true}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
