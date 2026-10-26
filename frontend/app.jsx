const { useEffect, useMemo, useState } = React;

const defaults = {
  mileage_km: 145000,
  soc: 68,
  voltage_delta_mv: 18,
  temp_delta_c: 3,
  mean_temp_c: 29,
  pack_voltage_v: 365,
  abs_current_a: 22,
  speed_kmph: 18,
  chemistry_lfp: 0,
  capacity_proxy_kwh: 52,
};

const labels = {
  mileage_km: "Mileage (km)",
  soc: "SOC (%)",
  voltage_delta_mv: "Voltage delta (mV)",
  temp_delta_c: "Temperature delta (C)",
  mean_temp_c: "Mean temperature (C)",
  pack_voltage_v: "Pack voltage (V)",
  abs_current_a: "Current magnitude (A)",
  speed_kmph: "Speed (km/h)",
  capacity_proxy_kwh: "Current max capacity proxy (kWh)",
};

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function metric(label, value, unit) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value ?? "--"}</strong>
      <small>{unit}</small>
    </div>
  );
}

function Graph({ points, mode }) {
  if (!points?.length) return <div className="empty">No trajectory yet</div>;
  const width = 720;
  const height = 300;
  const pad = 38;
  const xs = points.map((p) => (mode === "months" ? p.month : p.mileage_km));
  const ys = points.map((p) => p.max_capacity_kwh);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys) * 0.96;
  const maxY = Math.max(...ys) * 1.04;
  const x = (v) => pad + ((v - minX) / Math.max(1, maxX - minX)) * (width - 2 * pad);
  const y = (v) => height - pad - ((v - minY) / Math.max(1, maxY - minY)) * (height - 2 * pad);
  const d = points.map((p, i) => `${i ? "L" : "M"} ${x(mode === "months" ? p.month : p.mileage_km)} ${y(p.max_capacity_kwh)}`).join(" ");
  return (
    <svg className="chart" viewBox={`0 0 ${width} ${height}`} role="img">
      <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} />
      <line x1={pad} y1={pad} x2={pad} y2={height - pad} />
      <path d={d} />
      {points.map((p, i) => (
        <circle key={i} cx={x(mode === "months" ? p.month : p.mileage_km)} cy={y(p.max_capacity_kwh)} r="3" />
      ))}
      <text x={pad} y={22}>{maxY.toFixed(1)} kWh</text>
      <text x={pad} y={height - 8}>{minY.toFixed(1)} kWh</text>
      <text x={width - 170} y={height - 8}>{mode === "months" ? "months" : "mileage km"}</text>
    </svg>
  );
}

function XaiList({ rows }) {
  return (
    <div className="xai">
      {(rows || []).map((row) => (
        <div className="xai-row" key={row.feature}>
          <span>{row.feature}</span>
          <b className={row.contribution >= 0 ? "positive" : "negative"}>{row.contribution.toFixed(2)}</b>
        </div>
      ))}
    </div>
  );
}

function App() {
  const [metadata, setMetadata] = useState(null);
  const [telemetry, setTelemetry] = useState([]);
  const [form, setForm] = useState(defaults);
  const [prediction, setPrediction] = useState(null);
  const [useOverride, setUseOverride] = useState(false);
  const [override, setOverride] = useState({});
  const [graphMode, setGraphMode] = useState("mileage");
  const [selectedXai, setSelectedXai] = useState("final");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    async function boot() {
      setLoading(true);
      const train = await fetch("/api/train").then((r) => r.json());
      setMetadata(train.metadata);
      const rows = await fetch("/api/telemetry").then((r) => r.json());
      setTelemetry(rows);
      const vehicles = train.metadata?.vehicles || [];
      const equal = Object.fromEntries(vehicles.map((v) => [v, 1 / vehicles.length]));
      setOverride(equal);
      setLoading(false);
    }
    boot();
  }, []);

  async function predict(nextForm = form, nextOverride = override, manual = useOverride) {
    const body = { ...nextForm };
    if (manual) body.weight_override = nextOverride;
    const result = await fetch("/api/predict", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => r.json());
    setPrediction(result);
    if (!manual) setOverride(result.client_mix || {});
  }

  useEffect(() => {
    if (metadata) predict();
  }, [metadata]);

  function update(key, value) {
    const next = { ...form, [key]: Number(value) };
    setForm(next);
  }

  function autoGenerate() {
    if (!telemetry.length) return;
    const groups = telemetry.reduce((acc, row) => {
      (acc[row.vin] ||= []).push(row);
      return acc;
    }, {});
    const vins = Object.keys(groups);
    const vin = vins[Math.floor(Math.random() * vins.length)];
    const rows = groups[vin];
    const base = rows[Math.floor(rows.length * (0.25 + Math.random() * 0.5))];
    const next = {
      mileage_km: Math.round(base.mileage_km + (Math.random() - 0.5) * 6000),
      soc: Math.round(clamp(base.soc + (Math.random() - 0.5) * 10, 10, 100)),
      voltage_delta_mv: Math.round(clamp(base.voltage_delta_mv + (Math.random() - 0.5) * 8, 2, 95)),
      temp_delta_c: Number(clamp(base.temp_delta_c + (Math.random() - 0.5) * 2, 0, 18).toFixed(1)),
      mean_temp_c: Number(clamp(base.mean_temp_c + (Math.random() - 0.5) * 5, 5, 55).toFixed(1)),
      pack_voltage_v: Number(clamp(base.pack_voltage_v + (Math.random() - 0.5) * 12, 280, 570).toFixed(1)),
      abs_current_a: Number(clamp(base.abs_current_a + (Math.random() - 0.5) * 18, 0, 180).toFixed(1)),
      speed_kmph: Number(clamp(base.speed_kmph + (Math.random() - 0.5) * 20, 0, 120).toFixed(1)),
      chemistry_lfp: base.chemistry_lfp,
      capacity_proxy_kwh: Number(base.capacity_proxy_kwh.toFixed(2)),
    };
    setForm(next);
    predict(next, override, useOverride);
  }

  function setOverrideWeight(vin, value) {
    const next = { ...override, [vin]: Number(value) / 100 };
    setOverride(next);
  }

  const vehicles = metadata?.vehicles || [];
  const xaiRows = selectedXai === "final" ? prediction?.xai : prediction?.client_xai?.[selectedXai];

  return (
    <main>
      <header className="topbar">
        <div>
          <h1>Federated EV Battery Life Console</h1>
          <p>Manual EV status input, similarity-weighted federated models, digital twin feedback, and explainable AI.</p>
        </div>
        <div className="actions">
          <button onClick={autoGenerate} disabled={!telemetry.length}>Auto-generate</button>
          <button onClick={() => predict()}>Predict</button>
        </div>
      </header>

      {loading && <section className="banner">Loading model cache and dataset-derived plausible telemetry profiles.</section>}

      <section className="grid main-grid">
        <div className="panel">
          <div className="panel-title">
            <h2>Battery Status Input</h2>
            <span>manual or generated</span>
          </div>
          <div className="form-grid">
            <label className="select-field">
              Chemistry
              <select value={form.chemistry_lfp} onChange={(e) => update("chemistry_lfp", e.target.value)}>
                <option value={0}>NMC-like / 95 cells</option>
                <option value={1}>LFP / 156 cells</option>
              </select>
            </label>
            {Object.keys(labels).map((key) => (
              <label key={key}>
                {labels[key]}
                <input type="number" value={form[key]} onChange={(e) => update(key, e.target.value)} />
              </label>
            ))}
          </div>
        </div>

        <div className="panel result">
          <div className="panel-title">
            <h2>Final Weighted Prediction</h2>
            <span>{prediction?.weight_mode === "manual_override" ? "manual override" : "automatic similarity"}</span>
          </div>
          <div className="hero-number">
            <strong>{prediction?.soh_pct ?? "--"}</strong>
            <span>% SOH</span>
          </div>
          <div className="metrics compact">
            {metric("Max Capacity", prediction?.max_capacity_kwh, "kWh")}
            {metric("Remaining Mileage", prediction?.remaining_km_to_80, "km")}
            {metric("Remaining Time", prediction?.remaining_months_to_80, "months")}
            {metric("Twin Stress", prediction?.digital_twin?.stress_score, "score")}
          </div>
        </div>
      </section>

      <section className="grid lower">
        <div className="panel">
          <div className="panel-title">
            <h2>Federated Model Predictions</h2>
            <span>each client model</span>
          </div>
          <div className="client-cards">
            {vehicles.map((vin) => (
              <div className="client-card" key={vin}>
                <h3>{vin}</h3>
                <strong>{prediction?.client_predictions?.[vin]?.soh_pct ?? "--"}% SOH</strong>
                <span>{prediction?.client_predictions?.[vin]?.max_capacity_kwh ?? "--"} kWh</span>
              </div>
            ))}
          </div>
        </div>

        <div className="panel">
          <div className="panel-title">
            <h2>Model Weights</h2>
            <span>automatic with override</span>
          </div>
          <label className="toggle">
            <input type="checkbox" checked={useOverride} onChange={(e) => setUseOverride(e.target.checked)} />
            Manual override
          </label>
          <div className="bars">
            {vehicles.map((vin) => {
              const value = override[vin] ?? prediction?.client_mix?.[vin] ?? 0;
              return (
                <div className="bar-row" key={vin}>
                  <label>{vin}</label>
                  {useOverride ? (
                    <input type="range" min="0" max="100" value={Math.round(value * 100)} onChange={(e) => setOverrideWeight(vin, e.target.value)} />
                  ) : (
                    <div><span style={{ width: `${value * 100}%` }} /></div>
                  )}
                  <b>{Math.round(value * 100)}%</b>
                </div>
              );
            })}
          </div>
          {useOverride && <button className="full" onClick={() => predict(form, override, true)}>Apply Override</button>}
        </div>
      </section>

      <section className="panel">
        <div className="panel-title">
          <h2>Reasoning for Selection</h2>
          <span>plain language</span>
        </div>
        <div className="reasoning">
          {Object.entries(prediction?.weight_reasoning || {}).map(([vin, text]) => (
            <p key={vin}>{text}</p>
          ))}
        </div>
      </section>

      <section className="panel">
        <div className="panel-title">
          <h2>Predicted Max Capacity Over Time</h2>
          <div className="segmented">
            <button className={graphMode === "mileage" ? "active" : ""} onClick={() => setGraphMode("mileage")}>Mileage</button>
            <button className={graphMode === "months" ? "active" : ""} onClick={() => setGraphMode("months")}>Months</button>
          </div>
        </div>
        <Graph points={prediction?.trajectory} mode={graphMode} />
      </section>

      <section className="panel">
        <div className="panel-title">
          <h2>Explainable AI</h2>
          <select value={selectedXai} onChange={(e) => setSelectedXai(e.target.value)}>
            <option value="final">Final weighted model</option>
            {vehicles.map((vin) => <option key={vin} value={vin}>{vin} model</option>)}
          </select>
        </div>
        <XaiList rows={xaiRows} />
      </section>
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
