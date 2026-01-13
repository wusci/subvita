"use client";

import { useMemo, useState } from "react";
import { apiGet, apiPost } from "@/lib/api";

type PredictRequestT2D = {
  request_id?: string | null;

  age_years: number;
  sex_at_birth: "male" | "female" | "unknown";

  height_cm?: number | null;
  weight_kg?: number | null;
  bmi?: number | null;
  waist_circumference_cm: number;

  systolic_bp_mmHg: number;
  diastolic_bp_mmHg: number;

  fasting_glucose_mg_dL: number;
  triglycerides_mg_dL: number;
  hdl_mg_dL: number;
  total_cholesterol_mg_dL?: number | null;
  hba1c_percent?: number | null;

  alt_U_L?: number | null;
  creatinine_mg_dL?: number | null;

  race_ethnicity?:
    | "mexican_american"
    | "other_hispanic"
    | "non_hispanic_white"
    | "non_hispanic_black"
    | "non_hispanic_asian"
    | "other_or_multiracial"
    | "unknown";
  pregnancy_status?: "pregnant" | "not_pregnant" | "unknown";
};

type PredictResponse = {
  request_id?: string | null;
  disease: string;
  predicted_label: string;
  probabilities: Record<string, number>;
  suggested_next_steps: string[];
  notes: string[];
};

type PredictResponseStored = PredictResponse & {
  run_id: string;
};

type RunSummary = {
  run_id: string;
  created_at: string;
  disease: string;
  predicted_label: string;
  probabilities: Record<string, number>;
  model_version: string;
  user_id?: string | null;
};

function numOrNull(v: string): number | null {
  const t = v.trim();
  if (!t) return null;
  const n = Number(t);
  return Number.isFinite(n) ? n : null;
}

export default function Home() {
  const [userId, setUserId] = useState<string>(() => {
    if (typeof window === "undefined") return "demo-user";
    return localStorage.getItem("user_id") ?? "demo-user";
  });

  const [form, setForm] = useState(() => ({
    age_years: "45",
    sex_at_birth: "male",
    height_cm: "175",
    weight_kg: "85",
    bmi: "",
    waist_circumference_cm: "95",
    systolic_bp_mmHg: "128",
    diastolic_bp_mmHg: "82",
    fasting_glucose_mg_dL: "110",
    triglycerides_mg_dL: "160",
    hdl_mg_dL: "45",
    total_cholesterol_mg_dL: "190",
    hba1c_percent: "5.8",
    alt_U_L: "",
    creatinine_mg_dL: "",
    race_ethnicity: "non_hispanic_white",
    pregnancy_status: "unknown",
  }));

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | PredictResponseStored | null>(null);

  const [history, setHistory] = useState<RunSummary[] | null>(null);
  const [historyLoading, setHistoryLoading] = useState(false);

  const requestPayload: PredictRequestT2D | null = useMemo(() => {
    const age = numOrNull(form.age_years);
    const waist = numOrNull(form.waist_circumference_cm);
    const sbp = numOrNull(form.systolic_bp_mmHg);
    const dbp = numOrNull(form.diastolic_bp_mmHg);
    const glu = numOrNull(form.fasting_glucose_mg_dL);
    const tg = numOrNull(form.triglycerides_mg_dL);
    const hdl = numOrNull(form.hdl_mg_dL);

    if (
      age === null ||
      waist === null ||
      sbp === null ||
      dbp === null ||
      glu === null ||
      tg === null ||
      hdl === null
    ) {
      return null;
    }

    return {
      request_id: `web-${Date.now()}`,
      age_years: age,
      sex_at_birth: form.sex_at_birth as PredictRequestT2D["sex_at_birth"],
      height_cm: numOrNull(form.height_cm),
      weight_kg: numOrNull(form.weight_kg),
      bmi: numOrNull(form.bmi),
      waist_circumference_cm: waist,
      systolic_bp_mmHg: sbp,
      diastolic_bp_mmHg: dbp,
      fasting_glucose_mg_dL: glu,
      triglycerides_mg_dL: tg,
      hdl_mg_dL: hdl,
      total_cholesterol_mg_dL: numOrNull(form.total_cholesterol_mg_dL),
      hba1c_percent: numOrNull(form.hba1c_percent),
      alt_U_L: numOrNull(form.alt_U_L),
      creatinine_mg_dL: numOrNull(form.creatinine_mg_dL),
      race_ethnicity: form.race_ethnicity as PredictRequestT2D["race_ethnicity"],
      pregnancy_status: form.pregnancy_status as PredictRequestT2D["pregnancy_status"],
    };
  }, [form]);

  function setField(k: string, v: string) {
    setForm((prev) => ({ ...prev, [k]: v }));
  }

  function persistUserId(v: string) {
    setUserId(v);
    if (typeof window !== "undefined") localStorage.setItem("user_id", v);
  }

  async function doPredict(store: boolean) {
    setError(null);
    setResult(null);

    if (!requestPayload) {
      setError("Please fill in all required numeric fields (age, waist, BP, glucose, TG, HDL).");
      return;
    }

    setLoading(true);
    try {
      if (store) {
        const data = await apiPost<PredictResponseStored>(
          "/v1/predict-and-store/t2d",
          requestPayload,
          { "X-User-ID": userId }
        );
        setResult(data);
      } else {
        const data = await apiPost<PredictResponse>("/v1/predict/t2d", requestPayload);
        setResult(data);
      }
    } catch (e: any) {
      setError(e?.message ?? "Request failed");
    } finally {
      setLoading(false);
    }
  }

  async function loadHistory() {
    setHistory(null);
    setHistoryLoading(true);
    setError(null);
    try {
      const runs = await apiGet<RunSummary[]>(`/v1/users/${encodeURIComponent(userId)}/runs?limit=25&offset=0`);
      setHistory(runs);
    } catch (e: any) {
      setError(e?.message ?? "Failed to load history");
    } finally {
      setHistoryLoading(false);
    }
  }

  return (
    <main className="min-h-screen p-6">
      <div className="mx-auto max-w-3xl space-y-6">
	<header className="space-y-2">
  	  <div className="flex items-baseline justify-between gap-4">
    	    <h1 className="text-2xl font-semibold">Early Disease Detection</h1>
    	    <a className="text-sm underline" href="/history">View history</a>
  	  </div>

  	  <p className="text-sm text-gray-600">
    	    Created by Jachin Thilak & Jiang Wu
  	  </p>
	</header>
        
        <section className="rounded-lg border p-4 space-y-4">
          <div className="flex flex-col gap-2">
            <label className="text-sm font-medium">User ID (for saving & history)</label>
            <input
              className="rounded border p-2"
              value={userId}
              onChange={(e) => persistUserId(e.target.value)}
              placeholder="e.g. alice"
            />
            
          </div>
        </section>

        <section className="rounded-lg border p-4 space-y-4">
          <h2 className="text-lg font-semibold">Inputs</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <Field label="Age (years)*" value={form.age_years} onChange={(v) => setField("age_years", v)} />
            <Select
              label="Sex at birth*"
              value={form.sex_at_birth}
              onChange={(v) => setField("sex_at_birth", v)}
              options={[
                ["male", "male"],
                ["female", "female"],
                ["unknown", "unknown"],
              ]}
            />

            <Field label="Waist Circumference (cm)*" value={form.waist_circumference_cm} onChange={(v) => setField("waist_circumference_cm", v)} />
            <Field label="Systolic Blood Pressure (mmHg)*" value={form.systolic_bp_mmHg} onChange={(v) => setField("systolic_bp_mmHg", v)} />
            <Field label="Diastolic Blood Pressure (mmHg)*" value={form.diastolic_bp_mmHg} onChange={(v) => setField("diastolic_bp_mmHg", v)} />

            <Field label="Fasting Glucose (mg/dL)*" value={form.fasting_glucose_mg_dL} onChange={(v) => setField("fasting_glucose_mg_dL", v)} />
            <Field label="Triglycerides (mg/dL)*" value={form.triglycerides_mg_dL} onChange={(v) => setField("triglycerides_mg_dL", v)} />
            <Field label="HDL (mg/dL)*" value={form.hdl_mg_dL} onChange={(v) => setField("hdl_mg_dL", v)} />

            <Field label="Total Cholesterol (mg/dL)" value={form.total_cholesterol_mg_dL} onChange={(v) => setField("total_cholesterol_mg_dL", v)} />
            <Field label="Hemoglobin A1c (%)" value={form.hba1c_percent} onChange={(v) => setField("hba1c_percent", v)} />

            <Field label="Height (cm)" value={form.height_cm} onChange={(v) => setField("height_cm", v)} />
            <Field label="Weight (kg)" value={form.weight_kg} onChange={(v) => setField("weight_kg", v)} />
            <Field label="Body Mass Index" value={form.bmi} onChange={(v) => setField("bmi", v)} />

            <Select
              label="Race/Ethnicity"
              value={form.race_ethnicity}
              onChange={(v) => setField("race_ethnicity", v)}
              options={[
                ["unknown", "unknown"],
                ["mexican_american", "mexican_american"],
                ["other_hispanic", "other_hispanic"],
                ["non_hispanic_white", "non_hispanic_white"],
                ["non_hispanic_black", "non_hispanic_black"],
                ["non_hispanic_asian", "non_hispanic_asian"],
                ["other_or_multiracial", "other_or_multiracial"],
              ]}
            />

            <Select
              label="Pregnancy Status"
              value={form.pregnancy_status}
              onChange={(v) => setField("pregnancy_status", v)}
              options={[
                ["unknown", "unknown"],
                ["not_pregnant", "not_pregnant"],
                ["pregnant", "pregnant"],
              ]}
            />

            <Field label="Alanine Transaminase (Units per Liter)" value={form.alt_U_L} onChange={(v) => setField("alt_U_L", v)} />
            <Field label="Creatinine (mg/dL)" value={form.creatinine_mg_dL} onChange={(v) => setField("creatinine_mg_dL", v)} />
          </div>

          <div className="flex flex-wrap gap-2">
            <button
              className="rounded bg-black text-white px-4 py-2 disabled:opacity-50"
              disabled={loading}
              onClick={() => doPredict(false)}
            >
              {loading ? "Running..." : "Predict (without saving)"}
            </button>

            <button
              className="rounded border px-4 py-2 disabled:opacity-50"
              disabled={loading}
              onClick={() => doPredict(true)}
            >
              {loading ? "Running..." : "Predict & Save"}
            </button>

            <button
              className="rounded border px-4 py-2 disabled:opacity-50"
              disabled={historyLoading}
              onClick={() => loadHistory()}
            >
              {historyLoading ? "Loading..." : "Load My History"}
            </button>
          </div>

          <p className="text-xs text-gray-500">* required fields</p>
        </section>

        {error && (
          <section className="rounded-lg border border-red-300 bg-red-50 p-4 text-sm text-red-800">
            {error}
          </section>
        )}

        {result && (
          <section className="rounded-lg border p-4 space-y-3">
            <h2 className="text-lg font-semibold">Your Results</h2>
            {"run_id" in result ? (
              <p className="text-sm text-gray-600">Saved run_id: <span className="font-mono">{result.run_id}</span></p>
            ) : null}

            <div className="text-sm">
              <div><span className="font-medium">Predicted:</span> {result.predicted_label}</div>
            </div>

            <div className="text-sm">
              <div className="font-medium mb-1">Probabilities</div>
              <ul className="list-disc ml-6">
                {Object.entries(result.probabilities).map(([k, v]) => (
                  <li key={k}>{k}: {(v * 100).toFixed(1)}%</li>
                ))}
              </ul>
            </div>

            <div className="text-sm">
              <div className="font-medium mb-1">Suggested next steps</div>
              <ul className="list-disc ml-6">
                {result.suggested_next_steps.map((s, i) => <li key={i}>{s}</li>)}
              </ul>
            </div>

            <div className="text-sm">
              <div className="font-medium mb-1">Notes</div>
              <ul className="list-disc ml-6">
                {result.notes.map((n, i) => <li key={i}>{n}</li>)}
              </ul>
            </div>
          </section>
        )}

        {history && (
          <section className="rounded-lg border p-4 space-y-3">
            <h2 className="text-lg font-semibold">My History (latest 25)</h2>
            {history.length === 0 ? (
              <p className="text-sm text-gray-600">No runs found for this user.</p>
            ) : (
              <div className="space-y-2">
                {history.map((r) => (
                  <div key={r.run_id} className="rounded border p-3 text-sm">
                    <div className="flex flex-wrap gap-x-3 gap-y-1">
                      <span className="font-mono">{r.run_id}</span>
                      <span className="text-gray-600">{r.created_at}</span>
                      <span className="font-medium">{r.predicted_label}</span>
                      <span className="text-gray-600">{r.model_version}</span>
                    </div>
                    <div className="text-gray-700 mt-2">
                      p_diabetes: {(Number(r.probabilities?.p_diabetes ?? 0) * 100).toFixed(1)}%
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>
        )}
      </div>
    </main>
  );
}

function Field({ label, value, onChange }: { label: string; value: string; onChange: (v: string) => void }) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-sm font-medium">{label}</label>
      <input className="rounded border p-2" value={value} onChange={(e) => onChange(e.target.value)} />
    </div>
  );
}

function Select({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: [string, string][];
}) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-sm font-medium">{label}</label>
      <select className="rounded border p-2" value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map(([v, text]) => (
          <option key={v} value={v}>
            {text}
          </option>
        ))}
      </select>
    </div>
  );
}
