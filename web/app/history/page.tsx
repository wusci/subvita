"use client";

import { useEffect, useState } from "react";
import { apiGet } from "@/lib/api";

type RunSummary = {
  run_id: string;
  created_at: string;
  disease: string;
  predicted_label: string;
  probabilities: Record<string, number>;
  model_version: string;
  user_id?: string | null;
};

export default function HistoryPage() {
  const [userId, setUserId] = useState<string>("demo-user");
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const saved = localStorage.getItem("user_id") ?? "demo-user";
    setUserId(saved);
  }, []);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const data = await apiGet<RunSummary[]>(
        `/v1/users/${encodeURIComponent(userId)}/runs?limit=50&offset=0`
      );
      setRuns(data);
    } catch (e: any) {
      setError(e?.message ?? "Failed to load history");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (userId) load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userId]);

  return (
    <main className="min-h-screen p-6">
      <div className="mx-auto max-w-3xl space-y-4">
        <header className="flex items-center justify-between">
          <h1 className="text-2xl font-semibold">My History</h1>
          <a className="text-sm underline" href="/">Back</a>
        </header>

        <div className="rounded border p-4 space-y-2">
          <div className="text-sm text-gray-600">User ID</div>
          <div className="font-mono">{userId}</div>
          <button
            className="rounded border px-3 py-2 text-sm disabled:opacity-50"
            disabled={loading}
            onClick={load}
          >
            {loading ? "Loading..." : "Refresh"}
          </button>
        </div>

        {error && (
          <div className="rounded border border-red-300 bg-red-50 p-4 text-sm text-red-800">
            {error}
          </div>
        )}

        {runs && runs.length === 0 && (
          <div className="rounded border p-4 text-sm text-gray-600">
            No runs found.
          </div>
        )}

        {runs && runs.length > 0 && (
          <div className="space-y-2">
            {runs.map((r) => (
              <div key={r.run_id} className="rounded border p-3 text-sm">
                <div className="flex flex-wrap gap-x-3 gap-y-1">
                  <span className="font-mono">{r.run_id}</span>
                  <span className="text-gray-600">{r.created_at}</span>
                  <span className="font-medium">{r.predicted_label}</span>
                  <span className="text-gray-600">{r.model_version}</span>
                </div>
                <div className="mt-2 text-gray-700">
                  p_diabetes: {(Number(r.probabilities?.p_diabetes ?? 0) * 100).toFixed(1)}%
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </main>
  );
}
