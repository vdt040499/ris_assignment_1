import type { SearchResponse } from "../types";

interface Props {
  response: SearchResponse | null;
  error: string | null;
  loading: boolean;
}

export default function StatusBar({ response, error, loading }: Props) {
  if (error) {
    return (
      <p className="rounded-md border border-red-800 bg-red-950/60 px-3 py-2 text-sm text-red-200">
        {error}
      </p>
    );
  }
  if (loading) {
    return <p className="text-sm text-slate-400">Đang tìm…</p>;
  }
  if (!response) {
    return <p className="text-sm text-slate-500">Nhập câu chữ hoặc đưa vào một tấm ảnh.</p>;
  }
  return (
    <p className="text-sm text-slate-400">
      <span className="text-slate-200">{response.space}</span> ·{" "}
      {response.results.length}/{response.total_searched} ảnh ·{" "}
      {response.exact ? "exact" : "HNSW"} · {response.latency_ms.toFixed(1)}ms (encode{" "}
      {response.encode_ms.toFixed(1)} + tìm {response.search_ms.toFixed(1)})
    </p>
  );
}
