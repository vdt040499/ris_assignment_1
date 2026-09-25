import type { Filters } from "../types";

/** Hình dạng state có kiểm soát của panel nâng cao — App sở hữu state này. */
export interface AdvancedParams {
  k: number;
  exact: boolean;
  hnswEf: number;
  filters: Filters;
}

interface Props {
  params: AdvancedParams;
  onChange: (params: AdvancedParams) => void;
  categories: string[];
}

export default function AdvancedPanel({ params, onChange, categories }: Props) {
  return (
    <details className="rounded-lg border border-slate-700 p-3 text-sm">
      <summary className="cursor-pointer text-slate-300">Tuỳ chọn nâng cao</summary>
      <div className="mt-3 flex flex-wrap items-end gap-4">
        <label className="flex flex-col gap-1">
          <span className="text-slate-400">Số kết quả: {params.k}</span>
          <input
            type="range"
            min={5}
            max={50}
            step={5}
            value={params.k}
            onChange={(event) => onChange({ ...params, k: Number(event.target.value) })}
          />
        </label>

        <label className="flex items-center gap-2 text-slate-300">
          <input
            type="checkbox"
            checked={params.exact}
            onChange={(event) => onChange({ ...params, exact: event.target.checked })}
          />
          Tìm chính xác (tắt để dùng HNSW)
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-slate-400">hnsw_ef</span>
          <input
            className="w-24 rounded-md border border-slate-700 bg-slate-800 px-2 py-1 text-slate-100 disabled:opacity-40"
            type="number"
            min={8}
            max={512}
            value={params.hnswEf}
            disabled={params.exact}
            onChange={(event) =>
              onChange({ ...params, hnswEf: Number(event.target.value) })
            }
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-slate-400">Lọc theo category</span>
          <select
            className="h-24 w-48 rounded-md border border-slate-700 bg-slate-800 px-2 py-1 text-slate-100"
            multiple
            value={params.filters.categories}
            onChange={(event) =>
              onChange({
                ...params,
                filters: {
                  ...params.filters,
                  categories: Array.from(event.target.selectedOptions, (o) => o.value),
                },
              })
            }
          >
            {categories.map((category) => (
              <option key={category} value={category}>
                {category}
              </option>
            ))}
          </select>
        </label>
      </div>
    </details>
  );
}
