import { useState } from "react";

import { ApiError, searchText } from "../api/client";
import type { Filters, SearchResponse, SpaceInfo } from "../types";
import ModelSelect from "./ModelSelect";
import ResultGrid from "./ResultGrid";

interface Props {
  spaces: SpaceInfo[];
  defaultLeft: string;
  defaultRight: string;
  params: { k: number; exact: boolean; hnswEf: number; filters: Filters };
}

interface Side {
  space: string;
  response: SearchResponse | null;
  error: string | null;
}

/**
 * Cùng một query, hai model, hai cột. Hai truy vấn chạy song song để thời gian
 * chờ bằng truy vấn chậm hơn chứ không phải tổng hai bên.
 */
export default function CompareView({ spaces, defaultLeft, defaultRight, params }: Props) {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [left, setLeft] = useState<Side>({ space: defaultLeft, response: null, error: null });
  const [right, setRight] = useState<Side>({
    space: defaultRight,
    response: null,
    error: null,
  });

  async function runBoth(text: string) {
    setLoading(true);
    const call = (space: string) =>
      searchText({
        query: text,
        space,
        k: params.k,
        exact: params.exact,
        hnswEf: params.exact ? null : params.hnswEf,
        filters: params.filters,
      })
        .then((response) => ({ response, error: null }))
        .catch((cause) => ({
          response: null,
          error: cause instanceof ApiError ? cause.message : "Lỗi không xác định",
        }));

    const [leftResult, rightResult] = await Promise.all([
      call(left.space),
      call(right.space),
    ]);
    setLeft({ ...left, ...leftResult });
    setRight({ ...right, ...rightResult });
    setLoading(false);
  }

  return (
    <section className="flex flex-col gap-4 rounded-lg border border-slate-700 p-4">
      <h2 className="text-lg font-medium">So sánh hai model trên cùng một query</h2>

      <form
        className="flex flex-wrap items-end gap-3"
        onSubmit={(event) => {
          event.preventDefault();
          if (query.trim()) runBoth(query.trim());
        }}
      >
        <input
          className="min-w-64 flex-1 rounded-md border border-slate-700 bg-slate-800 px-3 py-2 text-slate-100 placeholder:text-slate-500 focus:border-indigo-500 focus:outline-none"
          placeholder="Nhập một query rồi xem hai model xếp hạng khác nhau thế nào"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
        <ModelSelect
          spaces={spaces}
          value={left.space}
          onChange={(space) => setLeft({ ...left, space })}
          label="Model A"
        />
        <ModelSelect
          spaces={spaces}
          value={right.space}
          onChange={(space) => setRight({ ...right, space })}
          label="Model B"
        />
        <button
          className="rounded-md bg-indigo-500 px-4 py-2 font-medium text-white hover:bg-indigo-400 disabled:opacity-50"
          type="submit"
          disabled={loading || !query.trim()}
        >
          So sánh
        </button>
      </form>

      <div className="grid gap-4 lg:grid-cols-2">
        {[left, right].map((side, index) => (
          <div key={index} className="flex flex-col gap-2">
            <p className="text-sm text-slate-400">
              <span className="text-slate-200">{side.space}</span>
              {side.response
                ? ` · ${side.response.latency_ms.toFixed(1)}ms`
                : side.error
                  ? ` · ${side.error}`
                  : ""}
            </p>
            <ResultGrid
              items={side.response?.results ?? []}
              loading={loading}
              onOpen={() => undefined}
            />
          </div>
        ))}
      </div>
    </section>
  );
}
