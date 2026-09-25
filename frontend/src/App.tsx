import { useCallback, useEffect, useMemo, useState } from "react";

import { fetchExamples, fetchSpaces } from "./api/client";
import AdvancedPanel, { type AdvancedParams } from "./components/AdvancedPanel";
import ExampleChips from "./components/ExampleChips";
import ModelSelect from "./components/ModelSelect";
import ResultGrid from "./components/ResultGrid";
import SearchBar from "./components/SearchBar";
import StatusBar from "./components/StatusBar";
import { useSearch } from "./hooks/useSearch";
import type { ExampleQuery, SpaceInfo } from "./types";

const DEFAULT_PARAMS: AdvancedParams = {
  k: 20,
  exact: true,
  hnswEf: 128,
  filters: { categories: [], supercategories: [] },
};

/**
 * Placeholder initial value for `space` before `fetchSpaces()` resolves.
 * Overwritten almost immediately by the first `ready` space returned by the
 * API (see the `firstReady` logic below) — never relied on for an actual
 * search. Named here instead of inlined so it isn't a bare string literal
 * duplicated at the call site.
 */
const DEFAULT_SPACE = "clip-b32";

export default function App() {
  const [spaces, setSpaces] = useState<SpaceInfo[]>([]);
  const [examples, setExamples] = useState<ExampleQuery[]>([]);
  const [space, setSpace] = useState(DEFAULT_SPACE);
  const [params, setParams] = useState<AdvancedParams>(DEFAULT_PARAMS);
  const [bootError, setBootError] = useState<string | null>(null);
  const { response, loading, error, runText, runImage } = useSearch();

  useEffect(() => {
    Promise.all([fetchSpaces(), fetchExamples()])
      .then(([spaceList, exampleList]) => {
        setSpaces(spaceList);
        setExamples(exampleList);
        const firstReady = spaceList.find((item) => item.ready);
        if (firstReady) setSpace(firstReady.name);
      })
      .catch((cause) => setBootError(String(cause.message ?? cause)));
  }, []);

  /** Category có mặt trong kết quả hiện tại — đủ để lọc mà không cần endpoint riêng. */
  const categories = useMemo(() => {
    const found = new Set<string>();
    response?.results.forEach((item) => item.categories.forEach((c) => found.add(c)));
    return Array.from(found).sort();
  }, [response]);

  const search = useCallback(
    (query: string, overrideSpace?: string) =>
      runText({
        query,
        space: overrideSpace ?? space,
        k: params.k,
        exact: params.exact,
        hnswEf: params.exact ? null : params.hnswEf,
        filters: params.filters,
      }),
    [runText, space, params],
  );

  return (
    <main className="mx-auto flex min-h-screen max-w-6xl flex-col gap-5 bg-slate-900 p-6 text-slate-100">
      <header className="flex flex-col gap-1">
        <h1 className="text-2xl font-semibold">Tìm kiếm ngữ nghĩa trên COCO val2017</h1>
        <p className="text-sm text-slate-400">
          Tìm bằng câu chữ hoặc bằng một tấm ảnh, trên 5.000 ảnh.
        </p>
      </header>

      {bootError && (
        <p className="rounded-md border border-red-800 bg-red-950/60 px-3 py-2 text-sm text-red-200">
          {bootError}
        </p>
      )}

      <div className="flex flex-wrap items-end gap-4">
        <ModelSelect spaces={spaces} value={space} onChange={setSpace} label="Model" />
      </div>

      <SearchBar
        onSearchText={(query) => search(query)}
        onSearchImage={(file) =>
          runImage({
            file,
            space,
            k: params.k,
            exact: params.exact,
            hnswEf: params.exact ? null : params.hnswEf,
            filters: params.filters,
          })
        }
        disabled={loading}
      />

      <ExampleChips
        examples={examples}
        onPick={(example) => {
          setSpace(example.space);
          search(example.query, example.space);
        }}
      />

      <AdvancedPanel params={params} onChange={setParams} categories={categories} />
      <StatusBar response={response} error={error} loading={loading} />

      <ResultGrid
        items={response?.results ?? []}
        loading={loading}
        onOpen={() => undefined}
      />
    </main>
  );
}
