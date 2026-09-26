import { useCallback, useEffect, useMemo, useState } from "react";

import { fetchExamples, fetchSpaces } from "./api/client";
import AdvancedPanel, { type AdvancedParams } from "./components/AdvancedPanel";
import CompareView from "./components/CompareView";
import DetailModal from "./components/DetailModal";
import ExampleChips from "./components/ExampleChips";
import ModelSelect from "./components/ModelSelect";
import ResultGrid from "./components/ResultGrid";
import SearchBar from "./components/SearchBar";
import StatusBar from "./components/StatusBar";
import { useSearch } from "./hooks/useSearch";
import type { ExampleQuery, SearchResultItem, SpaceInfo } from "./types";

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

/**
 * Default model pair for compare mode — two models with different encoder
 * architectures (OpenAI CLIP vs LAION OpenCLIP), so ranking results tend to
 * diverge noticeably, which illustrates the comparison well. Named as a
 * constant instead of a scattered literal in JSX, following the same
 * pattern as `DEFAULT_SPACE` above.
 */
const COMPARE_DEFAULT_LEFT = "clip-b32";
const COMPARE_DEFAULT_RIGHT = "laion-b32";

export default function App() {
  const [spaces, setSpaces] = useState<SpaceInfo[]>([]);
  const [examples, setExamples] = useState<ExampleQuery[]>([]);
  const [space, setSpace] = useState(DEFAULT_SPACE);
  const [params, setParams] = useState<AdvancedParams>(DEFAULT_PARAMS);
  const [bootError, setBootError] = useState<string | null>(null);
  const [selected, setSelected] = useState<SearchResultItem | null>(null);
  const [comparing, setComparing] = useState(false);
  const [spaceSwitchNotice, setSpaceSwitchNotice] = useState<string | null>(null);
  const { response, loading, error, runText, runImage, runSimilar } = useSearch();

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

  /** Categories present in the current results — enough to filter without a dedicated endpoint. */
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

  /**
   * If the current space doesn't support `image2image` (e.g. after clicking
   * a Vietnamese chip that sets space = `mclip-b32`, which only supports
   * text2image), a search-by-image request is guaranteed to be rejected by
   * the backend (400 `ModeNotSupportedError`), and `useSearch` clears the
   * whole results grid to show the error — exactly the path the spec calls
   * the line between "a smooth demo" and "a demo that gets stuck." Auto
   * switch to the first `ready` space that supports `image2image` (same
   * pattern as `firstReady` in the effect above) so we never send a request
   * that is certain to fail.
   *
   * @returns the space to use when calling `runSimilar`/`runImage`, along
   *   with a `switched` flag to let the user know the model was changed.
   */
  const resolveImageSpace = useCallback(
    (current: string): { space: string; switched: boolean } => {
      const currentInfo = spaces.find((item) => item.name === current);
      if (currentInfo?.modes.includes("image2image")) {
        return { space: current, switched: false };
      }
      const fallback = spaces.find(
        (item) => item.ready && item.modes.includes("image2image"),
      );
      return fallback
        ? { space: fallback.name, switched: true }
        : { space: current, switched: false };
    },
    [spaces],
  );

  /** Closes the modal, then runs a similar-image search using the `image_id` of the image being viewed. */
  const findSimilar = useCallback(
    (imageId: number) => {
      setSelected(null);
      const target = resolveImageSpace(space);
      if (target.switched) {
        setSpace(target.space);
        setSpaceSwitchNotice(
          `Automatically switched to model "${target.space}" because "${space}" doesn't support similar-image search.`,
        );
      } else {
        setSpaceSwitchNotice(null);
      }
      runSimilar({
        imageId,
        space: target.space,
        k: params.k,
        exact: params.exact,
        hnswEf: params.exact ? null : params.hnswEf,
        filters: params.filters,
      });
    },
    [runSimilar, resolveImageSpace, space, params],
  );

  /** Shared for uploaded images (file picker / drag-drop / paste) — same
   * auto-switch-space mechanism as `findSimilar` above, since both call
   * `/search/image` and can hit the same space that doesn't support
   * `image2image`. */
  const searchByImage = useCallback(
    (file: File) => {
      const target = resolveImageSpace(space);
      if (target.switched) {
        setSpace(target.space);
        setSpaceSwitchNotice(
          `Automatically switched to model "${target.space}" because "${space}" doesn't support image search.`,
        );
      } else {
        setSpaceSwitchNotice(null);
      }
      runImage({
        file,
        space: target.space,
        k: params.k,
        exact: params.exact,
        hnswEf: params.exact ? null : params.hnswEf,
        filters: params.filters,
      });
    },
    [runImage, resolveImageSpace, space, params],
  );

  return (
    <main className="mx-auto flex min-h-screen max-w-6xl flex-col gap-5 bg-slate-900 p-6 text-slate-100">
      <header className="flex flex-col gap-1">
        <h1 className="text-2xl font-semibold">Semantic Search on COCO val2017</h1>
        <p className="text-sm text-slate-400">
          Search by text or by an image, across 5,000 images.
        </p>
      </header>

      {bootError && (
        <p className="rounded-md border border-red-800 bg-red-950/60 px-3 py-2 text-sm text-red-200">
          {bootError}
        </p>
      )}

      <div className="flex flex-wrap items-end gap-4">
        <ModelSelect
          spaces={spaces}
          value={space}
          onChange={(value) => {
            setSpace(value);
            setSpaceSwitchNotice(null);
          }}
          label="Model"
        />
        <button
          className={`rounded-md border px-3 py-2 text-sm ${
            comparing
              ? "border-indigo-500 text-indigo-300"
              : "border-slate-700 text-slate-300 hover:border-indigo-500"
          }`}
          type="button"
          onClick={() => setComparing((value) => !value)}
        >
          {comparing ? "Compare mode active" : "Compare two models"}
        </button>
      </div>

      <SearchBar
        onSearchText={(query) => search(query)}
        onSearchImage={searchByImage}
        disabled={loading}
      />

      <ExampleChips
        examples={examples}
        onPick={(example) => {
          setSpace(example.space);
          setSpaceSwitchNotice(null);
          search(example.query, example.space);
        }}
      />

      <AdvancedPanel params={params} onChange={setParams} categories={categories} />
      {spaceSwitchNotice && (
        <p className="rounded-md border border-indigo-800 bg-indigo-950/60 px-3 py-2 text-sm text-indigo-200">
          {spaceSwitchNotice}
        </p>
      )}
      <StatusBar response={response} error={error} loading={loading} />

      <ResultGrid
        items={response?.results ?? []}
        loading={loading}
        onOpen={setSelected}
      />

      {comparing && (
        <CompareView
          spaces={spaces}
          defaultLeft={COMPARE_DEFAULT_LEFT}
          defaultRight={COMPARE_DEFAULT_RIGHT}
          params={params}
        />
      )}

      <DetailModal item={selected} onClose={() => setSelected(null)} onFindSimilar={findSimilar} />
    </main>
  );
}
