import { useCallback, useEffect, useState } from "react";

import { fetchSpaces } from "./api/client";
import type { AdvancedParams } from "./components/AdvancedPanel";
import CompareView from "./components/CompareView";
import DetailModal from "./components/DetailModal";
import ModelSelect from "./components/ModelSelect";
import ResultCountSelect from "./components/ResultCountSelect";
import ResultGrid from "./components/ResultGrid";
import SearchBar from "./components/SearchBar";
import StatusBar from "./components/StatusBar";
import { useSearch } from "./hooks/useSearch";
import type { SearchResultItem, SpaceInfo } from "./types";

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
 * Default model pair for compare mode. The two models were trained on
 * different data (OpenAI CLIP vs LAION OpenCLIP), so their rankings usually
 * differ clearly, which makes for a good comparison. Named constants instead
 * of scattered literals in JSX, following the `DEFAULT_SPACE` pattern above.
 */
const COMPARE_DEFAULT_LEFT = "clip-b32";
const COMPARE_DEFAULT_RIGHT = "laion-b32";

export default function App() {
  const [spaces, setSpaces] = useState<SpaceInfo[]>([]);
  const [space, setSpace] = useState(DEFAULT_SPACE);
  const [params, setParams] = useState<AdvancedParams>(DEFAULT_PARAMS);
  const [bootError, setBootError] = useState<string | null>(null);
  const [selected, setSelected] = useState<SearchResultItem | null>(null);
  const [comparing, setComparing] = useState(false);
  const [spaceSwitchNotice, setSpaceSwitchNotice] = useState<string | null>(null);
  const { response, loading, error, runText, runImage, runSimilar } = useSearch();

  useEffect(() => {
    fetchSpaces()
      .then((spaceList) => {
        setSpaces(spaceList);
        const firstReady = spaceList.find((item) => item.ready);
        if (firstReady) setSpace(firstReady.name);
      })
      .catch((cause) => setBootError(String(cause.message ?? cause)));
  }, []);

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
   * If the current space does not support `image2image` (e.g. a Vietnamese
   * example chip was just clicked, setting space = `mclip-b32`, which only
   * supports text2image), an image search would certainly be rejected by the
   * backend (400 `ModeNotSupportedError`), and `useSearch` would clear the whole
   * result grid to show the error, which is the path the spec calls the line
   * between a "smooth demo" and a "stuck demo". Automatically switch to the
   * first `ready` space that supports `image2image` (same pattern as
   * `firstReady` in the effect above) so a request that is bound to fail is
   * never sent.
   *
   * @returns The space to use for `runSimilar`/`runImage`, plus a `switched`
   *   flag so the user can be told the model changed.
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
          `Automatically switched to model "${target.space}" because "${space}" does not support similar-image search.`,
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

  /** Shared by uploaded images (file picker / drag-drop / paste), with the same
   * automatic space switching as `findSimilar` above, since both call
   * `/search/image` and can land on a space that does not support `image2image`. */
  const searchByImage = useCallback(
    (file: File) => {
      const target = resolveImageSpace(space);
      if (target.switched) {
        setSpace(target.space);
        setSpaceSwitchNotice(
          `Automatically switched to model "${target.space}" because "${space}" does not support search by image.`,
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
    <main className="mx-auto flex min-h-screen w-full max-w-screen-2xl flex-col gap-5 px-4 py-6 sm:px-6 lg:px-8">
      <header className="flex flex-col gap-1">
        <h1 className="text-2xl font-semibold">Image Search on COCO</h1>
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
        <ResultCountSelect
          value={params.k}
          onChange={(k) => setParams({ ...params, k })}
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
          {comparing ? "Comparing two models" : "Compare two models"}
        </button>
      </div>

      <SearchBar
        onSearchText={(query) => search(query)}
        onSearchImage={searchByImage}
        disabled={loading}
      />

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
