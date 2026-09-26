import { useCallback, useState } from "react";

import { ApiError, searchImage, searchText } from "../api/client";
import type { SearchParams, SearchResponse } from "../types";

interface SearchState {
  response: SearchResponse | null;
  loading: boolean;
  error: string | null;
}

/**
 * A single search and all of its state.
 *
 * Keeps the previous result while loading so the grid doesn't flash blank
 * between two searches; it's only cleared on error. Returns the response so
 * the caller (compare mode) can use it directly.
 *
 * @returns `response` (the latest valid result, or `null`), `loading`,
 *   `error` (a human-readable message, or `null`), plus three functions to
 *   run a search: `runText` (search by text), `runImage` (search by an
 *   uploaded image), `runSimilar` (find images similar to one already in the
 *   corpus via `imageId`). All three return `Promise<SearchResponse | null>`
 *   — `null` on error.
 */
export function useSearch() {
  const [state, setState] = useState<SearchState>({
    response: null,
    loading: false,
    error: null,
  });

  const run = useCallback(async (task: () => Promise<SearchResponse>) => {
    setState((prev) => ({ ...prev, loading: true, error: null }));
    try {
      const response = await task();
      setState({ response, loading: false, error: null });
      return response;
    } catch (error) {
      const message =
        error instanceof ApiError ? error.message : "Unknown error";
      setState({ response: null, loading: false, error: message });
      return null;
    }
  }, []);

  const runText = useCallback(
    (params: SearchParams) => run(() => searchText(params)),
    [run],
  );

  const runImage = useCallback(
    (params: SearchParams & { file: File }) => run(() => searchImage(params)),
    [run],
  );

  const runSimilar = useCallback(
    (params: SearchParams & { imageId: number }) => run(() => searchImage(params)),
    [run],
  );

  return { ...state, runText, runImage, runSimilar };
}
