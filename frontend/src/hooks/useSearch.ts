import { useCallback, useState } from "react";

import { ApiError, searchImage, searchText } from "../api/client";
import type { SearchParams, SearchResponse } from "../types";

interface SearchState {
  response: SearchResponse | null;
  loading: boolean;
  error: string | null;
}

/**
 * Một lần tìm kiếm và toàn bộ trạng thái của nó.
 *
 * Giữ kết quả cũ trong lúc đang tải để grid không nháy trắng giữa hai lần tìm;
 * chỉ xoá khi có lỗi. Trả về response để chỗ gọi (chế độ so sánh) dùng trực tiếp.
 *
 * @returns `response` (kết quả gần nhất còn hợp lệ, hoặc `null`), `loading`,
 *   `error` (message tiếng người, hoặc `null`), cùng ba hàm chạy tìm kiếm:
 *   `runText` (tìm theo văn bản), `runImage` (tìm theo ảnh upload), `runSimilar`
 *   (tìm ảnh tương tự một ảnh đã có trong corpus qua `imageId`). Cả ba trả về
 *   `Promise<SearchResponse | null>` — `null` khi có lỗi.
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
        error instanceof ApiError ? error.message : "Lỗi không xác định";
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
