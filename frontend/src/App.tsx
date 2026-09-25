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
 * Cặp model mặc định cho chế độ so sánh — hai model có kiến trúc encoder
 * khác nhau (OpenAI CLIP vs LAION OpenCLIP) nên kết quả xếp hạng thường lệch
 * nhau rõ, minh hoạ tốt cho việc so sánh. Đặt tên hằng thay vì literal rời
 * rạc trong JSX, theo đúng pattern của `DEFAULT_SPACE` ở trên.
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

  /**
   * Nếu space hiện tại không hỗ trợ `image2image` (ví dụ vừa bấm một chip
   * tiếng Việt, đặt space = `mclip-b32`, chỉ hỗ trợ text2image), request tìm
   * bằng ảnh chắc chắn bị backend từ chối (400 `ModeNotSupportedError`), và
   * `useSearch` xoá trắng cả grid kết quả để hiện lỗi — đúng đường đi mà spec
   * gọi là ranh giới giữa "demo mượt" và "demo bị kẹt". Tự chuyển sang space
   * `ready` đầu tiên hỗ trợ `image2image` (giống pattern `firstReady` ở effect
   * phía trên) để không bao giờ gửi một request chắc chắn hỏng.
   *
   * @returns space nên dùng để gọi `runSimilar`/`runImage`, cùng cờ `switched`
   *   để báo cho người dùng biết là có đổi model.
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

  /** Đóng modal rồi chạy tìm ảnh tương tự bằng `image_id` của ảnh đang xem. */
  const findSimilar = useCallback(
    (imageId: number) => {
      setSelected(null);
      const target = resolveImageSpace(space);
      if (target.switched) {
        setSpace(target.space);
        setSpaceSwitchNotice(
          `Đã tự chuyển sang model "${target.space}" vì "${space}" không hỗ trợ tìm ảnh tương tự.`,
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

  /** Dùng chung cho ảnh upload (chọn file / kéo-thả / dán) — cùng cơ chế tự
   * chuyển space như `findSimilar` ở trên, vì cả hai đều gọi `/search/image`
   * và có thể rơi vào cùng space không hỗ trợ `image2image`. */
  const searchByImage = useCallback(
    (file: File) => {
      const target = resolveImageSpace(space);
      if (target.switched) {
        setSpace(target.space);
        setSpaceSwitchNotice(
          `Đã tự chuyển sang model "${target.space}" vì "${space}" không hỗ trợ tìm bằng ảnh.`,
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
          {comparing ? "Đang ở chế độ so sánh" : "So sánh hai model"}
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
