import type {
  ExampleQuery,
  SearchParams,
  SearchResponse,
  SpaceInfo,
} from "../types";

/** Địa chỉ API, lấy từ biến môi trường Vite; không viết cứng trong component. */
export const API_BASE = (
  import.meta.env.VITE_API_BASE ?? "http://localhost:8000"
).replace(/\/$/, "");

/**
 * Ghép `API_BASE` với một path, tránh double-slash dù `path` có hay không có
 * dấu `/` ở đầu.
 *
 * @param path - Đường dẫn tương đối của endpoint, ví dụ `"/spaces"`.
 * @returns URL đầy đủ tới API.
 */
export function apiUrl(path: string): string {
  return `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
}

/** Lỗi API kèm status, để UI phân biệt 409 "chưa build index" với 503 "Qdrant chết". */
export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/**
 * Đọc response lỗi HTTP và dựng `ApiError` tương ứng.
 *
 * FastAPI trả `{"detail": string}` cho lỗi miền, hoặc `{"detail": [...]}`
 * (mảng lỗi validation của Pydantic) cho lỗi 422. Hàm này chuẩn hoá cả hai
 * dạng thành một message duy nhất; nếu body không phải JSON hợp lệ thì giữ
 * thông báo mặc định theo status code.
 *
 * @param response - Response HTTP không `ok`.
 * @returns `ApiError` đã gắn `status` và message phù hợp.
 */
async function readError(response: Response): Promise<ApiError> {
  let detail = `HTTP ${response.status}`;
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") {
      detail = body.detail;
    } else if (Array.isArray(body?.detail)) {
      detail = body.detail.map((d: { msg?: string }) => d.msg ?? "").join("; ");
    }
  } catch {
    // Body không phải JSON: giữ thông báo mặc định theo status.
  }
  return new ApiError(response.status, detail);
}

/**
 * Gọi fetch tới API và giải JSON, dịch mọi lỗi (HTTP không ok, hoặc network
 * fail) thành `ApiError`.
 *
 * @param path - Đường dẫn endpoint, xem {@link apiUrl}.
 * @param init - Tuỳ chọn `fetch` (method, headers, body).
 * @returns Body JSON đã parse, ép kiểu `T`.
 * @throws {ApiError} status 0 khi network lỗi (không gọi được server); status
 *   HTTP thật khi server trả lỗi.
 */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(apiUrl(path), init);
  } catch (cause) {
    throw new ApiError(0, `Không gọi được API tại ${API_BASE}. API đã chạy chưa?`);
  }
  if (!response.ok) {
    throw await readError(response);
  }
  return (await response.json()) as T;
}

export function fetchSpaces(): Promise<SpaceInfo[]> {
  return request<SpaceInfo[]>("/spaces");
}

export function fetchExamples(): Promise<ExampleQuery[]> {
  return request<ExampleQuery[]>("/examples");
}

export function searchText(params: SearchParams): Promise<SearchResponse> {
  return request<SearchResponse>("/search/text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: params.query ?? "",
      space: params.space,
      k: params.k,
      exact: params.exact ?? true,
      hnsw_ef: params.hnswEf ?? null,
      filters: params.filters ?? { categories: [], supercategories: [] },
      prompt_template: params.promptTemplate ?? null,
    }),
  });
}

/**
 * Tìm bằng ảnh. Truyền `file` cho ảnh upload, hoặc `imageId` cho ảnh đã có
 * trong corpus — API yêu cầu đúng một trong hai, nên hàm này chỉ gắn field nào
 * thực sự có.
 *
 * @param params - Tham số tìm kiếm chung, cộng đúng một trong `file` (ảnh
 *   upload từ máy người dùng) hoặc `imageId` (ảnh đã có trong corpus, dùng
 *   cho "tìm ảnh tương tự"). Truyền cả hai hoặc không truyền gì là lỗi dùng
 *   sai hợp đồng API, không được validate ở tầng này.
 * @returns Kết quả tìm kiếm, cùng hình dạng với `searchText`.
 */
export function searchImage(
  params: SearchParams & { file?: File; imageId?: number },
): Promise<SearchResponse> {
  const form = new FormData();
  form.set("space", params.space);
  form.set("exact", String(params.exact ?? true));
  if (params.k !== undefined) form.set("k", String(params.k));
  if (params.hnswEf != null) form.set("hnsw_ef", String(params.hnswEf));
  if (params.filters) form.set("filters_json", JSON.stringify(params.filters));
  if (params.file) form.set("file", params.file);
  if (params.imageId !== undefined) form.set("image_id", String(params.imageId));
  return request<SearchResponse>("/search/image", { method: "POST", body: form });
}
