import type {
  ExampleQuery,
  SearchParams,
  SearchResponse,
  SpaceInfo,
} from "../types";

/** API base URL, read from a Vite environment variable; not hardcoded in components. */
export const API_BASE = (
  import.meta.env.VITE_API_BASE ?? "http://localhost:8000"
).replace(/\/$/, "");

/**
 * Joins `API_BASE` with a path, avoiding a double slash whether or not `path`
 * starts with `/`.
 *
 * @param path - Relative endpoint path, e.g. `"/spaces"`.
 * @returns The full API URL.
 */
export function apiUrl(path: string): string {
  return `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
}

/** API error carrying the HTTP status, so the UI can tell 409 "index not built" from 503 "Qdrant down". */
export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/**
 * Reads an HTTP error response and builds the matching `ApiError`.
 *
 * FastAPI returns `{"detail": string}` for domain errors, or `{"detail": [...]}`
 * (a Pydantic validation error array) for 422. This function normalizes both
 * into a single message; if the body is not valid JSON, the default message
 * for the status code is kept.
 *
 * @param response - A non-`ok` HTTP response.
 * @returns An `ApiError` with `status` and a suitable message.
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
    // Body is not JSON: keep the default message for the status.
  }
  return new ApiError(response.status, detail);
}

/**
 * Calls the API with fetch and parses the JSON, turning every failure (non-ok
 * HTTP or a network error) into an `ApiError`.
 *
 * @param path - Endpoint path, see {@link apiUrl}.
 * @param init - `fetch` options (method, headers, body).
 * @returns The parsed JSON body, cast to `T`.
 * @throws {ApiError} status 0 on a network failure (server unreachable); the
 *   real HTTP status when the server returns an error.
 */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(apiUrl(path), init);
  } catch (cause) {
    throw new ApiError(0, `Cannot reach the API at ${API_BASE}. Is the API running?`);
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
 * Searches by image. Pass `file` for an uploaded image, or `imageId` for an
 * image already in the corpus. The API requires exactly one of the two, so
 * this function only sets whichever field is actually present.
 *
 * @param params - Common search params plus exactly one of `file` (an image
 *   uploaded from the user's machine) or `imageId` (an image already in the
 *   corpus, used for "find similar images"). Passing both or neither breaks
 *   the API contract and is not validated at this layer.
 * @returns The search results, same shape as `searchText`.
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
