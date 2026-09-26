import type {
  ExampleQuery,
  SearchParams,
  SearchResponse,
  SpaceInfo,
} from "../types";

/** API base URL, taken from the Vite environment variable; not hardcoded in components. */
export const API_BASE = (
  import.meta.env.VITE_API_BASE ?? "http://localhost:8000"
).replace(/\/$/, "");

/**
 * Joins `API_BASE` with a path, avoiding a double slash whether or not
 * `path` starts with `/`.
 *
 * @param path - The endpoint's relative path, e.g. `"/spaces"`.
 * @returns The full URL to the API.
 */
export function apiUrl(path: string): string {
  return `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
}

/** An API error with a status, so the UI can distinguish 409 "index not built yet" from 503 "Qdrant is down". */
export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/**
 * Reads an HTTP error response and builds the corresponding `ApiError`.
 *
 * FastAPI returns `{"detail": string}` for domain errors, or
 * `{"detail": [...]}` (an array of Pydantic validation errors) for 422
 * errors. This function normalizes both forms into a single message; if the
 * body isn't valid JSON, it keeps the default message based on the status
 * code.
 *
 * @param response - The non-`ok` HTTP response.
 * @returns An `ApiError` with the appropriate `status` and message attached.
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
    // Body isn't JSON: keep the default message based on status.
  }
  return new ApiError(response.status, detail);
}

/**
 * Calls fetch against the API and parses JSON, translating any error (a
 * non-ok HTTP response, or a network failure) into an `ApiError`.
 *
 * @param path - The endpoint path, see {@link apiUrl}.
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
    throw new ApiError(0, `Could not reach the API at ${API_BASE}. Is the API running?`);
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
 * Search by image. Pass `file` for an uploaded image, or `imageId` for an
 * image already in the corpus — the API requires exactly one of the two, so
 * this function only attaches whichever field is actually present.
 *
 * @param params - Common search parameters, plus exactly one of `file` (an
 *   image uploaded from the user's machine) or `imageId` (an image already
 *   in the corpus, used for "find similar images"). Passing both or neither
 *   is a misuse of the API contract and isn't validated at this layer.
 * @returns The search result, in the same shape as `searchText`.
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
