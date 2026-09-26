import { File } from "node:buffer";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, apiUrl, searchImage, searchText } from "./client";

// `File` only became a global in Node 20; this machine runs Node 18, so it
// must be imported explicitly from node:buffer, otherwise the test throws
// ReferenceError.

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("apiUrl", () => {
  it("joins the base with the path", () => {
    expect(apiUrl("/spaces")).toMatch(/\/spaces$/);
  });

  it("does not double the slash", () => {
    expect(apiUrl("/spaces")).not.toMatch(/\/\/spaces$/);
  });
});

describe("searchText", () => {
  it("sends the documented json body", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse({ results: [] }));
    await searchText({ query: "a dog", space: "clip-b32", k: 5 });
    const [, init] = fetchMock.mock.calls[0];
    expect(JSON.parse(String(init?.body))).toMatchObject({
      query: "a dog",
      space: "clip-b32",
      k: 5,
    });
  });

  it("turns an error response into ApiError with its status and detail", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ detail: "index not built" }, 409),
    );
    await expect(searchText({ query: "a dog", space: "siglip-b16" })).rejects.toMatchObject({
      status: 409,
      message: "index not built",
    });
  });

  it("reports a network failure as ApiError with status 0", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("failed to fetch"));
    const error = await searchText({ query: "a dog", space: "clip-b32" }).catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(0);
  });
});

describe("searchImage", () => {
  it("posts multipart with the file and omits image_id", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse({ results: [] }));
    const file = new File([new Uint8Array([1, 2, 3])], "q.png", { type: "image/png" });
    // node:buffer's `File` (imported above so the test runs under Node 18, which
    // has no global `File`) is structurally narrower than lib.dom.d.ts's `File`
    // (missing e.g. `webkitRelativePath`), so `tsc` rejects passing it directly
    // where `searchImage` expects the browser `File` type. Note the identifier
    // `File` here is the node:buffer import (it shadows the DOM global in this
    // module), so we cast through `Parameters<...>` rather than naming the DOM
    // type directly. The cast is test-only; at runtime under vitest, `FormData`
    // accepts it identically either way.
    await searchImage({
      space: "clip-b32",
      file: file as unknown as Parameters<typeof searchImage>[0]["file"],
    });
    const body = fetchMock.mock.calls[0][1]?.body as FormData;
    expect(body.get("space")).toBe("clip-b32");
    expect(body.get("file")).toBeInstanceOf(File);
    expect(body.get("image_id")).toBeNull();
  });

  it("posts image_id when searching by a corpus image", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse({ results: [] }));
    await searchImage({ space: "clip-b32", imageId: 42 });
    const body = fetchMock.mock.calls[0][1]?.body as FormData;
    expect(body.get("image_id")).toBe("42");
    expect(body.get("file")).toBeNull();
  });

  it("serialises filters as json", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse({ results: [] }));
    await searchImage({
      space: "clip-b32",
      imageId: 1,
      filters: { categories: ["dog"], supercategories: [] },
    });
    const body = fetchMock.mock.calls[0][1]?.body as FormData;
    expect(JSON.parse(String(body.get("filters_json")))).toEqual({
      categories: ["dog"],
      supercategories: [],
    });
  });
});
