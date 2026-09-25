export interface SpaceInfo {
  name: string;
  hf_id: string | null;
  dim: number | null;
  backend: string;
  modes: string[];
  languages: string[];
  ready: boolean;
  n_points: number;
  note: string;
}

export interface SearchResultItem {
  image_id: number;
  file_name: string;
  thumb_url: string;
  score: number;
  captions: string[];
  categories: string[];
  rank: number;
  matched_caption: string | null;
  caption_index: number | null;
}

export interface SearchResponse {
  results: SearchResultItem[];
  latency_ms: number;
  encode_ms: number;
  search_ms: number;
  space: string;
  exact: boolean;
  total_searched: number;
  query_echo: string;
}

export interface ExampleQuery {
  label: string;
  query: string;
  space: string;
  language: string;
}

export interface Filters {
  categories: string[];
  supercategories: string[];
}

export interface SearchParams {
  query?: string;
  space: string;
  k?: number;
  exact?: boolean;
  hnswEf?: number | null;
  filters?: Filters;
  promptTemplate?: string | null;
}
