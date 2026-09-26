import { API_BASE } from "../api/client";
import type { SearchResultItem } from "../types";

interface Props {
  item: SearchResultItem;
  onOpen: (item: SearchResultItem) => void;
}

export default function ResultCard({ item, onOpen }: Props) {
  return (
    <button
      className="group relative overflow-hidden rounded-lg border border-slate-700 text-left hover:border-indigo-500"
      type="button"
      onClick={() => onOpen(item)}
    >
      <img
        className="aspect-square w-full object-cover"
        src={`${API_BASE}${item.thumb_url}`}
        alt={item.captions[0] ?? `image ${item.image_id}`}
        loading="lazy"
      />
      <span className="absolute left-1 top-1 rounded bg-slate-900/80 px-1.5 py-0.5 text-xs text-slate-200">
        #{item.rank} · {item.score.toFixed(3)}
      </span>
      <span className="absolute inset-x-0 bottom-0 line-clamp-2 bg-slate-900/85 p-2 text-xs text-slate-200 opacity-0 transition-opacity group-hover:opacity-100">
        {item.matched_caption ?? item.captions[0] ?? ""}
      </span>
    </button>
  );
}
