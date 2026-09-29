import type { SearchResultItem } from "../types";
import ResultCard from "./ResultCard";

interface Props {
  items: SearchResultItem[];
  loading: boolean;
  onOpen: (item: SearchResultItem) => void;
}

export default function ResultGrid({ items, loading, onOpen }: Props) {
  if (!loading && items.length === 0) {
    return (
      <p className="py-8 text-center text-sm text-slate-500">
        No matching results. Try removing filters or rephrasing your query.
      </p>
    );
  }
  return (
    <div
      className={`grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6 ${
        loading ? "opacity-50" : ""
      }`}
    >
      {items.map((item) => (
        <ResultCard key={`${item.image_id}-${item.rank}`} item={item} onOpen={onOpen} />
      ))}
    </div>
  );
}
