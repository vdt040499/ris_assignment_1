import { API_BASE } from "../api/client";
import type { SearchResultItem } from "../types";

interface Props {
  item: SearchResultItem | null;
  onClose: () => void;
  onFindSimilar: (imageId: number) => void;
}

export default function DetailModal({ item, onClose, onFindSimilar }: Props) {
  if (!item) return null;
  return (
    <div
      className="fixed inset-0 z-10 flex items-center justify-center bg-slate-950/80 p-4"
      onClick={onClose}
    >
      <div
        className="flex max-h-full w-full max-w-3xl flex-col gap-4 overflow-auto rounded-lg border border-slate-700 bg-slate-900 p-4"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-lg font-medium">Ảnh {item.image_id}</h2>
            <p className="text-sm text-slate-400">
              hạng {item.rank} · điểm {item.score.toFixed(4)}
            </p>
          </div>
          <button
            className="rounded-md border border-slate-700 px-3 py-1 text-sm hover:border-indigo-500"
            type="button"
            onClick={onClose}
          >
            Đóng
          </button>
        </div>

        <img
          className="max-h-[50vh] w-full rounded-md object-contain"
          src={`${API_BASE}/images/${item.file_name}`}
          alt={item.captions[0] ?? `ảnh ${item.image_id}`}
        />

        <div className="flex flex-col gap-2 text-sm">
          <h3 className="text-slate-300">Caption của người viết</h3>
          <ul className="list-disc pl-5 text-slate-400">
            {item.captions.map((caption, index) => (
              <li key={index}>{caption}</li>
            ))}
          </ul>
          {item.categories.length > 0 && (
            <p className="text-slate-400">
              <span className="text-slate-300">Category: </span>
              {item.categories.join(", ")}
            </p>
          )}
        </div>

        <button
          className="self-start rounded-md bg-indigo-500 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-400"
          type="button"
          onClick={() => onFindSimilar(item.image_id)}
        >
          Tìm ảnh tương tự
        </button>
      </div>
    </div>
  );
}
