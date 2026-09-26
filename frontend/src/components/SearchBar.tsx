import { useRef, useState } from "react";

interface Props {
  onSearchText: (query: string) => void;
  onSearchImage: (file: File) => void;
  disabled: boolean;
}

export default function SearchBar({ onSearchText, onSearchImage, disabled }: Props) {
  const [text, setText] = useState("");
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  // Shared by drag-drop, clipboard paste, and file input: takes the first
  // image in the file list, ignoring it if it's not an image (e.g. pasted text).
  function pickFirstImage(items: FileList | null) {
    const file = items?.[0];
    if (file && file.type.startsWith("image/")) {
      onSearchImage(file);
    }
  }

  return (
    <div
      className={`flex flex-col gap-3 rounded-lg border-2 border-dashed p-4 transition-colors ${
        dragging ? "border-indigo-500 bg-slate-800" : "border-slate-700"
      }`}
      onDragOver={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => {
        event.preventDefault();
        setDragging(false);
        pickFirstImage(event.dataTransfer.files);
      }}
      onPaste={(event) => pickFirstImage(event.clipboardData.files)}
    >
      <form
        className="flex gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          if (text.trim()) onSearchText(text.trim());
        }}
      >
        <input
          className="flex-1 rounded-md border border-slate-700 bg-slate-800 px-3 py-2 text-slate-100 placeholder:text-slate-500 focus:border-indigo-500 focus:outline-none"
          placeholder="Describe the image you want to find, e.g.: a man riding a horse on the beach"
          value={text}
          onChange={(event) => setText(event.target.value)}
        />
        <button
          className="rounded-md bg-indigo-500 px-4 py-2 font-medium text-white hover:bg-indigo-400 disabled:opacity-50"
          type="submit"
          disabled={disabled || !text.trim()}
        >
          Search
        </button>
      </form>

      <div className="flex items-center gap-3 text-sm text-slate-400">
        <button
          className="rounded-md border border-slate-700 px-3 py-1 hover:border-indigo-500"
          type="button"
          onClick={() => fileInput.current?.click()}
        >
          Choose image
        </button>
        <span>or drag-and-drop / paste an image into this box to search by image</span>
        <input
          ref={fileInput}
          className="hidden"
          type="file"
          accept="image/*"
          onChange={(event) => pickFirstImage(event.target.files)}
        />
      </div>
    </div>
  );
}
