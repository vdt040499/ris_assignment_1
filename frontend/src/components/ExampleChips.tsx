import type { ExampleQuery } from "../types";

interface Props {
  examples: ExampleQuery[];
  onPick: (example: ExampleQuery) => void;
}

export default function ExampleChips({ examples, onPick }: Props) {
  return (
    <div className="flex flex-wrap gap-2">
      {examples.map((example) => (
        <button
          key={example.label}
          className="rounded-full border border-slate-700 px-3 py-1 text-xs text-slate-300 hover:border-indigo-500 hover:text-slate-100"
          type="button"
          onClick={() => onPick(example)}
        >
          {example.language === "vi" ? "🇻🇳 " : ""}
          {example.label}
        </button>
      ))}
    </div>
  );
}
