import type { SpaceInfo } from "../types";

interface Props {
  spaces: SpaceInfo[];
  value: string;
  onChange: (value: string) => void;
  label: string;
}

export default function ModelSelect({ spaces, value, onChange, label }: Props) {
  return (
    <label className="flex flex-col gap-1 text-sm">
      <span className="text-slate-400">{label}</span>
      <select
        className="rounded-md border border-slate-700 bg-slate-800 px-3 py-2 text-slate-100 focus:border-indigo-500 focus:outline-none"
        value={value}
        onChange={(event) => onChange(event.target.value)}
      >
        {spaces.map((space) => (
          <option key={space.name} value={space.name} disabled={!space.ready}>
            {space.name}
            {space.languages.includes("vi") ? " · Vietnamese" : ""}
            {space.ready ? "" : " (index not built)"}
          </option>
        ))}
      </select>
    </label>
  );
}
