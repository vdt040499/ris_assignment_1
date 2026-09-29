/** Choices offered for the number of returned images. */
const RESULT_COUNT_OPTIONS = [5, 10, 20, 30, 40, 50];

interface Props {
  value: number;
  onChange: (value: number) => void;
}

export default function ResultCountSelect({ value, onChange }: Props) {
  return (
    <label className="flex flex-col gap-1 text-sm">
      <span className="text-slate-400">Number of images</span>
      <select
        className="rounded-md border border-slate-700 bg-slate-800 px-3 py-2 text-slate-100 focus:border-indigo-500 focus:outline-none"
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      >
        {RESULT_COUNT_OPTIONS.map((count) => (
          <option key={count} value={count}>
            {count} images
          </option>
        ))}
      </select>
    </label>
  );
}
