'use client';

import React, { useState, useId } from 'react';
import { ca, enUS } from 'date-fns/locale';

import {
  ChevronDownIcon,
  ChevronUpIcon,
  AlertCircleIcon,
  PaperclipIcon,
  UploadIcon,
  XIcon,
} from 'lucide-react';
import { format } from 'date-fns';
import {
  Input,
  Label,
  Checkbox,
  Button,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  Calendar,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Badge,
} from '@/components/ui';
import MultipleSelector, { Option } from '@/components/ui/multiselect';
import { formatBytes, useFileUpload } from '@/hooks/use-file-upload';

import {
  PLATFORMS,
  PUBLISHER_CLASS,
  REQUIRED_AGE,
  GENRES,
  CATEGORIES,
} from '@/lib/constants';

import api from '@/lib/api';

type Prediction = {
  owners: number;
  players: number;
  copiesSold: number;
  revenue: number;
};

type FeatureContribution = {
  feature: string;
  value: number | boolean | string | null;
  contrib_normal: number;
  abs_contrib: number;
};

type ExplainResponse = {
  predictions: Prediction;
  xai: {
    owners: FeatureContribution[];
    players: FeatureContribution[];
    copiesSold: FeatureContribution[];
    revenue: FeatureContribution[];
  };
};

const asBool = (v: boolean | 'indeterminate') => v === true;
const toISO = (d?: Date | null) => (d ? format(d, 'yyyy-MM-dd') : '');
const fromISO = (s?: string) => (s ? new Date(s + 'T00:00:00') : undefined);
type CaptionLayout = React.ComponentProps<typeof Calendar>['captionLayout'];

const fmtNum = (n: number) => n.toLocaleString();
const pctErr = (pred: number, actual: number) => {
  if (!isFinite(pred) || !isFinite(actual) || actual === 0) return '—';
  const pct = ((pred - actual) / actual) * 100;
  return `${Math.abs(pct).toFixed(2)}%`;
};
const pctAcc = (pred: number, actual: number) => {
  if (!isFinite(pred) || !isFinite(actual) || actual === 0) return '—';
  const pct = (1 - Math.abs(pred - actual) / actual) * 100;
  return `${pct.toFixed(2)}%`;
};

function downloadJSON(filename: string, data: unknown) {
  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: 'application/json',
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function Bar({ value, max }: { value: number; max: number }) {
  const width = Math.min(100, (Math.abs(value) / (max || 1)) * 100);
  const positive = value >= 0;
  return (
    <div
      className="h-2 w-full rounded bg-white/10 overflow-hidden"
      title={`${value.toFixed(2)}`}
    >
      <div
        className={`h-2 ${positive ? 'bg-emerald-500' : 'bg-rose-500'}`}
        style={{ width: `${width}%` }}
      />
    </div>
  );
}

function prettyVal(v: FeatureContribution['value']) {
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  if (typeof v === 'number')
    return Number.isInteger(v) ? v.toString() : v.toFixed(2);
  return v ?? '—';
}

function ContributionsList({ items }: { items: FeatureContribution[] }) {
  if (!items?.length)
    return <p className="text-xs text-muted-foreground">No contributions.</p>;
  const maxAbs = Math.max(...items.map((i) => i.abs_contrib || 0), 1e-6);
  return (
    <div className="flex flex-col gap-2">
      {items.map((c) => (
        <div key={c.feature} className="grid grid-cols-12 gap-3 items-center">
          <div className="col-span-4 truncate">
            <span className="text-sm text-white">{c.feature}</span>
            <span className="text-xs text-accent-mist">
              {' '}
              • {prettyVal(c.value)}
            </span>
          </div>
          <div className="col-span-6">
            <Bar value={c.contrib_normal} max={maxAbs} />
          </div>
          <div className="col-span-2 text-white text-right text-xs tabular-nums">
            {c.contrib_normal >= 0 ? '+' : ''}
            {c.contrib_normal.toFixed(2)}
            {c.contrib_normal >= 0 ? (
              <ChevronUpIcon className="inline-block ml-0.5 w-3 text-emerald-400" />
            ) : (
              <ChevronDownIcon className="inline-block ml-0.5 w-3 text-rose-400" />
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

const Predict = () => {
  const id = useId();

  const [mounted, setMounted] = React.useState(false);
  React.useEffect(() => setMounted(true), []);

  const [form, setForm] = useState({
    price: 39.99,
    is_free: false,
    required_age: 18,
    achievements: 10,
    english: true,
    windows: true,
    mac: false,
    linux: false,
    release_date: '2019-01-24',
    extract_date: '2025-02-20',
    publisherClass_encoded: 3,
  } as Record<string, any>);

  // Store selected VALUES (backend-encoded)
  const [genres, setGenres] = useState<string[]>(['Action']);
  const [categories, setCategories] = useState<string[]>([
    'Single_player',
    'Steam_Achievements',
  ]);

  const [loading, setLoading] = useState(false);
  const [pred, setPred] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [releaseDate, setReleaseDate] = React.useState<Date | undefined>(
    fromISO(form.release_date)
  );
  const [extractDate, setExtractDate] = React.useState<Date | undefined>(
    fromISO(form.extract_date)
  );
  const [releaseCaption, setReleaseCaption] =
    React.useState<CaptionLayout>('dropdown');
  const [extractCaption, setExtractCaption] =
    React.useState<CaptionLayout>('dropdown');

  const [xai, setXai] = useState<ExplainResponse['xai'] | null>(null);
  const [explaining, setExplaining] = useState(false);
  const [explainError, setExplainError] = useState<string | null>(null);

  const currentPublisherLabel =
    PUBLISHER_CLASS.find((o) => o.value === form.publisherClass_encoded)
      ?.label ?? 'Select';

  function update<K extends string>(key: K, val: any) {
    setForm((f) => ({ ...f, [key]: val }));
  }

  // derive selected options from form booleans
  const selectedPlatformOptions: Option[] = PLATFORMS.filter(
    (o) => !!form[o.value]
  );

  function handlePlatformsChange(opts: Option[]) {
    const chosen = new Set(opts.map((o) => o.value));
    update('windows', chosen.has('windows'));
    update('mac', chosen.has('mac'));
    update('linux', chosen.has('linux'));
  }

  const selectedGenreOptions = GENRES.filter((g) =>
    (genres ?? []).includes(g.value)
  );

  function handleGenresChange(opts: { label: string; value: string }[]) {
    const values = opts.map((o) => o.value);
    setGenres(values);
    update('genres', values);
  }

  const selectedCategoryOptions = CATEGORIES.filter((c) =>
    (categories ?? []).includes(c.value)
  );

  function handleCategoriesChange(opts: { label: string; value: string }[]) {
    const values = opts.map((o) => o.value);
    setCategories(values);
    update('categories', values);
  }

  function buildPayload() {
    const payload: Record<string, any> = {
      // base
      price: Number(form.price) || 0,
      is_free: Boolean(form.is_free),
      required_age: Number(form.required_age),
      achievements: Number(form.achievements) || 0,
      english: Boolean(form.english),
      windows: Boolean(form.windows),
      mac: Boolean(form.mac),
      linux: Boolean(form.linux),
      release_date: form.release_date,
      extract_date: form.extract_date,

      publisherClass_encoded: Math.min(
        3,
        Math.max(0, Number(form.publisherClass_encoded))
      ),
    };

    // genres -> boolean flags
    for (const opt of GENRES) {
      payload[opt.value] = genres.includes(opt.value);
    }

    // categories -> boolean flags
    for (const opt of CATEGORIES) {
      payload[opt.value] = categories.includes(opt.value);
    }

    return payload;
  }

  // capture the actual values
  const [actual, setActual] = useState<null | {
    owners: number;
    players: number;
    copiesSold: number;
    revenue: number;
  }>(null);

  // ---------- JSON upload helpers ----------
  const maxSize = 100 * 1024 * 1024;
  const [
    { files, isDragging, errors },
    {
      handleDragEnter,
      handleDragLeave,
      handleDragOver,
      handleDrop,
      openFileDialog,
      removeFile,
      getInputProps,
    },
  ] = useFileUpload({
    maxSize,
    multiple: false,
    accept: 'application/json,.json',
  });

  const file = files[0];

  React.useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!file) {
        setUploadedObj(null);
        setUploadError(null);
        return;
      }
      try {
        let text: string = '';
        if (file.file instanceof File) {
          text = await file.file.text();
        } else {
          setUploadedObj(null);
          setUploadError('Invalid file type.');
          return;
        }
        const json = JSON.parse(text);
        if (!cancelled) {
          setUploadedObj(json);
          setUploadError(null);
        }
      } catch (err) {
        console.error(err);
        if (!cancelled) {
          setUploadedObj(null);
          setUploadError('Invalid JSON file.');
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [file]);

  function normalizeFlagKey(k: string) {
    // "Single-player" -> "Single_player"
    return k.replace(/[\/\-\s]/g, '_');
  }

  function clampPubClass(n: any) {
    const v = Number(n);
    return Math.min(3, Math.max(0, isNaN(v) ? 0 : v));
  }

  // // Build the exact payload you already POST
  // function buildPayloadFromState() {
  //   const payload: Record<string, any> = {
  //     price: Number(form.price) || 0,
  //     is_free: Boolean(form.is_free),
  //     required_age: Number(form.required_age),
  //     achievements: Number(form.achievements) || 0,
  //     english: Boolean(form.english),
  //     windows: Boolean(form.windows),
  //     mac: Boolean(form.mac),
  //     linux: Boolean(form.linux),
  //     release_date: form.release_date,
  //     extract_date: form.extract_date,
  //     publisherClass_encoded: clampPubClass(form.publisherClass_encoded),
  //   };

  //   for (const opt of GENRES) {
  //     payload[opt.value] = genres.includes(opt.value);
  //   }
  //   for (const opt of CATEGORIES) {
  //     payload[opt.value] = categories.includes(opt.value);
  //   }

  //   return payload;
  // }

  // Apply a "payload-style" JSON object (the one you POST)
  function applyFromPayload(obj: any) {
    // base fields
    update('price', Number(obj.price) || 0);
    update('is_free', Boolean(obj.is_free));
    update('required_age', Number(obj.required_age) || 0);
    update('achievements', Number(obj.achievements) || 0);
    update('english', Boolean(obj.english));
    update('windows', Boolean(obj.windows));
    update('mac', Boolean(obj.mac));
    update('linux', Boolean(obj.linux));
    update('release_date', obj.release_date || '');
    update('extract_date', obj.extract_date || '');
    update('publisherClass_encoded', clampPubClass(obj.publisherClass_encoded));

    // dates -> calendars
    setReleaseDate(fromISO(obj.release_date));
    setExtractDate(fromISO(obj.extract_date));

    // genres/categories flags -> arrays
    const pickedGenres = GENRES.map((g) => g.value).filter((v) => !!obj[v]);
    const pickedCats = CATEGORIES.map((c) => c.value).filter((v) => !!obj[v]);
    setGenres(pickedGenres);
    setCategories(pickedCats);
    update('genres', pickedGenres);
    update('categories', pickedCats);
  }

  // Apply a "dataset_entry-style" JSON (your saved game file)
  function applyFromDatasetEntry(entry: any) {
    // basic numerics/booleans
    update('price', Number(entry.price) || 0);
    update('is_free', Boolean(entry.is_free));
    update('required_age', Number(entry.required_age) || 0);
    update('achievements', Number(entry.achievements) || 0);
    update('english', Boolean(entry.english));
    update('windows', Boolean(entry.windows));
    update('mac', Boolean(entry.mac));
    update('linux', Boolean(entry.linux));
    update(
      'publisherClass_encoded',
      clampPubClass(entry.publisherClass_encoded)
    );

    // dates from Y/M/D
    const rd = `${String(entry.release_year ?? '').split('.')[0]}-${String(
      entry.release_month ?? ''
    )
      .split('.')[0]
      .padStart(2, '0')}-${String(entry.release_day ?? '')
      .split('.')[0]
      .padStart(2, '0')}`;
    const ed = `${String(entry.extract_year ?? '').split('.')[0]}-${String(
      entry.extract_month ?? ''
    )
      .split('.')[0]
      .padStart(2, '0')}-${String(entry.extract_day ?? '')
      .split('.')[0]
      .padStart(2, '0')}`;
    update('release_date', rd);
    update('extract_date', ed);
    setReleaseDate(fromISO(rd));
    setExtractDate(fromISO(ed));

    // genres/categories: detect flags == 1 and normalize keys to your constants
    const genreVals = new Set(GENRES.map((g) => g.value));
    const catVals = new Set(CATEGORIES.map((c) => c.value));

    const pickedGenres: string[] = [];
    const pickedCats: string[] = [];

    for (const [key, rawVal] of Object.entries(entry)) {
      if (rawVal !== 1 && rawVal !== 1.0) continue;
      const norm = normalizeFlagKey(key);

      if (genreVals.has(norm)) pickedGenres.push(norm);
      if (catVals.has(norm)) pickedCats.push(norm);
    }

    setGenres(pickedGenres);
    setCategories(pickedCats);
    update('genres', pickedGenres);
    update('categories', pickedCats);

    setActual({
      owners: Number(entry.owners ?? 0),
      players: Number(entry.players ?? 0),
      copiesSold: Number(entry.copiesSold ?? 0),
      revenue: Number(entry.revenue ?? 0),
    });
  }

  // Decide which shape we received
  function applyUploadedObject(obj: any) {
    if (!obj || typeof obj !== 'object') return;

    if (
      'dataset_entry' in obj &&
      obj.dataset_entry &&
      typeof obj.dataset_entry === 'object'
    ) {
      applyFromDatasetEntry(obj.dataset_entry);
    } else {
      applyFromPayload(obj);
    }
  }

  const [uploadedObj, setUploadedObj] = useState<any | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  async function handleFileSelect(e: React.ChangeEvent<HTMLInputElement>) {
    setUploadError(null);
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      const json = JSON.parse(text);
      setUploadedObj(json);
    } catch (err: any) {
      console.error(err);
      setUploadedObj(null);
      setUploadError('Invalid JSON file.');
    }
  }

  function onPropagate() {
    if (!uploadedObj) {
      setUploadError('Please choose a JSON file first.');
      return;
    }
    try {
      applyUploadedObject(uploadedObj);
    } catch (err: any) {
      console.error(err);
      setUploadError('Failed to apply data from JSON.');
    }
  }

  const [submitted, setSubmitted] = useState(false);

  async function onSubmit() {
    setLoading(true);
    setError(null);
    setPred(null);
    setSubmitted(false);

    try {
      const payload = buildPayload();
      const res = await api.post<Prediction>('/predict', payload);
      setPred(res.data);

      try {
        setExplaining(true);
        setExplainError(null);

        // Use the SAME payload so the explanation matches the prediction
        const explainRes = await api.post<ExplainResponse>('/explain', payload);
        setXai(explainRes.data.xai); // predictions also available if you want to cross-check
      } catch (err: any) {
        console.error(err);
        setXai(null);
        const msg =
          err?.response?.data?.detail ?? 'Could not generate explanation';
        setExplainError(msg);
      } finally {
        setExplaining(false);
      }

      // Download the payload
      // downloadJSON(`payload-${Date.now()}.json`, payload);

      setSubmitted(true);
    } catch (e: any) {
      console.error(e);
      const detail = e?.response?.data?.detail || 'Prediction failed';
      setError(detail);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div
      id="predict"
      className="flex flex-col justify-center items-center mx-4 my-25"
    >
      <div className="py-10 space-y-6 w-full max-w-3xl">
        <h1 className="text-2xl font-semibold">How Will Your Game Perform?</h1>

        {/* Upload + Propagate (JSON) */}
        <div className="flex flex-col gap-2">
          {/* Drop area */}
          <div
            role="button"
            onClick={openFileDialog}
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            data-dragging={isDragging || undefined}
            className="border-input hover:bg-accent-fog data-[dragging=true]:bg-accent/50 has-[input:focus]:border-ring has-[input:focus]:ring-ring/50 flex min-h-40 flex-col items-center justify-center rounded-md border border-dashed p-4 transition-colors has-disabled:pointer-events-none has-disabled:opacity-50 has-[input:focus]:ring-[3px]"
            title="Drop a JSON file or click to browse"
          >
            <input
              {...getInputProps({
                accept: 'application/json,.json',
                multiple: false,
              })}
              className="sr-only"
              aria-label="Upload JSON file"
              disabled={Boolean(file)}
            />

            <div className="flex flex-col items-center justify-center text-center">
              <div
                className="bg-background mb-2 flex size-11 shrink-0 items-center justify-center rounded-full border"
                aria-hidden="true"
              >
                <UploadIcon className="size-4 opacity-60" />
              </div>
              <p className="mb-1.5 text-sm font-medium">Upload JSON</p>
              <p className="text-muted-foreground text-xs">
                Drag & drop or click to browse (max. {formatBytes(maxSize)})
              </p>
            </div>
          </div>

          {/* Errors from the hook or parse */}
          {(errors.length > 0 || uploadError) && (
            <div
              className="text-destructive flex items-center gap-1 text-xs"
              role="alert"
            >
              <AlertCircleIcon className="size-3 shrink-0" />
              <span>{uploadError ?? errors[0]}</span>
            </div>
          )}

          {/* File chip + actions */}
          {file && (
            <div className="flex items-center justify-between gap-2 rounded-md border border-accent-nvidia border-dashed px-4 py-2">
              <div className="flex items-center gap-3 overflow-hidden">
                <PaperclipIcon
                  className="size-4 shrink-0 opacity-60"
                  aria-hidden="true"
                />
                <div className="min-w-0">
                  <p className="truncate text-[13px] font-medium">
                    {file.file.name}
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={onPropagate}
                  disabled={!uploadedObj}
                  className="rounded-md px-4 py-2 bg-accent-nvidia text-sm font-bold text-black/80 border border-white/5 cursor-pointer hover:bg-accent-nvidia-dim transition-all active:scale-[99%]"
                >
                  Propagate from file
                </Button>

                <Button
                  size="icon"
                  variant="ghost"
                  className="text-muted-foreground/80 hover:text-foreground -me-2 size-8 hover:bg-transparent"
                  onClick={() => removeFile(file.id)}
                  aria-label="Remove file"
                >
                  <XIcon className="size-4" aria-hidden="true" />
                </Button>
              </div>
            </div>
          )}
        </div>

        <div className="border-b border-accent-fog pb-4">
          <Label htmlFor={id}>Enter Your Game Details</Label>
        </div>

        <div className="flex flex-col gap-3">
          {/* Price */}
          <div>
            {/* <Label htmlFor={id}>LABEL</Label> */}
            <div className="flex rounded-md shadow-xs">
              <span className="bg-background text-accent-mist border-accent-fog -z-10 inline-flex items-center rounded-s-md border px-3 text-sm whitespace-nowrap">
                Price USD
              </span>
              <Input
                id={id}
                className="-ms-px rounded-s-none shadow-none border-accent-fog focus-visible:border-accent-mist focus-visible:ring-1 text-accent-nvidia"
                type="number"
                min={0}
                step="0.01"
                value={form.price}
                onChange={(e) => update('price', parseFloat(e.target.value))}
              />
            </div>
          </div>

          {/* Is Free */}
          <div className="rounded-md has-data-[state=checked]:bg-gradient-to-r from-accent-nvidia/30 from-10% to-50% to-accent-charcol">
            <div className="border-accent-fog relative flex w-full items-start gap-2 rounded-md border p-4 shadow-xs outline-none backdrop-blur-3xl">
              <Checkbox
                id="is_free"
                className="order-1 after:absolute after:inset-0"
                aria-describedby="is_free-description"
                checked={!!form.is_free}
                onCheckedChange={(v) => update('is_free', asBool(v))}
              />
              <div className="grid grow gap-2">
                <Label htmlFor="is_free">
                  Is Free{' '}
                  <span className="text-muted-foreground text-xs leading-[inherit] font-normal">
                    (Pricing)
                  </span>
                </Label>
                <p
                  id="is_free-description"
                  className="text-muted-foreground text-xs"
                >
                  Is the game free to play?
                </p>
              </div>
            </div>
          </div>

          {/* English */}
          <div className="rounded-md has-data-[state=checked]:bg-gradient-to-r from-accent-nvidia/30 from-10% to-50% to-accent-charcol">
            <div className="border-accent-fog relative flex w-full items-start gap-2 rounded-md border p-4 shadow-xs outline-none backdrop-blur-3xl">
              <Checkbox
                id="english"
                className="order-1 after:absolute after:inset-0"
                aria-describedby="english-description"
                checked={!!form.english}
                onCheckedChange={(v) => update('english', v === true)}
              />
              <div className="grid grow gap-2">
                <Label htmlFor="english">
                  English{' '}
                  <span className="text-muted-foreground text-xs leading-[inherit] font-normal">
                    (Language Support)
                  </span>
                </Label>
                <p
                  id="english-description"
                  className="text-muted-foreground text-xs"
                >
                  Does the game support English?
                </p>
              </div>
            </div>
          </div>

          {/* Required Age */}
          <div>
            {/* <Label htmlFor={id}>LABEL</Label> */}
            <div className="flex rounded-md shadow-xs">
              <span className="bg-background text-accent-mist border-accent-fog -z-10 inline-flex items-center rounded-s-md border px-3 text-sm whitespace-nowrap">
                Required Age
              </span>

              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant="outline"
                    className="w-full justify-between -ms-px rounded-s-none shadow-none border-accent-fog hover:bg-accent-dusk text-accent-nvidia hover:text-accent-nvidia"
                  >
                    {form.required_age}
                    <ChevronDownIcon
                      className="opacity-60"
                      size={16}
                      aria-hidden="true"
                    />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent className="w-[var(--radix-dropdown-menu-trigger-width)] bg-accent-dusk text-white border-accent-fog p-1.5">
                  {REQUIRED_AGE.map((age) => (
                    <DropdownMenuItem
                      key={age}
                      onClick={() => update('required_age', age)}
                      className="focus:bg-accent-fog/40 focus:text-accent-white"
                    >
                      {age}
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </div>

          {/* Achievements */}
          <div>
            {/* <Label htmlFor={id}>LABEL</Label> */}
            <div className="flex rounded-md shadow-xs">
              <span className="bg-background text-accent-mist border-accent-fog -z-10 inline-flex items-center rounded-s-md border px-3 text-sm whitespace-nowrap">
                Achievements
              </span>
              <Input
                id={id}
                className="-ms-px rounded-s-none shadow-none border-accent-fog focus-visible:border-accent-mist focus-visible:ring-1 text-accent-nvidia"
                type="number"
                min={0}
                step="1"
                value={form.achievements}
                onChange={(e) =>
                  update('achievements', parseInt(e.target.value))
                }
              />
            </div>
          </div>

          {/* Platforms */}
          <div>
            <MultipleSelector
              commandProps={{ label: 'Select platforms' }}
              defaultOptions={PLATFORMS}
              value={selectedPlatformOptions}
              onChange={handlePlatformsChange}
              placeholder="Select platforms"
              emptyIndicator={
                <p className="text-center text-sm">No results found</p>
              }
              className="border-accent-fog focus-within:border-accent-dusk focus-within:ring-[2px]"
            />
          </div>

          {/* Publisher class */}
          <div className="flex rounded-md shadow-xs">
            <span className="bg-background text-accent-mist border-accent-fog -z-10 inline-flex items-center rounded-s-md border px-3 text-sm whitespace-nowrap">
              Publisher Class
            </span>

            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="outline"
                  className="w-full justify-between -ms-px rounded-s-none shadow-none border-accent-fog hover:bg-accent-dusk hover:text-accent-nvidia text-accent-nvidia"
                >
                  {currentPublisherLabel} ({form.publisherClass_encoded})
                  <ChevronDownIcon
                    className="opacity-60"
                    size={16}
                    aria-hidden="true"
                  />
                </Button>
              </DropdownMenuTrigger>

              <DropdownMenuContent className="w-[var(--radix-dropdown-menu-trigger-width)] bg-accent-dusk text-white border-accent-fog p-1.5">
                {PUBLISHER_CLASS.map((opt) => (
                  <DropdownMenuItem
                    key={opt.value}
                    onClick={() => update('publisherClass_encoded', opt.value)}
                    className="focus:bg-accent-fog/40 focus:text-accent-white"
                  >
                    {opt.label} ({opt.value})
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          </div>

          {/* Genres (multi) */}
          <div className="*not-first:mt-2">
            <MultipleSelector
              commandProps={{ label: 'Select genres' }}
              defaultOptions={GENRES}
              value={selectedGenreOptions}
              onChange={handleGenresChange}
              placeholder="Select genres"
              emptyIndicator={
                <p className="text-center text-sm">No results found</p>
              }
              className="border-accent-fog focus-within:border-accent-dusk focus-within:ring-[2px]"
            />
          </div>

          {/* Categories (multi) */}
          <div className="*not-first:mt-2">
            <MultipleSelector
              commandProps={{ label: 'Select categories' }}
              defaultOptions={CATEGORIES}
              value={selectedCategoryOptions}
              onChange={handleCategoriesChange}
              placeholder="Select categories"
              emptyIndicator={
                <p className="text-center text-sm">No results found</p>
              }
              className="border-accent-fog focus-within:border-accent-dusk focus-within:ring-[2px]"
            />
          </div>
        </div>

        {/* Dates */}
        <div className="flex flex-col sm:flex-row gap-5 justify-center w-full">
          <div className="flex flex-col gap-2 w-full">
            <span className="px-1">Release Date</span>

            <Badge variant="default" className="text-accent-nvidia">
              {releaseDate ? toISO(releaseDate) : 'Not selected'}
            </Badge>

            {mounted ? (
              <Calendar
                locale={enUS} // force stable month labels
                mode="single"
                // Avoid calling new Date() during SSR; rely on state you already have
                defaultMonth={releaseDate || undefined}
                selected={releaseDate}
                onSelect={(d) => {
                  setReleaseDate(d ?? undefined);
                  update('release_date', toISO(d));
                  if (d && extractDate && extractDate < d) {
                    setExtractDate(d);
                    update('extract_date', toISO(d));
                  }
                }}
                captionLayout={releaseCaption}
                className="rounded-lg border shadow-sm w-full"
              />
            ) : (
              // Optional: a tiny skeleton to keep layout stable
              <div className="h-80 w-[320px] rounded-lg border shadow-sm animate-pulse" />
            )}

            <Select
              value={releaseCaption ?? 'dropdown'}
              onValueChange={(v) => setReleaseCaption(v as CaptionLayout)}
            >
              <SelectTrigger
                id="release-caption"
                size="sm"
                className="bg-background w-full"
              >
                <SelectValue placeholder="Month and Year" />
              </SelectTrigger>
              <SelectContent align="center">
                <SelectItem value="dropdown">Month and Year</SelectItem>
                <SelectItem value="dropdown-months">Month Only</SelectItem>
                <SelectItem value="dropdown-years">Year Only</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="flex flex-col gap-2 w-full">
            <span className="px-1">Extract Date</span>

            <Badge variant="default" className="text-accent-nvidia">
              {extractDate ? toISO(extractDate) : 'Not selected'}
            </Badge>

            {mounted ? (
              <Calendar
                locale={enUS}
                mode="single"
                defaultMonth={extractDate || releaseDate || undefined}
                selected={extractDate}
                onSelect={(d) => {
                  if (d && releaseDate && d < releaseDate) {
                    setExtractDate(releaseDate);
                    update('extract_date', toISO(releaseDate));
                    return;
                  }
                  setExtractDate(d ?? undefined);
                  update('extract_date', toISO(d));
                }}
                captionLayout={extractCaption}
                className="rounded-lg border shadow-sm w-full"
              />
            ) : (
              <div className="h-80 w-[320px] rounded-lg border shadow-sm animate-pulse" />
            )}

            <Select
              value={extractCaption ?? 'dropdown'}
              onValueChange={(v) => setExtractCaption(v as CaptionLayout)}
            >
              <SelectTrigger
                id="extract-caption"
                size="sm"
                className="bg-background w-full"
              >
                <SelectValue placeholder="Month and Year" />
              </SelectTrigger>
              <SelectContent align="center">
                <SelectItem value="dropdown">Month and Year</SelectItem>
                <SelectItem value="dropdown-months">Month Only</SelectItem>
                <SelectItem value="dropdown-years">Year Only</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="flex gap-2 justify-center items-center flex-wrap w-full">
          <button
            onClick={onSubmit}
            disabled={loading}
            className="rounded-md px-4 py-2 bg-accent-nvidia text-sm font-bold text-black/80 border border-white/5 cursor-pointer hover:bg-accent-nvidia-dim transition-all active:scale-95 sm:w-fit w-full"
          >
            {loading ? 'Foreseeing...' : 'Foresee Your Game Sucess Now'}
          </button>

          <button
            type="button"
            onClick={() =>
              downloadJSON(`payload-${Date.now()}.json`, buildPayload())
            }
            className="rounded-md px-4 py-2 bg-accent-fog font-semibold shadow disabled:opacity-50 hover:bg-accent-fog/70 transition-all duration-100 sm:w-fit w-full"
          >
            Download Payload
          </button>
        </div>

        {error && <div className="text-red-600">{error}</div>}
      </div>

      {pred && (
        <div className="mt-6 w-full max-w-3xl">
          <h2 className="text-lg font-medium mb-2">Prediction</h2>

          <div className="flex flex-col gap-2">
            {/* horizontal rule */}
            <span className="block h-px w-full bg-gradient-to-r from-white via-white/50 to-transparent" />

            {(() => {
              // decide display source per metric; default to model pred
              const hasActual = Boolean(actual);

              const ownersVal = pred?.owners ?? 0;
              const playersVal = pred?.players ?? 0;
              const copiesVal = pred?.copiesSold ?? 0;
              const revenueVal = pred?.revenue ?? 0;

              return (
                <>
                  {/* Owners */}
                  <div className="flex justify-between">
                    <div className="flex gap-2">
                      <Badge variant="default" className="gap-1.5">
                        <span
                          className="size-1.5 rounded-full bg-accent-nvidia"
                          aria-hidden="true"
                        ></span>
                      </Badge>
                      <span>Owners</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-accent-nvidia font-semibold">
                        {fmtNum(ownersVal)}
                      </span>
                      {submitted && hasActual && (
                        <>
                          <span className="text-muted-foreground">•</span>
                          <span className="text-xs sm:text-sm">
                            Actual: {fmtNum(actual!.owners)}
                          </span>
                          <span className="text-muted-foreground">•</span>
                          <span className="text-xs sm:text-sm">
                            Error: {pctErr(ownersVal, actual!.owners)}
                          </span>
                          <span className="text-muted-foreground">•</span>
                          <span className="text-xs sm:text-sm">
                            Accuracy: {pctAcc(ownersVal, actual!.owners)}
                          </span>
                        </>
                      )}
                    </div>
                  </div>

                  <span className="block h-px w-full bg-gradient-to-r from-white via-white/50 to-transparent" />

                  {/* Players */}
                  <div className="flex justify-between">
                    <div className="flex gap-2">
                      <Badge variant="default" className="gap-1.5">
                        <span
                          className="size-1.5 rounded-full bg-blue-500"
                          aria-hidden="true"
                        ></span>
                      </Badge>
                      <span>Players</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-blue-500 font-semibold">
                        {fmtNum(playersVal)}
                      </span>
                      {submitted && hasActual && (
                        <>
                          <span className="text-muted-foreground">•</span>
                          <span className="text-xs sm:text-sm">
                            Actual: {fmtNum(actual!.players)}
                          </span>
                          <span className="text-muted-foreground">•</span>
                          <span className="text-xs sm:text-sm">
                            Error: {pctErr(playersVal, actual!.players)}
                          </span>
                          <span className="text-muted-foreground">•</span>
                          <span className="text-xs sm:text-sm">
                            Accuracy: {pctAcc(playersVal, actual!.players)}
                          </span>
                        </>
                      )}
                    </div>
                  </div>

                  <span className="block h-px w-full bg-gradient-to-r from-white via-white/50 to-transparent" />

                  {/* Copies Sold */}
                  <div className="flex justify-between">
                    <div className="flex gap-2">
                      <Badge variant="default" className="gap-1.5">
                        <span
                          className="size-1.5 rounded-full bg-rose-500"
                          aria-hidden="true"
                        ></span>
                      </Badge>
                      <span>Copies Sold</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-rose-500 font-semibold">
                        {fmtNum(copiesVal)}
                      </span>
                      {submitted && hasActual && (
                        <>
                          <span className="text-muted-foreground">•</span>
                          <span className="text-xs sm:text-sm">
                            Actual: {fmtNum(actual!.copiesSold)}
                          </span>
                          <span className="text-muted-foreground">•</span>
                          <span className="text-xs sm:text-sm">
                            Error: {pctErr(copiesVal, actual!.copiesSold)}
                          </span>
                          <span className="text-muted-foreground">•</span>
                          <span className="text-xs sm:text-sm">
                            Accuracy: {pctAcc(copiesVal, actual!.copiesSold)}
                          </span>
                        </>
                      )}
                    </div>
                  </div>

                  <span className="block h-px w-full bg-gradient-to-r from-white via-white/50 to-transparent" />

                  {/* Revenue */}
                  <div className="flex justify-between">
                    <div className="flex gap-2">
                      <Badge variant="default" className="gap-1.5">
                        <span
                          className="size-1.5 rounded-full bg-yellow-600"
                          aria-hidden="true"
                        ></span>
                      </Badge>
                      <span>Revenue</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-yellow-600 font-semibold">
                        ${fmtNum(revenueVal)}
                      </span>
                      {submitted && hasActual && (
                        <>
                          <span className="text-muted-foreground">•</span>
                          <span className="text-xs sm:text-sm">
                            Actual: ${fmtNum(actual!.revenue)}
                          </span>
                          <span className="text-muted-foreground">•</span>
                          <span className="text-xs sm:text-sm">
                            Error: {pctErr(revenueVal, actual!.revenue)}
                          </span>
                          <span className="text-muted-foreground">•</span>
                          <span className="text-xs sm:text-sm">
                            Accuracy: {pctAcc(revenueVal, actual!.revenue)}
                          </span>
                        </>
                      )}
                    </div>
                  </div>

                  <span className="block h-px w-full bg-gradient-to-r from-white via-white/50 to-transparent" />
                </>
              );
            })()}
          </div>
        </div>
      )}

      {xai && (
        <div className="mt-8 w-full max-w-3xl">
          <div className="flex items-center justify-between mb-2">
            {!explaining && (
              <h2 className="text-lg font-medium">Why did we predict this?</h2>
            )}
            {explaining && (
              <span className="text-md text-accent-nvidia">
                Computing Explanations…
              </span>
            )}
          </div>

          {explainError && (
            <div className="text-destructive text-sm mb-3">{explainError}</div>
          )}

          {!explaining && (
            <div className="text-sm text-muted-foreground">
              <div className="space-y-6">
                <div className="rounded-lg border border-accent-fog p-4">
                  <div className="flex items-center gap-1 mb-2">
                    <Badge
                      variant="default"
                      className="px-4 py-1 gap-2 bg-accent-nvidia rounded-md rounded-r-none"
                    >
                      <span className="font-semibold text-black">Owners</span>
                    </Badge>
                    <Badge
                      variant="default"
                      className="px-4 py-1 bg-accent-nvidia/20 rounded-l-none"
                    >
                      <span className="font-semibold">Top Drivers</span>
                    </Badge>
                  </div>

                  <ContributionsList items={xai.owners} />
                </div>

                <div className="rounded-lg border border-accent-fog p-4">
                  <div className="flex items-center gap-1 mb-2">
                    <Badge
                      variant="default"
                      className="px-4 py-1 gap-2 bg-blue-600 rounded-md rounded-r-none"
                    >
                      <span className="font-semibold text-black">Players</span>
                    </Badge>
                    <Badge
                      variant="default"
                      className="px-4 py-1 bg-blue-600/20 rounded-l-none"
                    >
                      <span className="font-semibold">Top Drivers</span>
                    </Badge>
                  </div>
                  <ContributionsList items={xai.players} />
                </div>

                <div className="rounded-lg border border-accent-fog p-4">
                  <div className="flex items-center gap-1 mb-2">
                    <Badge
                      variant="default"
                      className="px-4 py-1 gap-2 bg-rose-600 rounded-md rounded-r-none"
                    >
                      <span className="font-semibold text-black">
                        Copies Sold
                      </span>
                    </Badge>
                    <Badge
                      variant="default"
                      className="px-4 py-1 bg-rose-600/20 rounded-l-none"
                    >
                      <span className="font-semibold">Top Drivers</span>
                    </Badge>
                  </div>
                  <ContributionsList items={xai.copiesSold} />
                </div>

                <div className="rounded-lg border border-accent-fog p-4">
                  <div className="flex items-center gap-1 mb-2">
                    <Badge
                      variant="default"
                      className="px-4 py-1 gap-2 bg-yellow-500 rounded-md rounded-r-none"
                    >
                      <span className="font-semibold text-black">Revenue</span>
                    </Badge>
                    <Badge
                      variant="default"
                      className="px-4 py-1 bg-yellow-600/20 rounded-l-none"
                    >
                      <span className="font-semibold">Top Drivers</span>
                    </Badge>
                  </div>
                  <ContributionsList items={xai.revenue} />
                </div>
              </div>
            </div>
          )}

          {/* <div className="space-y-6">
            <div className="rounded-lg border border-accent-fog p-4">
              <div className="flex items-center gap-1 mb-2">
                <Badge
                  variant="default"
                  className="px-4 py-1 gap-2 bg-accent-nvidia rounded-md rounded-r-none"
                >
                  <span className="font-semibold text-black">Owners</span>
                </Badge>
                <Badge
                  variant="default"
                  className="px-4 py-1 bg-accent-nvidia/20 rounded-l-none"
                >
                  <span className="font-semibold">Top Drivers</span>
                </Badge>
              </div>

              <ContributionsList items={xai.owners} />
            </div>

            <div className="rounded-lg border border-accent-fog p-4">
              <div className="flex items-center gap-1 mb-2">
                <Badge
                  variant="default"
                  className="px-4 py-1 gap-2 bg-blue-600 rounded-md rounded-r-none"
                >
                  <span className="font-semibold text-black">Players</span>
                </Badge>
                <Badge
                  variant="default"
                  className="px-4 py-1 bg-blue-600/20 rounded-l-none"
                >
                  <span className="font-semibold">Top Drivers</span>
                </Badge>
              </div>
              <ContributionsList items={xai.players} />
            </div>

            <div className="rounded-lg border border-accent-fog p-4">
              <div className="flex items-center gap-1 mb-2">
                <Badge
                  variant="default"
                  className="px-4 py-1 gap-2 bg-rose-600 rounded-md rounded-r-none"
                >
                  <span className="font-semibold text-black">Copies Sold</span>
                </Badge>
                <Badge
                  variant="default"
                  className="px-4 py-1 bg-rose-600/20 rounded-l-none"
                >
                  <span className="font-semibold">Top Drivers</span>
                </Badge>
              </div>
              <ContributionsList items={xai.copiesSold} />
            </div>

            <div className="rounded-lg border border-accent-fog p-4">
              <div className="flex items-center gap-1 mb-2">
                <Badge
                  variant="default"
                  className="px-4 py-1 gap-2 bg-yellow-500 rounded-md rounded-r-none"
                >
                  <span className="font-semibold text-black">Revenue</span>
                </Badge>
                <Badge
                  variant="default"
                  className="px-4 py-1 bg-yellow-600/20 rounded-l-none"
                >
                  <span className="font-semibold">Top Drivers</span>
                </Badge>
              </div>
              <ContributionsList items={xai.revenue} />
            </div>
          </div> */}
        </div>
      )}
    </div>
  );
};

export default Predict;
