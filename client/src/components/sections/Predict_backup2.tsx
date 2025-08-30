'use client';

import React, { useState, useId } from 'react';
import { enUS } from 'date-fns/locale';

import { ChevronDownIcon } from 'lucide-react';
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
} from '@/components/ui';
import MultipleSelector, { Option } from '@/components/ui/multiselect';

import {
  PLATFORMS,
  PUBLISHER_CLASS,
  REQUIRED_AGE,
  GENRES,
  CATEGORIES,
  DEVELOPERS,
  PUBLISHERS,
} from '@/lib/constants';

import api from '@/lib/api';

type Prediction = {
  owners: number;
  players: number;
  copiesSold: number;
  revenue: number;
};

const asBool = (v: boolean | 'indeterminate') => v === true;
const toISO = (d?: Date | null) => (d ? format(d, 'yyyy-MM-dd') : '');
const fromISO = (s?: string) => (s ? new Date(s + 'T00:00:00') : undefined);
type CaptionLayout = React.ComponentProps<typeof Calendar>['captionLayout'];

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

const Predict = () => {
  const id = useId();

  const [dropdown, setDropdown] =
    React.useState<React.ComponentProps<typeof Calendar>['captionLayout']>(
      'dropdown'
    );
  const [date, setDate] = React.useState<Date | undefined>(new Date());

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
  const [developers, setDevelopers] = useState<string[]>(['dev__Other']);
  const [publishers, setPublishers] = useState<string[]>(['pub__Other']);

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

  const currentPublisherLabel =
    PUBLISHER_CLASS.find((o) => o.value === form.publisherClass_encoded)
      ?.label ?? 'Select';

  function update<K extends string>(key: K, val: any) {
    setForm((f) => ({ ...f, [key]: val }));
  }

  function onMultiChange(
    e: React.ChangeEvent<HTMLSelectElement>,
    setter: (vals: string[]) => void
  ) {
    const selected = Array.from(e.target.selectedOptions).map((o) => o.value);
    setter(selected);
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

  const selectedDeveloperOptions = DEVELOPERS.filter((d) =>
    (form.developers ?? []).includes(d.value)
  );

  function handleDevelopersChange(opts: { label: string; value: string }[]) {
    const values = opts.map((o) => o.value);
    setDevelopers(values);
    update('developers', values);
  }

  const selectedPublisherOptions = PUBLISHERS.filter((p) =>
    (form.publishers ?? []).includes(p.value)
  );

  function handlePublishersChange(opts: { label: string; value: string }[]) {
    const values = opts.map((o) => o.value);
    setPublishers(values);
    update('publishers', values);
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
    update('categories', values); // keep in sync with form
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

      // single encoded for current backend
      developer: developers[0] ?? 'dev__Other',
      publisher: publishers[0] ?? 'pub__Other',

      // keep arrays too (future-proof / debugging)
      developers,
      publishers,
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

  async function onSubmit() {
    setLoading(true);
    setError(null);
    setPred(null);

    try {
      const payload = buildPayload();
      const res = await api.post<Prediction>('/predict', payload);
      setPred(res.data);

      // Download exactly what you sent
      downloadJSON(`payload-${Date.now()}.json`, payload);
    } catch (e: any) {
      console.error(e);
      const detail = e?.response?.data?.detail || 'Prediction failed';
      setError(detail);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="md:max-w-4xl sm:max-w-sm sm:mx-auto top-0 left-0 right-0 mt-25">
      <div className="max-w-4xl mx-auto py-10 space-y-6">
        <h1 className="text-2xl font-semibold">Predict Steam Game Metrics</h1>

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
                className="-ms-px rounded-s-none shadow-none border-accent-fog focus-visible:border-accent-mist focus-visible:ring-1"
                type="number"
                min={0}
                step="0.01"
                value={form.price}
                onChange={(e) => update('price', parseFloat(e.target.value))}
              />
            </div>
          </div>

          {/* Is Free */}
          <div className="rounded-md has-data-[state=checked]:bg-gradient-to-r from-accent-mist/30 from-10% to-50% to-accent-charcol">
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
          <div className="rounded-md has-data-[state=checked]:bg-gradient-to-r from-accent-mist/30 from-10% to-50% to-accent-charcol">
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
                {'Required Age >='}
              </span>

              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant="outline"
                    className="w-full justify-between -ms-px rounded-s-none shadow-none border-accent-fog hover:bg-accent-dusk hover:text-white"
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
                className="-ms-px rounded-s-none shadow-none border-accent-fog focus-visible:border-accent-mist focus-visible:ring-1"
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
          <div className="*not-first:mt-2">
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
                  className="w-full justify-between -ms-px rounded-s-none shadow-none border-accent-fog hover:bg-accent-dusk hover:text-white"
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

          {/* Developers (multi) */}
          <div className="*not-first:mt-2">
            <MultipleSelector
              commandProps={{ label: 'Select developers' }}
              defaultOptions={DEVELOPERS}
              value={selectedDeveloperOptions}
              onChange={handleDevelopersChange}
              placeholder="Select developers"
              emptyIndicator={
                <p className="text-center text-sm">No results found</p>
              }
              className="border-accent-fog focus-within:border-accent-dusk focus-within:ring-[2px]"
            />
          </div>

          {/* Publishers (multi) */}
          <div className="*not-first:mt-2">
            <MultipleSelector
              commandProps={{ label: 'Select publishers' }}
              defaultOptions={PUBLISHERS}
              value={selectedPublisherOptions}
              onChange={handlePublishersChange}
              placeholder="Select publishers"
              emptyIndicator={
                <p className="text-center text-sm">No results found</p>
              }
              className="border-accent-fog focus-within:border-accent-dusk focus-within:ring-[2px]"
            />
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
        <div className="flex gap-5">
          <div className="flex flex-col gap-2 w-fit">
            <span className="px-1">Release Date</span>

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
                className="rounded-lg border shadow-sm"
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

          <div className="flex flex-col gap-2 w-fit">
            <span className="px-1">Extract Date</span>

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
                className="rounded-lg border shadow-sm"
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

        <button
          onClick={onSubmit}
          disabled={loading}
          className="rounded-md px-4 py-2 bg-accent-nvidia text-accent-charcol font-semibold shadow disabled:opacity-50 hover:bg-accent-nvidia-dim transition-all duration-100"
        >
          {loading ? 'Predicting…' : 'Predict'}
        </button>

        <button
          type="button"
          onClick={() =>
            downloadJSON(`payload-${Date.now()}.json`, buildPayload())
          }
          className="rounded-md px-4 py-2 bg-accent-fog text-accent-charcol font-semibold shadow disabled:opacity-50 hover:bg-accent-mist transition-all duration-100"
        >
          Download JSON
        </button>

        {error && <div className="text-red-600">{error}</div>}

        {pred && (
          <div className="mt-6">
            <h2 className="text-lg font-medium mb-2">Prediction</h2>
            <div className="overflow-x-auto">
              <table className="min-w-full text-left text-sm">
                <tbody>
                  <tr>
                    <td className="px-3 py-2 border-b">Owners</td>
                    <td className="px-3 py-2 border-b">
                      {pred.owners.toLocaleString()}
                    </td>
                  </tr>
                  <tr>
                    <td className="px-3 py-2 border-b">Players</td>
                    <td className="px-3 py-2 border-b">
                      {pred.players.toLocaleString()}
                    </td>
                  </tr>
                  <tr>
                    <td className="px-3 py-2 border-b">Copies Sold</td>
                    <td className="px-3 py-2 border-b">
                      {pred.copiesSold.toLocaleString()}
                    </td>
                  </tr>
                  <tr>
                    <td className="px-3 py-2 border-b">Revenue</td>
                    <td className="px-3 py-2 border-b">
                      ${pred.revenue.toLocaleString()}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Predict;
