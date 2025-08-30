'use client';

import React, { useState, useId } from 'react';

import { Input, Label, Checkbox } from '@/components/ui';

import api from '@/lib/api';
import {
  PUBLISHER_CLASS_OPTIONS,
  REQUIRED_AGE_OPTIONS,
  GENRES,
  CATEGORIES,
  DEVELOPERS,
  PUBLISHERS,
} from '@/lib/constants';

type Prediction = {
  owners: number;
  players: number;
  copiesSold: number;
  revenue: number;
};

const Predict = () => {
  const id = useId();

  const asBool = (v: boolean | 'indeterminate') => v === true;

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

  async function onSubmit() {
    setLoading(true);
    setError(null);
    setPred(null);

    try {
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

        // publisher class (validated to 0..3)
        publisherClass_encoded: Math.min(
          3,
          Math.max(0, Number(form.publisherClass_encoded))
        ),

        // Send encoded values (single for current backend)
        developer: developers[0] ?? 'dev__Other',
        publisher: publishers[0] ?? 'pub__Other',

        // also send arrays in case backend supports multi later
        developers,
        publishers,
      };

      // Turn genre selections into boolean flags using encoded values
      for (const opt of GENRES) {
        payload[opt.value] = genres.includes(opt.value);
      }

      // Turn category selections into boolean flags using encoded values
      for (const opt of CATEGORIES) {
        payload[opt.value] = categories.includes(opt.value);
      }

      const res = await api.post<Prediction>('/predict', payload);
      setPred(res.data);
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

        <div className="flex flex-col">
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
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={form.is_free}
              onChange={(e) => update('is_free', e.target.checked)}
            />
            <span>Is Free</span>
          </label>
          <div className="border-input has-data-[state=checked]:border-primary/50 relative flex w-full items-start gap-2 rounded-md border p-4 shadow-xs outline-none">
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

          <div className="border-input has-data-[state=checked]:border-primary/50 relative flex w-full items-start gap-2 rounded-md border p-4 shadow-xs outline-none">
            <Checkbox
              id={id}
              className="order-1 after:absolute after:inset-0"
              aria-describedby={`${id}-description`}
            />
            <div className="grid grow gap-2">
              <Label htmlFor={id}>
                English{' '}
                <span className="text-muted-foreground text-xs leading-[inherit] font-normal">
                  (Language Support)
                </span>
              </Label>
              <p
                id={`${id}-description`}
                className="text-muted-foreground text-xs"
              >
                Does the game support English?
              </p>
            </div>
          </div>

          {/* Required Age */}
          <label className="flex flex-col">
            <span>Required Age</span>
            <select
              value={form.required_age}
              onChange={(e) => update('required_age', parseInt(e.target.value))}
              className="rounded-xl border px-3 py-2"
            >
              {REQUIRED_AGE_OPTIONS.map((age) => (
                <option key={age} value={age}>
                  {age}
                </option>
              ))}
            </select>
          </label>

          {/* Achievements */}
          <label className="flex flex-col">
            <span>Achievements</span>
            <input
              type="number"
              min={0}
              value={form.achievements}
              onChange={(e) => update('achievements', parseInt(e.target.value))}
              className="rounded-xl border px-3 py-2"
            />
          </label>

          {/* English */}
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={form.english}
              onChange={(e) => update('english', e.target.checked)}
            />
            <span>English</span>
          </label>

          {/* Platforms */}
          <div className="flex flex-wrap items-center gap-4">
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={form.windows}
                onChange={(e) => update('windows', e.target.checked)}
              />
              <span>Windows</span>
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={form.mac}
                onChange={(e) => update('mac', e.target.checked)}
              />
              <span>Mac</span>
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={form.linux}
                onChange={(e) => update('linux', e.target.checked)}
              />
              <span>Linux</span>
            </label>
          </div>

          {/* Dates */}
          <label className="flex flex-col">
            <span>Release Date</span>
            <input
              type="date"
              value={form.release_date}
              onChange={(e) => update('release_date', e.target.value)}
              className="rounded-xl border px-3 py-2"
            />
          </label>

          <label className="flex flex-col">
            <span>Extract Date</span>
            <input
              type="date"
              value={form.extract_date}
              onChange={(e) => update('extract_date', e.target.value)}
              className="rounded-xl border px-3 py-2"
            />
          </label>

          {/* Publisher class */}
          <label className="flex flex-col">
            <span>Publisher Class</span>
            <select
              value={form.publisherClass_encoded}
              onChange={(e) =>
                update('publisherClass_encoded', parseInt(e.target.value))
              }
              className="rounded-xl border px-3 py-2"
            >
              {PUBLISHER_CLASS_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label} ({opt.value})
                </option>
              ))}
            </select>
          </label>

          {/* Developers (multi) */}
          <label className="flex flex-col">
            <span>Developers (multi)</span>
            <select
              multiple
              value={developers}
              onChange={(e) => onMultiChange(e, setDevelopers)}
              className="rounded-xl border px-3 py-2 h-40"
            >
              {DEVELOPERS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
            <p className="text-xs text-gray-500 mt-1">
              Tip: This list is a subset. You can expand it or load options from
              the backend.
            </p>
          </label>

          {/* Publishers (multi) */}
          <label className="flex flex-col">
            <span>Publishers (multi)</span>
            <select
              multiple
              value={publishers}
              onChange={(e) => onMultiChange(e, setPublishers)}
              className="rounded-xl border px-3 py-2 h-40"
            >
              {PUBLISHERS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
            <p className="text-xs text-gray-500 mt-1">
              Tip: This list is a subset. You can expand it or load options from
              the backend.
            </p>
          </label>

          {/* Genres (multi) */}
          <label className="flex flex-col">
            <span>Genres (multi)</span>
            <select
              multiple
              value={genres}
              onChange={(e) => onMultiChange(e, setGenres)}
              className="rounded-xl border px-3 py-2 h-40"
            >
              {GENRES.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </label>

          {/* Categories (multi) */}
          <label className="flex flex-col">
            <span>Categories (multi)</span>
            <select
              multiple
              value={categories}
              onChange={(e) => onMultiChange(e, setCategories)}
              className="rounded-xl border px-3 py-2 h-56"
            >
              {CATEGORIES.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </label>
        </div>

        <button
          onClick={onSubmit}
          disabled={loading}
          className="rounded-2xl px-4 py-2 border shadow disabled:opacity-50"
        >
          {loading ? 'Predicting…' : 'Predict'}
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
