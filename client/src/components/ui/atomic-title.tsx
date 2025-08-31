'use client';

import React from 'react';
import clsx from 'clsx';

// IUPAC element symbols (1–2 letters)
const ELEMENT_SYMBOLS = [
  'H',
  'He',
  'Li',
  'Be',
  'B',
  'C',
  'N',
  'O',
  'F',
  'Ne',
  'Na',
  'Mg',
  'Al',
  'Si',
  'P',
  'S',
  'Cl',
  'Ar',
  'K',
  'Ca',
  'Sc',
  'Ti',
  'V',
  'Cr',
  'Mn',
  'Fe',
  'Co',
  'Ni',
  'Cu',
  'Zn',
  'Ga',
  'Ge',
  'As',
  'Se',
  'Br',
  'Kr',
  'Rb',
  'Sr',
  'Y',
  'Zr',
  'Nb',
  'Mo',
  'Tc',
  'Ru',
  'Rh',
  'Pd',
  'Ag',
  'Cd',
  'In',
  'Sn',
  'Sb',
  'Te',
  'I',
  'Xe',
  'Cs',
  'Ba',
  'La',
  'Ce',
  'Pr',
  'Nd',
  'Pm',
  'Sm',
  'Eu',
  'Gd',
  'Tb',
  'Dy',
  'Ho',
  'Er',
  'Tm',
  'Yb',
  'Lu',
  'Hf',
  'Ta',
  'W',
  'Re',
  'Os',
  'Ir',
  'Pt',
  'Au',
  'Hg',
  'Tl',
  'Pb',
  'Bi',
  'Po',
  'At',
  'Rn',
  'Fr',
  'Ra',
  'Ac',
  'Th',
  'Pa',
  'U',
  'Np',
  'Pu',
  'Am',
  'Cm',
  'Bk',
  'Cf',
  'Es',
  'Fm',
  'Md',
  'No',
  'Lr',
  'Rf',
  'Db',
  'Sg',
  'Bh',
  'Hs',
  'Mt',
  'Ds',
  'Rg',
  'Cn',
  'Nh',
  'Fl',
  'Mc',
  'Lv',
  'Ts',
  'Og',
];
const ELEMENT_SET = new Set(ELEMENT_SYMBOLS.map((s) => s.toLowerCase()));

type AtomicTitleProps = {
  title: string;
  className?: string;
  from?: string;
  to?: string;
  tileSize?: 'sm' | 'md' | 'lg';
};

type Token = { type: 'tile'; text: string } | { type: 'text'; text: string };

const sizeClasses = {
  sm: { box: 'w-7 h-7 text-lg', text: 'text-xl' },
  md: { box: 'w-10 h-10 text-2xl', text: 'text-3xl' },
  lg: { box: 'w-12 h-12 text-3xl', text: 'text-4xl' },
};

// deterministic tiny hash so 1-letter picks are stable per word
function hashStr(s: string) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

// Title-case a purely alphabetic word: First letter upper, rest lower
function toTitleCaseWord(w: string) {
  if (!w) return w;
  return w[0].toUpperCase() + w.slice(1).toLowerCase();
}

// Capitalize like a real element symbol: first letter upper, second lower (if any)
function formatElementSymbol(s: string) {
  if (!s) return s;
  return s.length === 1
    ? s.toUpperCase()
    : s[0].toUpperCase() + s.slice(1).toLowerCase();
}

// Create tokens for one word (letters only considered for picking; punctuation stays in place)
function tokenizeWord(word: string): Token[] {
  if (!word) return [{ type: 'text', text: word }];

  // Split into runs of letters and non-letters, so we can tile only once across the letter runs
  const parts: Array<{ text: string; isAlpha: boolean }> = [];
  let i = 0;
  const isLetter = (c: string) => /[a-z]/i.test(c);
  while (i < word.length) {
    const alpha = isLetter(word[i]);
    let j = i + 1;
    while (j < word.length && isLetter(word[j]) === alpha) j++;
    parts.push({ text: word.slice(i, j), isAlpha: alpha });
    i = j;
  }

  let tiled = false;
  const tokens: Token[] = [];

  for (const part of parts) {
    if (!part.isAlpha) {
      tokens.push({ type: 'text', text: part.text });
      continue;
    }

    const raw = part.text;
    const titled = toTitleCaseWord(raw);
    const lc = raw.toLowerCase();

    let chosen: { start: number; len: number } | null = null;

    // prefer the first (leftmost) 2-letter symbol
    if (!tiled) {
      for (let k = 0; k < lc.length - 1; k++) {
        if (ELEMENT_SET.has(lc.slice(k, k + 2))) {
          chosen = { start: k, len: 2 };
          break;
        }
      }
      // otherwise collect 1-letter options
      if (!chosen) {
        const ones: number[] = [];
        for (let k = 0; k < lc.length; k++) {
          if (ELEMENT_SET.has(lc[k])) ones.push(k);
        }
        if (ones.length === 1) chosen = { start: ones[0], len: 1 };
        else if (ones.length > 1) {
          const idx = hashStr(raw) % ones.length;
          chosen = { start: ones[idx], len: 1 };
        }
      }
    }

    if (!chosen || tiled) {
      // No tile or we've already used one in a previous alpha run
      tokens.push({ type: 'text', text: titled });
      continue;
    }

    // Build: pre (title-case), tile (proper element capitalization), post (title-case)
    const { start, len } = chosen;
    const pre = titled.slice(0, start);
    const midRaw = raw.slice(start, start + len);
    const mid = formatElementSymbol(midRaw);
    const post = titled.slice(start + len);

    if (pre) tokens.push({ type: 'text', text: pre });
    tokens.push({ type: 'tile', text: mid });
    if (post) tokens.push({ type: 'text', text: post });

    tiled = true;
  }

  // merge adjacent text tokens inside this word
  const merged: Token[] = [];
  for (const t of tokens) {
    const last = merged[merged.length - 1];
    if (t.type === 'text' && last?.type === 'text') last.text += t.text;
    else merged.push(t);
  }
  return merged;
}

const AtomicTitle = ({
  title,
  className,
  from = 'from-[#246206]',
  to = 'to-[#84bf13]',
  tileSize = 'lg',
}: AtomicTitleProps) => {
  const sz = sizeClasses[tileSize];

  const words = title.split(/\s+/).filter((w) => w.length > 0);

  return (
    <div
      className={clsx(
        'flex flex-wrap items-center justify-center gap-4 mb-10',
        className
      )}
    >
      {words.map((word, wIdx) => {
        const tokens = tokenizeWord(word);
        return (
          <span
            key={`w-${wIdx}`}
            className="inline-flex items-center flex-wrap"
          >
            {tokens.map((t, idx) =>
              t.type === 'tile' ? (
                <span
                  key={`tile-${wIdx}-${idx}`}
                  className={clsx(
                    'inline-flex select-none items-center justify-center bg-gradient-to-tl mx-2',
                    from,
                    to,
                    sz.box,
                    'font-bold text-white leading-none align-middle'
                  )}
                  aria-label={`Element tile: ${t.text}`}
                  title={t.text}
                >
                  {t.text}
                </span>
              ) : (
                <span
                  key={`text-${wIdx}-${idx}`}
                  className={clsx('leading-none align-middle', sz.text)}
                >
                  {t.text}
                </span>
              )
            )}
          </span>
        );
      })}
    </div>
  );
};

export default AtomicTitle;
