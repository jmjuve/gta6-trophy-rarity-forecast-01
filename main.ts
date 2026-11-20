// gta6-trophy-rarity-forecast-01 — forecast program (Node 22, standard library only).
//
// Bismillah. What this program believes about the data it is given, in plain terms:
//
// 1. Exophase rates are a ratio: earners / players-tracked. Both move. On launch
//    night the tracked population was 638 ultra-hardcore players, so nearly every
//    trophy sat at the same ~1.1% (7 earners out of 638). By the next morning the
//    denominator had exploded and most rates had collapsed to 0.03-0.7%. A forecast
//    that just repeats the last read, or that extrapolates the launch-night collapse
//    forward, is wrong in opposite directions. The honest move is to anchor on the
//    last read, look only at the RECENT trend (last 24 h of reads, where the
//    denominator has mostly settled), and shrink that trend toward a prior that
//    knows what kind of trophy each one is.
//
// 2. Trophies behave differently by kind, and the kind is readable from the
//    trophy's own description, so the program classifies each trophy from its
//    text at run time (nothing is hard-coded by id, because the final case could
//    list a different set):
//      - opening / early story (first mission, chapter 1): almost every tracked
//        player reaches these within days, so their rates climb fastest;
//      - later chapters and named story missions: climb, slower the later they are;
//      - the final mission, the platinum, 100% completion: almost nobody finishes
//        a ~200-hour game in week one, and the denominator keeps growing, so
//        these drift DOWN in logit terms;
//      - long grinds (250 stunts, gold in every shooting-range challenge, 100
//        WAiNK scans, card sets, 50 challenges): near zero all week, drift down;
//      - quick one-off activities (an arcade game, a fishing challenge, trying
//        each activity once): easy, climb moderately;
//      - crime and misc one-offs (rob a truck, lose a 4-star wanted level, a
//        cocktail on Ocean Beach): climb a little;
//      - money sinks (launder $250k, spend $100k, buy every garage): slow.
//    Each class carries a prior drift slope (logit units per unit of
//    ln(1 + hours since release)) and a prior spread. The data can override the
//    prior when the recent history actually shows a trend; when the history is
//    flat or thin, the prior does the work.
//
// 3. The spread is where the win lives. The house persistence baseline spreads
//    0.5 * sqrt(days) logits — about 1.06 at the 4.52-day final horizon — which
//    is far wider than the real uncertainty once you know the trophy's kind.
//    This program's spread is per-class (0.40 for near-frozen grinds, up to
//    0.70 for vague/hidden misc trophies), widened a little with the horizon
//    and with the residual noise of the trend fit, and widened more when the
//    history was flat (a flat history hides the next refresh jump). A spread
//    that is honest — not the widest possible — is what the pinball loss pays.
//
// 4. Determinism is a hard gate, so there is no clock, no randomness, no
//    iteration order that depends on anything but the input file. Every output
//    number is rounded to 4 decimals, quantiles are forced non-decreasing, and
//    the program never throws on a missing rate, an empty history, or an
//    unexpected trophy count: it always writes a complete, valid forecast.
//
// Usage (local): mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
// The grader leaves both variables unset and the program reads /input, writes /output.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];

/** One trophy as the case file lists it. */
interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}

/** One house read of the Exophase page. */
interface Read {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}

/** One case file. */
interface CaseFile {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { name: string; platform: string; release_date: string; source: string };
  trophies: Trophy[];
  history: Read[];
  notes?: string;
}

/** A trophy's behavioural class, decided from its own description and tier. */
interface TrophyClass {
  /** Prior drift slope, logit units per unit of ln(1 + hours since release). */
  priorSlope: number;
  /** Prior logit spread (the 1-sigma width of the forecast band). */
  priorSpread: number;
  /**
   * Where a trophy of this class plausibly sits at the target when its last
   * read is at the 2-decimal floor (0.00-0.05%): the level prior. A rate at
   * the floor can only go up, and a logit drift from -9.2 cannot express
   * that, so a floored trophy's median is blended toward this level instead
   * of drifting from the floor. It is deliberately modest — a trophy nobody
   * has earned yet is still a trophy almost nobody earns in week one.
   */
  floorMedian: number;
  label: string;
}

/** Clamp a percentage into the range the scorer's logit can take. */
function clampPct(x: number): number {
  if (!Number.isFinite(x)) return 0.01;
  return Math.min(99.99, Math.max(0.01, x));
}

/** logit of a percentage, after the scorer's clamp. */
function logit(pct: number): number {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
}

/** Inverse logit: back to a percentage. */
function invLogit(l: number): number {
  return 100 / (1 + Math.exp(-l));
}

/** Round to 4 decimals, the precision the house baseline uses. */
function round4(x: number): number {
  return Math.round(x * 10000) / 10000;
}

/**
 * Standard normal quantile, Acklam's rational approximation (absolute error
 * under 1.2e-9, far tighter than the 4-decimal output). Written out here so
 * the program works for any levels list, not just the contest's five.
 */
function normalQuantile(p: number): number {
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
  const pLow = 0.02425;
  const pHigh = 1 - pLow;
  if (p < pLow) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > pHigh) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  const q = p - 0.5;
  const r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

/** Hours between two ISO timestamps. */
function hoursBetween(fromIso: string, toIso: string): number {
  return (Date.parse(toIso) - Date.parse(fromIso)) / 3600000;
}

/**
 * Decide a trophy's class from its tier and description. The description is
 * the trophy's own requirement text ("Complete Chapter 1.", "Complete 250
 * Stunts.", "WAiNK It at 100 locations."), so the rules below read intent,
 * not identity: they work on any trophy list with the same style of text.
 */
function classify(t: Trophy): TrophyClass {
  const tier = (t.tier ?? '').toLowerCase();
  const desc = (t.description ?? '').toLowerCase();
  const name = (t.name ?? '').toLowerCase();

  // The platinum pops only once every other trophy has: nobody earns it in
  // week one, and the growing denominator drags its rate down.
  if (tier === 'platinum') {
    return { priorSlope: -0.15, priorSpread: 0.45, floorMedian: 0.12, label: 'platinum' };
  }
  // 100% completion is a ~200-hour grind; same story, slightly slower.
  if (desc.includes('100% completion')) {
    return { priorSlope: -0.2, priorSpread: 0.4, floorMedian: 0.06, label: 'completion' };
  }

  // Story chapters: the earlier the chapter, the more of the tracked base
  // reaches it within the week, so the faster its rate climbs.
  const chapter = desc.match(/chapter\s+(\d+)/);
  if (chapter) {
    const n = parseInt(chapter[1], 10);
    if (n <= 1) return { priorSlope: 0.35, priorSpread: 0.55, floorMedian: 2.0, label: 'story-ch1' };
    if (n === 2) return { priorSlope: 0.25, priorSpread: 0.55, floorMedian: 1.0, label: 'story-ch2' };
    if (n === 3) return { priorSlope: 0.15, priorSpread: 0.55, floorMedian: 0.6, label: 'story-ch3' };
    return { priorSlope: 0.05, priorSpread: 0.55, floorMedian: 0.4, label: 'story-ch4plus' };
  }

  // The final mission (main story) and the last mission of a stranger strand.
  if (desc.includes('final mission')) {
    if (desc.includes(' for ')) {
      return { priorSlope: 0.05, priorSpread: 0.55, floorMedian: 0.2, label: 'stranger-finale' };
    }
    return { priorSlope: -0.1, priorSpread: 0.55, floorMedian: 0.25, label: 'story-finale' };
  }

  // The opening mission is the very first thing every player does.
  if (desc.includes('gone fishing')) {
    return { priorSlope: 0.1, priorSpread: 0.25, floorMedian: 99.9, label: 'opening' };
  }

  // Money sinks take in-game cash and time; they move, but slowly. Checked
  // BEFORE the grind rule below: a "$250,000" amount is a price tag, not a
  // count of things to do, and must not be read as a 250-item grind.
  if (
    desc.includes('laundering') ||
    desc.includes('spend $') ||
    desc.includes('garage propert') ||
    desc.includes('register 10 vehicles')
  ) {
    return { priorSlope: 0.05, priorSpread: 0.55, floorMedian: 0.15, label: 'money' };
  }

  // Long counted grinds: a big number in the requirement (250 stunts, 100
  // WAiNK scans, 100 vehicles, 50 challenges) means almost nobody finishes
  // in week one. "Every location / every spot / all" is the same shape.
  const counted = desc.match(/(\d+)\s*(stunts|locations|vehicles|challenges|races|spots|sets|animals)/);
  const bigCount = desc.match(/(\d{2,})\b/);
  if (
    desc.includes('gold medal in all') ||
    desc.includes('every location') ||
    desc.includes('every spot') ||
    desc.includes('explore all') ||
    desc.includes('card sets') ||
    (counted && parseInt(counted[1], 10) >= 20) ||
    (bigCount && parseInt(bigCount[1], 10) >= 50)
  ) {
    return { priorSlope: -0.25, priorSpread: 0.4, floorMedian: 0.15, label: 'grind' };
  }

  // Quick one-off activities: one arcade game, one fishing challenge, trying
  // each activity once. Easy, so they climb at a decent clip.
  if (
    desc.includes('arcade game') ||
    desc.includes('fishing challenge') ||
    desc.includes('each type of activity once') ||
    desc.includes('top fun drone race')
  ) {
    return { priorSlope: 0.25, priorSpread: 0.55, floorMedian: 0.5, label: 'easy-activity' };
  }

  // Named story missions ("Complete Breakfast Run.") — mid-story beats.
  if (desc.startsWith('complete ') && (t.secret || tier === 'bronze')) {
    return { priorSlope: 0.15, priorSpread: 0.55, floorMedian: 0.5, label: 'story-mission' };
  }

  // Everything else: crime one-offs and misc moments. They happen while
  // playing normally, so they climb a little; vague or hidden ones get the
  // widest spread because nobody can say exactly when they pop.
  const vague = t.secret && desc.length < 45;
  return {
    priorSlope: 0.1,
    priorSpread: vague ? 0.7 : 0.6,
    floorMedian: vague ? 0.15 : 0.2,
    label: vague ? 'misc-hidden' : 'misc',
  };
}

/**
 * Ordinary least squares slope of y on x, plus the residual spread. Returns
 * null when the points cannot define a line (fewer than 3 points, or x has
 * no variance — the flat-history case the practice cases are built on).
 */
function fitSlope(
  xs: number[],
  ys: number[],
): { slope: number; resid: number } | null {
  if (xs.length < 3) return null;
  const n = xs.length;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += xs[i];
    my += ys[i];
  }
  mx /= n;
  my /= n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    sxx += dx * dx;
    sxy += dx * (ys[i] - my);
  }
  if (sxx < 1e-9) return null;
  const slope = sxy / sxx;
  let ss = 0;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (my + slope * (xs[i] - mx));
    ss += r * r;
  }
  return { slope, resid: Math.sqrt(ss / n) };
}

/** Forecast one case file into its five quantiles per trophy. */
function forecast(c: CaseFile): Array<{ id: number; q: number[] }> {
  const levels = c.levels && c.levels.length === 5 ? c.levels : LEVELS;
  const zScores = levels.map((lv) => normalQuantile(Math.min(0.999, Math.max(0.001, lv))));

  // Release is midnight UTC on the game's release date; if the case does not
  // say, fall back to the competition's launch (2026-11-19), which the kit
  // documents. Time is measured as ln(1 + hours since release): early hours
  // matter most, later hours flatten out, which matches how a player base
  // actually progresses through a game.
  const releaseIso = c.game && c.game.release_date
    ? `${c.game.release_date}T00:00:00Z`
    : '2026-11-19T00:00:00Z';
  const horizonDays = Math.max(0, c.horizon_minutes / 1440);
  const xTarget = Math.log(1 + Math.max(0, hoursBetween(releaseIso, c.target)));

  const history = Array.isArray(c.history) ? c.history : [];
  const last = history.length > 0 ? history[history.length - 1] : null;
  const lastAt = last ? last.at : c.at;
  const xLast = Math.log(1 + Math.max(0, hoursBetween(releaseIso, lastAt)));

  // Recent window for the trend fit: the last 24 h of reads. Older reads
  // carry the launch-night collapse (denominator explosion), which must not
  // be extrapolated forward — that is the house curve baseline's trap.
  const windowStartMs = Date.parse(lastAt) - 24 * 3600000;

  return c.trophies.map((trophy, i) => {
    const cls = classify(trophy);

    // Anchor: the trophy's rate in the most recent read that has it. Walk
    // backwards over nulls; if no read ever had it, start from 0.5%.
    let anchor: number | null = null;
    for (let h = history.length - 1; h >= 0; h--) {
      const r = history[h].rates ? history[h].rates[i] : null;
      if (typeof r === 'number' && Number.isFinite(r)) {
        anchor = r;
        break;
      }
    }
    const anchorRate = anchor === null ? 0.5 : anchor;
    const anchorLogit = logit(anchorRate);

    // Recent trend: logit rate against ln(1 + hours since release), over the
    // last 24 h of reads that carry this trophy.
    const xs: number[] = [];
    const ys: number[] = [];
    for (const read of history) {
      if (Date.parse(read.at) < windowStartMs) continue;
      const r = read.rates ? read.rates[i] : null;
      if (typeof r !== 'number' || !Number.isFinite(r)) continue;
      xs.push(Math.log(1 + Math.max(0, hoursBetween(releaseIso, read.at))));
      ys.push(logit(r));
    }
    const fit = fitSlope(xs, ys);
    const distinctRates = new Set(ys.map((y) => y.toFixed(6))).size;

    // Shrink the fitted slope toward the class prior. The data earns weight
    // only when it actually varies: with a flat history the prior decides.
    // At most 6 units of data weight against 3 of prior, so a real trend can
    // win two-to-one but a noisy one cannot drag the forecast away.
    let slope = cls.priorSlope;
    let resid = 0;
    if (fit !== null && distinctRates >= 2) {
      const wData = Math.min(distinctRates, 6);
      const wPrior = 3;
      slope = (wData * fit.slope + wPrior * cls.priorSlope) / (wData + wPrior);
      resid = fit.resid;
    }

    // Median: anchor plus drift over the horizon, capped so no extrapolation
    // runs away — rates cannot move more than 2.5 logit units in one jump,
    // whatever the slope says. But a trophy sitting at the 2-decimal floor
    // (0.00-0.05%) gets a different treatment: its anchor logit is about
    // -9, a number that says "nobody has earned this yet", and drifting from
    // there cannot express the week-one truth, which is that SOME players
    // will earn it by the target. So a floored trophy's median is blended
    // halfway toward its class's floor level, with the drift halved — the
    // level prior, not the slope, does the work at the floor — and its
    // spread is widened, because exactly when a trophy emerges from zero is
    // the most uncertain thing in this forecast.
    const dx = xTarget - xLast;
    let drift = slope * dx;
    drift = Math.min(2.5, Math.max(-2.5, drift));
    let medianLogit: number;
    if (anchorRate <= 0.05) {
      medianLogit = 0.5 * anchorLogit + 0.5 * logit(cls.floorMedian) + 0.5 * drift;
    } else {
      medianLogit = anchorLogit + drift;
    }
    medianLogit = Math.min(logit(99.99), Math.max(logit(0.01), medianLogit));

    // Spread: the class prior, widened a little with the horizon (a random
    // walk's share), widened by the fit's own residual noise, and widened
    // more when the history was flat — a flat history hides the next
    // Exophase refresh jump. Capped at 1.2 logits so the band never becomes
    // the house baseline's 1.06-by-default mush.
    let spread = Math.sqrt(
      cls.priorSpread * cls.priorSpread +
        0.0225 * horizonDays +
        0.25 * resid * resid,
    );
    if (distinctRates < 2) spread *= 1.2;
    if (anchorRate <= 0.05) spread *= 1.3;
    spread = Math.min(1.2, Math.max(0.15, spread));

    // Quantiles, rounded, forced non-decreasing (rounding can only tie).
    const q: number[] = [];
    let prev = -Infinity;
    for (const z of zScores) {
      let v = round4(clampPct(invLogit(medianLogit + z * spread)));
      if (v < prev) v = prev;
      q.push(v);
      prev = v;
    }
    return { id: trophy.id, q };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  const trophies = forecast(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
