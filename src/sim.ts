// Play-by-play game simulator and Monte Carlo win probability.
// Pure functions, no React or DOM access. All times are in whole seconds.
//
// The model is deliberately simple: each call has a fixed outcome distribution, both teams follow
// a rule-of-thumb policy after your decision, and a tie at 0:00 is a coin flip in overtime.

import { NFL_RULES, PLAY_TYPES, simulateRunOut, type Down } from "./engine";
import { formatClock } from "./time";
import type { GameState, GuideSection, PlayCall, Team } from "@situational-football/schemas";

// The game state and call types come from the schemas package, which validates them too.
export type { GameState, PlayCall, Team };

export const PLAY_CALLS: Readonly<Record<PlayCall, string>> = Object.freeze({
  timeout: "Timeout",
  run: "Run",
  "pass-sideline": "Pass to the sideline",
  "pass-middle": "Pass over the middle",
  "pass-deep": "Deep pass",
  "pass-play-action": "Play-action pass",
  "pass-screen": "Screen pass",
  spike: "Spike",
  kneel: "Kneel",
  fg: "Field goal",
  punt: "Punt",
});

export interface SimConfig {
  /** Seconds from the end of an in-bounds play to a no-huddle snap. */
  hurryUp: number;
  /** Seconds from the end of an in-bounds play to a spike. */
  spikeLineup: number;
  /** Seconds to rush the field-goal unit on with the clock running. */
  fgRush: number;
}

export const DEFAULT_SIM: SimConfig = Object.freeze({ hurryUp: 18, spikeLineup: 10, fgRush: 15 });

const WARNING = NFL_RULES.twoMinuteWarning;
const BURN = NFL_RULES.playClock - 1; // snap with 1 second on the play clock
/** The longest field-goal try: from the opponent's 47, a 65-yarder. */
export const FG_MAX_YTG = 47;
/** Punting is an option from 40 or more yards out. */
export const PUNT_MIN_YTG = 40;
/** On the last play, a team that needs only a field goal kicks from this close or closer (a 65-yarder). */
export const LAST_PLAY_FG_YTG = 47;
/** A defense that's behind uses its timeouts after in-bounds plays in the last 4 minutes. */
export const DEFENSE_TIMEOUT_WINDOW = 240;
/** Fewest seconds from the end of an in-bounds play to the next snap, even when rushing. */
export const MIN_SNAP = 8;
/**
 * A tied team more than 60 yards out kneels for overtime with this many seconds left or fewer. Real teams
 * in that spot play on with 0:30 to 1:00 left (0 of 74 kneeled, 2022-26) and win about 59%, against 50% for overtime.
 */
export const TIED_KNEEL_CLOCK = 30;
/** A field-goal try is this much longer than the yards to the goal line: the end zone (10) plus the hold (8). */
export const FG_EXTRA_YARDS = 18;
/** The length of a field-goal try from this many yards out. */
export const fgDistance = (ytg: number) => ytg + FG_EXTRA_YARDS;
/**
 * A team that needs a touchdown throws at the end zone from 40 yards or closer with this many seconds
 * left or fewer. Earlier than this, it works the ball; later, short throws use up downs before the clock.
 */
export const END_ZONE_SHOT_CLOCK = 45;

// ---------------------------------------------------------------------------
// Random numbers

export type Rng = () => number;

/** Small, fast seeded generator, so the same situation always gives the same estimates. */
export function seededRng(seed: number): Rng {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const gauss = (r: Rng) => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
const expo = (r: Rng, mean: number) => -mean * Math.log(1 - r());

// ---------------------------------------------------------------------------
// Rules helpers

const other = (t: Team): Team => (t === "you" ? "opp" : "you");
const marginFor = (g: GameState, team: Team) => (team === "you" ? g.margin : -g.margin);
const who = (team: Team) => (team === "you" ? "You" : "They");

/**
 * Field-goal make rates by distance, measured from 2022-2026 NFL regular seasons (about 3,800 tries, in
 * 5-yard bands, at each band's middle). Kickers have got so good from long range that the rate is nearly
 * flat from the high 40s to the high 50s. Re-measure with scripts/calibrate/kicks.ts.
 */
export const FG_MAKE_BY_DISTANCE: readonly (readonly [number, number])[] = [
  [20, 0.995], [32, 0.967], [37, 0.91], [42, 0.839], [47, 0.764], [52, 0.732], [57, 0.648], [62, 0.481], [67, 0.3],
];

/** Field-goal make chance for a try from this spot (distance = yards to the goal line + 18). */
export function fgMakeChance(ytg: number): number {
  const distance = fgDistance(ytg);
  const first = FG_MAKE_BY_DISTANCE[0];
  const last = FG_MAKE_BY_DISTANCE.at(-1);
  if (first === undefined || last === undefined) return 0;
  if (distance <= first[0]) return first[1];
  if (distance >= last[0]) return last[1];
  for (let i = 1; i < FG_MAKE_BY_DISTANCE.length; i++) {
    const hi = FG_MAKE_BY_DISTANCE[i];
    const lo = FG_MAKE_BY_DISTANCE[i - 1];
    if (hi !== undefined && lo !== undefined && distance <= hi[0]) {
      return lo[1] + ((hi[1] - lo[1]) * (distance - lo[0])) / (hi[0] - lo[0]);
    }
  }
  return last[1];
}

/** Down 3 this close to the goal line, a team plays for the touchdown, with the tying kick in hand. */
export const TOUCHDOWN_FIRST_YTG = 10;

/**
 * Whether the team with the ball is playing for a field goal: one ties or wins, and they're in comfortable
 * range. Then it drains the clock and kicks last. Not when it's down 3 inside the 10, though: there a field
 * goal only ties and the end zone is close, so it plays for the touchdown and kicks only if it has to
 * (better by 0.4 points on average over 2,970 down-3 situations).
 */
function fgPlay(g: GameState): boolean {
  const m = marginFor(g, g.poss);
  return m <= 0 && m >= -3 && g.ytg <= 35 && !(m === -3 && g.ytg <= TOUCHDOWN_FIRST_YTG);
}

/**
 * How long a team that's tied or behind needs to finish its drive at a hurry-up pace: a base, plus a rate
 * per yard still to go, to the end zone when it needs a touchdown or to the `fgTarget` yard line when a
 * field goal is enough. With more time than this, stopping the clock only saves time for the other team's
 * answer, so the team keeps its timeouts and lets the spare time run; with less, it stops the clock.
 *
 * Fitted by testing rules over 4,608 tied and trailing situations (every score from tied to down 8, 0:20
 * to 3:20 left, the whole field, with and without timeouts). Against stopping the clock whenever it runs
 * under 2:00, a team that needs a touchdown wins 24.2% instead of 22.6% on average, and up to 20 points
 * more near the goal line; under the old rule, having a timeout made a team worse off than having none
 * in 55 of 600 situations.
 */
export const DRIVE_TIME = { base: 25, perYard: 1.5, fgTarget: 20 } as const;

export function driveTimeNeeded(g: GameState): number {
  const yards = marginFor(g, g.poss) < -3 ? g.ytg : Math.max(0, g.ytg - DRIVE_TIME.fgTarget);
  return DRIVE_TIME.base + DRIVE_TIME.perYard * yards;
}

/** A team that's tied or behind, isn't draining for a kick, and has more time than its drive needs. */
export function hasTimeToSpare(g: GameState): boolean {
  return marginFor(g, g.poss) <= 0 && !fgPlay(g) && g.clock > driveTimeNeeded(g);
}

/** The clock management above applies from this many seconds left. */
const SPARE_TIME_WINDOW = 150;

/**
 * Tempo is automatic. A team that's ahead, or that only needs a field goal and is in range, drains
 * the play clock. A trailing team draining for a kick snaps early enough to stop the clock and kick.
 * A team that's tied or behind late with time to spare lets the spare time run. Everyone else hurries.
 */
/**
 * Seconds from the end of the last play to the snap when the clock is running, for a call: a leading team
 * drains the play clock, a trailing one hurries, a spike or a kick takes its own setup time.
 */
export function snapDelay(g: GameState, call: PlayCall, cfg: SimConfig = DEFAULT_SIM, tempo: Tempo = "auto"): number {
  return preSnapBurn(g, call, cfg, tempo);
}

/**
 * The pace of the next snap with the clock running. "auto" is the model's own choice (below); "hurry" snaps
 * as fast as the offense can line up; "drain" lets the play clock run down and snaps with 1 second left.
 * The trainer lets you pick it for the snap you're calling. The plays after it go back to "auto".
 */
export type Tempo = "auto" | "hurry" | "drain";
export const TEMPOS: readonly Tempo[] = ["auto", "hurry", "drain"];

function preSnapBurn(g: GameState, call: PlayCall, cfg: SimConfig, tempo: Tempo = "auto"): number {
  if (tempo === "drain") return BURN;
  if (tempo === "hurry") {
    if (call === "fg" || call === "punt") return cfg.fgRush;
    if (call === "spike") return cfg.spikeLineup;
    return g.clock <= LATE_HURRY_WINDOW ? Math.min(cfg.hurryUp, LATE_HURRY_UP) : cfg.hurryUp;
  }
  // A kneel is for running out the clock, so it lets the play clock run, ahead or tied. The exception is a
  // kneel to set up a field goal: that's paced like a run there (below), so there's still time for the kick.
  if (marginFor(g, g.poss) > 0 || (call === "kneel" && !fgPlay(g))) return BURN;
  // A kick that ties or wins: let the clock run down to 0:03 before it, or as far as one play clock allows,
  // so it leaves the other team as little as possible. It still takes `fgRush` seconds to get the unit on.
  if (call === "fg" && marginFor(g, g.poss) >= -3) return Math.min(BURN, Math.max(cfg.fgRush, g.clock - FG_TIMEOUT_AT));
  if (call === "fg" || call === "punt") return cfg.fgRush;
  if (call === "spike") return cfg.spikeLineup;
  if (fgPlay(g)) return Math.min(BURN, Math.max(MIN_SNAP, fgDrainBurn(g, cfg)));
  const hurry = g.clock <= LATE_HURRY_WINDOW ? Math.min(cfg.hurryUp, LATE_HURRY_UP) : cfg.hurryUp;
  // Late with more time than the drive needs: let the spare time run before the snap, so a score leaves
  // the other team as little as possible.
  if (g.clock <= SPARE_TIME_WINDOW) {
    const spare = Math.floor(g.clock - driveTimeNeeded(g));
    if (spare > hurry) return Math.min(BURN, spare);
  }
  return hurry;
}

/**
 * Clock a team draining for a field goal snaps its last play at: enough for a quick snap and a play,
 * then a timeout or a rushed field-goal unit.
 */
const kickFloor = (g: GameState, cfg: SimConfig) => (g.timeouts[g.poss] > 0 ? 20 : cfg.fgRush + 15);

/**
 * A defense that's ahead or tied still stops the clock when the team with the ball is draining for a
 * field goal that would beat it (tied or down 1-2), to save time for its own answer. Tested against never
 * doing so and against keeping one timeout back, over 240 late situations: spending them all helped the
 * defense most (3.5-5.7 points on average). When the kick would only tie (down 3), it saves them: there,
 * stopping the clock cost the defense 1.6 points on average.
 */
const ANSWER_TIME_WORTH_SAVING = 20;
function savesTimeForAnswer(g: GameState, def: Team): boolean {
  if (g.timeouts[def] === 0) return false;
  const m = marginFor(g, g.poss);
  if (fgPlay(g) && m > -3) {
    // Not once an offense with no timeout has too little time left to get its kicking unit on: there the
    // stoppage is exactly what the offense needs (it saves it a spike, or the game), and helps no one else.
    if (g.timeouts[g.poss] === 0 && g.clock <= DEFAULT_SIM.fgRush + 1) return false;
    // And only when the stoppages can leave enough time to answer with (20 seconds or more). The offense
    // lets the play clock run before every snap the defense can't stop, so with too few timeouts for that,
    // stopping the clock saves nothing and hands the offense extra plays. (Over 3,780 situations, the offense
    // wins 70.3% against this rule, 70.9% against a defense that always stops the clock, and 71.2% against
    // one that never does.)
    const snaps = 4 - g.down;
    const unstopped = Math.max(0, snaps + 1 - g.timeouts[def]);
    return g.clock > unstopped * BURN + snaps * DRAIN_PLAY_SECONDS + ANSWER_TIME_WORTH_SAVING;
  }
  // The same against a team that needs a touchdown, is inside the red zone under two minutes, and has more
  // time than it needs: it would let that time run, so the defense (ahead by a score) stops the clock to
  // keep it for an answer. Over 3,456 such situations this cut the offense's win probability from 31.4% to
  // 30.5%; holding the timeouts, or using them outside the red zone, did no better.
  return m < -3 && m >= -8 && g.ytg <= RED_ZONE && g.clock <= WARNING && g.clock > driveTimeNeeded(g);
}

/**
 * Under a minute, teams snap faster than the usual hurry-up. In 2022-2026 NFL play-by-play, a trailing team's
 * next snap after an in-bounds completion came 18.8 seconds after the one before with 0:20-1:00 left (23.0 with
 * 1:00-2:00). Less the 5.2-second play, that's about 13 seconds from the end of the play to the snap.
 */
export const LATE_HURRY_WINDOW = 60;
export const LATE_HURRY_UP = 13;

/**
 * A trailing team with no timeouts late has only the sideline to stop the clock, and its receivers work to
 * get out. In 2022-2026 play-by-play, 50% of short sideline catches went out of bounds for a trailing team
 * with no timeouts and 1:30 or less left, against about a third with timeouts and about a fifth earlier.
 */
export const NO_TIMEOUT_SIDELINE_OOB = 0.5;
export const MUST_GET_OUT_WINDOW = 90;
const mustGetOut = (g: GameState, clock: number) =>
  g.timeouts[g.poss] === 0 && marginFor(g, g.poss) < 0 && clock <= MUST_GET_OUT_WINDOW;

/** At the kick, a team draining for a field goal lets the clock run to this before its last timeout. */
export const FG_TIMEOUT_AT = 3;

/** Seconds a run takes, for planning a drain: the trailing mode's rule-of-thumb play length. */
const DRAIN_PLAY_SECONDS = 6;

/**
 * Seconds to let run before this snap when draining for a field goal. The spare clock is spread over the
 * snaps left before the 4th-down kick, so the team keeps running plays (and gaining yards for a shorter
 * kick) instead of burning it all before one snap, and still snaps its last play at the kick floor.
 */
function fgDrainBurn(g: GameState, cfg: SimConfig): number {
  const snaps = Math.max(1, 4 - g.down);
  return (g.clock - kickFloor(g, cfg) - (snaps - 1) * DRAIN_PLAY_SECONDS) / snaps;
}

/** A list with at least one element, so reading its first element can't be undefined. */
export type NonEmpty<T> = readonly [T, ...T[]];

export function availableCalls(g: GameState): readonly PlayCall[] {
  if (g.over) return [];
  const calls: PlayCall[] = [];
  if (g.timeouts[g.poss] > 0) calls.push("timeout");
  calls.push("run", "pass-screen", "pass-sideline", "pass-middle", "pass-play-action", "pass-deep");
  if (g.down <= 3) calls.push("spike");
  // Kneeling gives up a down and a yard. It helps a team that's ahead, one that's tied and settling for
  // overtime, and one in field-goal range that only needs the kick, draining the clock without a fumble.
  const m = marginFor(g, g.poss);
  if (m >= 0 || (m >= -3 && g.ytg <= FG_MAX_YTG)) calls.push("kneel");
  if (g.ytg <= FG_MAX_YTG) calls.push("fg");
  // Punts only on 4th down: teams don't quick-kick late in games.
  if (g.ytg >= PUNT_MIN_YTG && g.down === 4) calls.push("punt");
  return calls;
}

// ---------------------------------------------------------------------------
// One decision

export interface StepResult {
  state: GameState;
  /** What happened, for the play log. */
  text: string;
  /** Set when the team with the ball lost it or scored, e.g. "Touchdown". */
  driveEnded?: string;
  /** Set when the clock reached the two-minute warning before the snap: it stopped at 2:00, and no play ran. */
  twoMinuteWarning?: true;
}

/** What happened on a scrimmage play. Scenarios pass one to `step` to replay a specific result. */
export interface PlayOutcome {
  yards: number;
  seconds: number;
  outOfBounds: boolean;
  incomplete: boolean;
  turnover?: "interception" | "fumble";
  sack?: boolean;
}

/**
 * How the defense is playing:
 * - "prevent": ahead late, it protects the deep field (the last 4:00, outside the red zone), except on 3rd or
 *   4th and 6 or fewer, when giving up a short catch loses the down: there it guards the sticks instead.
 * - "obvious": everyone knows a pass is coming: 3rd or 4th & 7+, or the offense trailing or tied by a score
 *   in the last 5:00. More sacks, fewer completions, and play-action fools fewer defenders.
 * - "normal": anything else.
 */
export type Coverage = "normal" | "obvious" | "prevent";

/** A defense that's ahead plays prevent in the last 4 minutes. */
export const PREVENT_WINDOW = 240;
/** The offense is obviously passing when trailing or tied by a score in the last 5 minutes… */
export const OBVIOUS_WINDOW = 300;
/** …or on 3rd or 4th down with this many yards or more to go. */
export const OBVIOUS_TO_GO = 7;

/** Inside the opponent's 20 the field is compressed: less room to defend, and to throw into. */
export const RED_ZONE = 20;

export function coverageFor(g: GameState, clock: number): Coverage {
  const m = marginFor(g, g.poss);
  // Prevent protects the deep field, and in the red zone there isn't one. On 3rd or 4th and short, a defense
  // can't concede the underneath catch prevent gives up, so it plays the sticks: an obvious passing down.
  const sticks = g.down >= 3 && g.toGo < OBVIOUS_TO_GO;
  if (m < 0 && clock <= PREVENT_WINDOW && g.ytg > RED_ZONE && !sticks) return "prevent";
  if ((g.down >= 3 && g.toGo >= OBVIOUS_TO_GO) || (m <= 0 && m >= -8 && clock <= OBVIOUS_WINDOW)) return "obvious";
  return "normal";
}

/**
 * How much of a pass's normal completion rate survives at this distance from the goal line:
 * 1 outside the red zone, shrinking to 0.6 at the 1 as coverage tightens.
 */
export function compression(ytg: number): number {
  return ytg >= RED_ZONE ? 1 : 0.6 + (0.4 * Math.max(0, ytg - 1)) / (RED_ZONE - 1);
}

export type PassCall = "pass-sideline" | "pass-middle" | "pass-deep" | "pass-play-action" | "pass-screen";

interface PassOdds {
  /** Chance of a sack, per snap. */
  readonly sack: number;
  /** Chance of an interception, per snap. */
  readonly interception: number;
  /** Chance of a completion, per snap (sacks count as snaps, so this is below the per-throw rate). */
  readonly complete: number;
  /** Completions gain at least this many yards… */
  readonly minYards: number;
  /** …plus an exponential with this mean. */
  readonly extraYards: number;
  /** Share of completions that end out of bounds. */
  readonly outOfBounds: number;
}

/**
 * Pass odds by situation, fitted to the 2025 and 2026 (weeks 1-8) NFL regular seasons: about 28,800
 * dropbacks charted by FTN, so play-action and screens are known. Re-fit with scripts/calibrate/passes.ts.
 *
 * - Completion and interception rates are per throw in the data, converted to per snap here.
 * - Yards per catch: the minimum is about the 10th percentile, and the exponential's mean makes up the average.
 * - Sacks can't be tied to a target (a sacked passer never threw), so dropback passes share the measured
 *   rate: 6% normally, 10% in obvious passing situations, deep throws a little more as they take longer.
 * - Prevent has few plays (50 to 513 per kind of pass), so its numbers are pulled toward the normal ones in
 *   proportion to how few plays back them.
 * - Play-action in obvious passing and prevent keeps about half its usual edge over a dropback: the
 *   linebackers aren't fooled when a pass is expected. Real play-action passes in those spots are too few
 *   (118 and 50) to fit directly; they hint at more sacks (8.5% and 14%), which the 9% here reflects.
 */
export const PASS_ODDS: Readonly<Record<Coverage, Readonly<Record<PassCall, PassOdds>>>> = Object.freeze({
  normal: {
    // 66% complete, 9 yards a catch, 26% of catches out of bounds.
    "pass-sideline": { sack: 0.06, interception: 0.017, complete: 0.624, minYards: 3, extraYards: 6, outOfBounds: 0.26 },
    // 70% complete, 11 yards a catch, almost never out of bounds.
    "pass-middle": { sack: 0.06, interception: 0.025, complete: 0.653, minYards: 4, extraYards: 7, outOfBounds: 0.03 },
    // 36% complete for 32 yards a catch, 4% intercepted.
    "pass-deep": { sack: 0.08, interception: 0.037, complete: 0.328, minYards: 20, extraYards: 12, outOfBounds: 0.24 },
    // A run fake pulls the linebackers up: 63% complete for 13 yards a catch.
    "pass-play-action": { sack: 0.067, interception: 0.018, complete: 0.589, minYards: 2, extraYards: 11, outOfBounds: 0.25 },
    // A quick throw behind blockers: 86% complete, almost never sacked or picked, often stopped short (6 a catch).
    "pass-screen": { sack: 0.005, interception: 0.006, complete: 0.86, minYards: -2, extraYards: 8, outOfBounds: 0.14 },
  },
  obvious: {
    "pass-sideline": { sack: 0.1, interception: 0.02, complete: 0.537, minYards: 4, extraYards: 6, outOfBounds: 0.21 },
    "pass-middle": { sack: 0.1, interception: 0.035, complete: 0.597, minYards: 5, extraYards: 7, outOfBounds: 0.03 },
    "pass-deep": { sack: 0.12, interception: 0.054, complete: 0.295, minYards: 20, extraYards: 14, outOfBounds: 0.14 },
    "pass-play-action": { sack: 0.09, interception: 0.025, complete: 0.593, minYards: 3, extraYards: 9.5, outOfBounds: 0.2 },
    "pass-screen": { sack: 0.005, interception: 0.003, complete: 0.84, minYards: -1, extraYards: 8, outOfBounds: 0.08 },
  },
  prevent: {
    // Throws underneath complete as usual and gain a little more; catches near the sideline get out of bounds.
    "pass-sideline": { sack: 0.06, interception: 0.024, complete: 0.629, minYards: 3, extraYards: 7, outOfBounds: 0.32 },
    "pass-middle": { sack: 0.06, interception: 0.017, complete: 0.654, minYards: 4, extraYards: 7.5, outOfBounds: 0.03 },
    // The deep field is what prevent protects: 31% complete, and 8% intercepted on real late drives (2022-26).
    "pass-deep": { sack: 0.08, interception: 0.075, complete: 0.276, minYards: 20, extraYards: 11.5, outOfBounds: 0.24 },
    "pass-play-action": { sack: 0.09, interception: 0.02, complete: 0.593, minYards: 3, extraYards: 10.5, outOfBounds: 0.2 },
    "pass-screen": { sack: 0.005, interception: 0.005, complete: 0.864, minYards: -2, extraYards: 8.3, outOfBounds: 0.15 },
  },
});

/**
 * On 3rd and 4th down, defenses play to stop the conversion: they guard the first-down marker and give up
 * catches short of it. Each kind of pass completes this much less often, by yards to go, fitted to real
 * conversion rates on 3rd/4th down outside the red zone (2022-26; by kind of pass from FTN's 2025-26 charting):
 * - short (1-3 to go): passes convert about 55%;
 * - medium (4-6): sideline 51%, middle 58%, deep 35%, play-action 49%, screens 32% of throws;
 * - long (7+): sideline 36%, middle 41%, deep 34%, play-action 41%, screens 31% on 7-10.
 * Re-measure with scripts/calibrate/passes.ts.
 */
export type SticksBand = "short" | "medium" | "long";

export const STICKS_PASS: Readonly<Record<SticksBand, Readonly<Partial<Record<PassCall, number>>>>> = Object.freeze({
  short: { "pass-sideline": 0.86, "pass-middle": 0.86, "pass-deep": 0.86, "pass-play-action": 0.86, "pass-screen": 0.86 },
  medium: { "pass-sideline": 0.94, "pass-middle": 0.87, "pass-play-action": 0.92, "pass-screen": 0.81 },
  long: { "pass-middle": 0.86, "pass-deep": 0.85 },
});

/** On 3rd or 4th down, how far there is to go: short (1-3), medium (4-6) or long (7+). */
export function sticksBand(g: GameState): SticksBand | null {
  if (g.down < 3) return null;
  return g.toGo <= 3 ? "short" : g.toGo < OBVIOUS_TO_GO ? "medium" : "long";
}

/** Completion rate of a throw into the end zone from the 20, before compression. */
const END_ZONE_SHOT = 0.25;

/** Pass odds for a throw from this spot: completions get harder in the red zone. */
export function passOdds(coverage: Coverage, call: PassCall, ytg: number): PassOdds {
  const odds = PASS_ODDS[coverage][call];
  if (ytg > RED_ZONE) return odds;
  const c = compression(ytg);
  // From the 20 in there's no deep field: a "deep" pass is a throw into the end zone, complete
  // about 25% of the time at the 20 and less closer in.
  return call === "pass-deep" ? { ...odds, complete: END_ZONE_SHOT * c, interception: 0.06 } : { ...odds, complete: odds.complete * c };
}

/** Average run gain by coverage: prevent gives up a little more on the ground. */
const RUN_MEAN: Readonly<Record<Coverage, number>> = Object.freeze({ normal: 3.6, obvious: 3.6, prevent: 4.3 });

/**
 * A run the defense expects: the offense is ahead in the last five minutes, so the defense loads up to stop
 * it. Fitted to 2022-2026 1st- and 2nd-down runs by a team ahead by 1-8 in the last 5:00, in the open field:
 * 4.0 yards a carry, 22% stuffed at or behind the line and 9% going for 10 or more (against 4.7, 16% and 11.5%
 * earlier in close games). Near the goal line the packed-line shape applies as usual.
 */
const EXPECTED_RUN: RunShape = Object.freeze({ mean: 3.2, sd: 3.7, breakaway: 0.055 });
/** A run is expected when the team with the ball is ahead with this many seconds left or fewer. */
export const EXPECTED_RUN_WINDOW = 300;

/** Standard normal CDF (Abramowitz–Stegun 7.1.26, error under 1.5e-7). */
function normalCdf(z: number): number {
  const x = Math.abs(z) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * x);
  const erf = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return z >= 0 ? (1 + erf) / 2 : (1 - erf) / 2;
}

/** Chance a run from inside the 5 reaches the end zone (ignoring the rare fumble). */
export function goalLineRunChance(ytg: number): number {
  const shape = runShape("normal", ytg);
  // Gains are rounded to whole yards, so reaching the goal line means a raw gain of ytg - 0.5.
  return 1 - normalCdf((ytg - 0.5 - shape.mean) / shape.sd);
}

interface RunShape {
  /** Typical gain, before the occasional breakaway. */
  readonly mean: number;
  readonly sd: number;
  /** Chance of a breakaway adding extra yards. */
  readonly breakaway: number;
}

/**
 * Runs by field position. Inside the 5 the defense packs the line: short gains, no breakaways. From
 * there runs blend back to the open-field shape by the 12.
 */
export function runShape(coverage: Coverage, ytg: number, sticksToGo: number | null = null, expected = false): RunShape {
  const open: RunShape = expected ? EXPECTED_RUN : { mean: RUN_MEAN[coverage], sd: 3.2, breakaway: 0.1 };
  if (ytg <= 5) return { mean: 0.7 + 0.3 * ytg, sd: 2.2, breakaway: 0 };
  const t = Math.min(1, (ytg - 5) / 7);
  const lerp = (goalLine: number, field: number) => goalLine + (field - goalLine) * t;
  const shape = { mean: lerp(2.2, open.mean), sd: lerp(2.2, open.sd), breakaway: lerp(0, open.breakaway) };
  if (sticksToGo === null) return shape;
  // On 3rd or 4th and up to 6 to go, the defense plays to stop the run short of the marker. Fitted to 2022-26:
  // runs convert about 74% of the time on 1 yard to go, 61% on 2, 55% on 3, 44% on 4, 34% on 5 and 27% on 6.
  const mean = sticksToGo <= 3 ? 1.4 + 0.45 * sticksToGo : 1.1 + 0.45 * sticksToGo;
  return { mean: Math.min(shape.mean, mean), sd: Math.min(shape.sd, 2.4), breakaway: Math.min(shape.breakaway, 0.04) };
}

/** Yards to go on 3rd or 4th and 6 or fewer, when defenses play the run to stop it short of the marker. */
export const sticksRunFor = (g: GameState): number | null => (g.down >= 3 && g.toGo < OBVIOUS_TO_GO ? g.toGo : null);

/** A last-play deep pass from this close is a heave into the end zone rather than a normal deep throw. */
export const HAIL_MARY = Object.freeze({ maxClock: 10, maxYtg: 60, interception: 0.12, sack: 0.13 });

/**
 * Touchdown chance of an end-zone heave from this far out: about 9% from 25 yards, 4% from 45, 2% from 60.
 * Real last-play heaves (2022-26) scored about 2% of the time, and 5% of the throws that reached the end zone.
 */
export function hailMaryChance(ytg: number): number {
  return Math.max(0.02, 0.15 - 0.0025 * ytg);
}

function rollHailMary(ytg: number, r: Rng): PlayOutcome {
  const base = { outOfBounds: false, incomplete: false, seconds: 6 };
  const u = r();
  // A pick in the end zone is a touchback. The passer is sacked on about one heave in eight.
  if (u < HAIL_MARY.interception) return { ...base, yards: ytg, turnover: "interception" };
  if (u < HAIL_MARY.interception + HAIL_MARY.sack) return { ...base, yards: -8, seconds: 6, sack: true };
  if (u < HAIL_MARY.interception + HAIL_MARY.sack + hailMaryChance(ytg)) return { ...base, yards: ytg };
  return { ...base, yards: 0, incomplete: true };
}

/**
 * Lost fumbles, per run, per sack and per catch: real runs by a leading team late lose one about 0.5% of
 * the time, and strip-sacks and fumbles after the catch end about 4% of real late drives (2022-26).
 */
export const LOST_FUMBLE = Object.freeze({ run: 0.005, sack: 0.05, catch: 0.005 });

function rollPlay(
  call: PlayCall, r: Rng, coverage: Coverage, ytg: number, sticksToGo: number | null, sticks: SticksBand | null, mustStop = false,
  runExpected = false,
): PlayOutcome {
  const base = { outOfBounds: false, incomplete: false };
  switch (call) {
    case "kneel":
      return { ...base, yards: -1, seconds: PLAY_TYPES.kneel.duration };
    case "spike":
      return { ...base, yards: 0, seconds: 1, incomplete: true };
    case "run": {
      if (r() < LOST_FUMBLE.run) return { ...base, yards: 1, seconds: 5, turnover: "fumble" };
      const shape = runShape(coverage, ytg, sticksToGo, runExpected);
      let yards = Math.round(shape.mean + shape.sd * gauss(r));
      if (r() < shape.breakaway) yards += Math.round(6 + expo(r, 9));
      yards = Math.max(-6, yards);
      return { ...base, yards, seconds: Math.round(4 + Math.max(0, yards) / 7), outOfBounds: r() < 0.07 };
    }
    case "pass-sideline":
    case "pass-middle":
    case "pass-deep":
    case "pass-play-action":
    case "pass-screen": {
      const fitted = passOdds(coverage, call, ytg);
      const factor = sticks === null ? 1 : (STICKS_PASS[sticks][call] ?? 1);
      const odds = factor === 1 ? fitted : { ...fitted, complete: fitted.complete * factor };
      // Deep throws and play-action take longer to develop; a deep pick comes farther downfield.
      const deep = call === "pass-deep";
      const slow = deep || call === "pass-play-action" ? 1 : 0;
      const u = r();
      if (u < odds.sack) {
        const yards = deep ? -8 : -7;
        return { ...base, yards, seconds: 5 + slow, sack: true, ...(r() < LOST_FUMBLE.sack ? { turnover: "fumble" as const } : {}) };
      }
      if (u < odds.sack + odds.interception) return { ...base, yards: deep ? 25 : 12, seconds: 5 + slow, turnover: "interception" };
      if (u < odds.sack + odds.interception + odds.complete) {
        const yards = Math.round(odds.minYards + expo(r, odds.extraYards));
        if (r() < LOST_FUMBLE.catch) return { ...base, yards: Math.max(0, yards - 2), seconds: Math.round(4 + slow + yards / 10), turnover: "fumble" };
        const outOfBounds = call === "pass-sideline" && mustStop ? Math.max(odds.outOfBounds, NO_TIMEOUT_SIDELINE_OOB) : odds.outOfBounds;
        return { ...base, yards, seconds: Math.round(4 + slow + yards / 10), outOfBounds: r() < outOfBounds };
      }
      return { ...base, yards: 0, seconds: 5 + slow, incomplete: true };
    }
    case "timeout":
    case "fg":
    case "punt":
      throw new RangeError(`not a scrimmage play: ${call}`);
  }
}

function describePlay(call: PlayCall, p: PlayOutcome): string {
  if (call === "kneel") return "Kneel";
  if (call === "spike") return "Spike";
  if (p.turnover === "fumble") return p.sack ? `${PLAY_CALLS[call]}: strip-sack, lost` : `${PLAY_CALLS[call]}: fumble, lost`;
  if (p.turnover === "interception") return `${PLAY_CALLS[call]}: intercepted`;
  if (p.sack) return `${PLAY_CALLS[call]}: sacked for ${-p.yards}`;
  if (p.incomplete) return `${PLAY_CALLS[call]}: incomplete`;
  const gain = p.yards >= 0 ? `+${p.yards}` : `${p.yards}`;
  return `${PLAY_CALLS[call]}: ${gain}${p.outOfBounds ? ", out of bounds" : ", in bounds"}`;
}

/** The down after a play that doesn't convert. 4th down never gets here: it's a turnover on downs. */
const NEXT_DOWN = { 1: 2, 2: 3, 3: 4, 4: 4 } as const satisfies Record<Down, Down>;

function ordinal(n: Down) {
  return ["1st", "2nd", "3rd", "4th"][n - 1];
}

export function describeSpot(ytg: number): string {
  if (ytg === 50) return "midfield";
  return ytg > 50 ? `own ${100 - ytg}` : `opp ${ytg}`;
}

export function downAndDistance(g: GameState): string {
  return `${ordinal(g.down)} & ${g.toGo >= g.ytg ? "goal" : g.toGo} at ${describeSpot(g.ytg)}`;
}

/** Give the ball to the other team, `theirYtg` yards from the goal they attack. */
function changePossession(g: GameState, clock: number, theirYtg: number): GameState {
  const ytg = Math.min(99, Math.max(1, theirYtg));
  return {
    ...g, clock, running: false, poss: other(g.poss),
    down: 1, toGo: Math.min(10, ytg), ytg, over: clock <= 0,
  };
}

/** Kicking-game and try odds. The strategy guide reads these, so it always matches the model. */
export const SPECIAL_TEAMS = Object.freeze({
  /** Extra point success: 95.4% in 2022-2026 NFL regular seasons (about 5,400 kicks). */
  extraPoint: 0.954,
  /** Two-point try success: 48.1% over the same seasons (about 570 tries). */
  twoPoint: 0.481,
  /** Onside kick recovery by the kicking team: 5.5% for trailing teams in the last 5:00 (182 kicks, 2022-26). */
  onsideRecovery: 0.055,
  /**
   * A team still behind after scoring kicks onside with this many seconds left or fewer, or `onsideWindowShort`
   * with one timeout or none, or anywhere in the last `onsideWindowLong` when down 9 or more. Real teams down
   * 1-8 never kicked onside from 3:00 to 5:00, and with 3 timeouts never did inside 5:00 (0 of 82).
   */
  onsideWindow: 120,
  onsideWindowShort: 180,
  onsideWindowLong: 300,
  /** Late kickoffs under the 2025 rules: the receiving team starts at about its 32, after about 4 seconds of return. */
  kickoffStart: 32,
  kickoffSeconds: 4,
  /** Net punt distance from your own 40 or deeper (44.4 yards, 2022-26). */
  puntNet: 44,
  /** Net punt distance from midfield in, where the punter kicks for placement. */
  shortPuntNet: 40,
  /** A punt from the opponent's 40-49 pins them at about their 11 (63% are downed inside the 10). */
  puntPinnedAt: 11,
});

// ---------------------------------------------------------------------------
// The try after a touchdown

export type TryCall = "kick" | "two";
export const TRY_CALLS: Readonly<Record<TryCall, string>> = Object.freeze({ kick: "Kick the extra point", two: "Go for 2" });

/**
 * The scoring team's margin after the 6 points (before the try) at which it goes for two in the
 * simulations: where evaluateTry favors two across the final 10:00. NFL teams almost always go for two
 * down 10, 5 or 2 and up 1 or 5, and about half the time down 8; down 4 is the model's one departure
 * from practice (teams kick 98% of the time), a small edge from a field goal then winning instead of tying.
 */
export const GO_FOR_TWO_MARGINS: readonly number[] = [-10, -8, -5, -4, -2, 1, 5];

/** The rule-of-thumb try both teams follow in the simulations. */
export const tryPolicy = (marginAfterSix: number): TryCall => (GO_FOR_TWO_MARGINS.includes(marginAfterSix) ? "two" : "kick");

/**
 * Where the receiving team starts after a punt from this many yards out: 44 net from deep, shading to 40 at
 * midfield, and pinned at about their 11 from closer in.
 */
export function puntLanding(ytg: number): number {
  const deep = Math.min(1, Math.max(0, (ytg - 50) / 10));
  const net = SPECIAL_TEAMS.shortPuntNet + deep * (SPECIAL_TEAMS.puntNet - SPECIAL_TEAMS.shortPuntNet);
  return Math.max(SPECIAL_TEAMS.puntPinnedAt, Math.round(ytg - net));
}

/** Points, the try after a touchdown, then the kickoff (onside if the kicking team is still behind late). */
function score(g: GameState, clock: number, points: 3 | 6, r: Rng): { state: GameState; text: string } {
  const scorer = g.poss;
  const sign = scorer === "you" ? 1 : -1;
  let margin = g.margin + sign * points;
  let text = points === 6 ? "Touchdown" : "Field goal is good";
  if (points === 6) {
    if (tryPolicy(marginFor({ ...g, margin }, scorer)) === "two") {
      const good = r() < SPECIAL_TEAMS.twoPoint;
      if (good) margin += sign * 2;
      text += good ? ", 2-point try good" : ", 2-point try fails";
    } else {
      const good = r() < SPECIAL_TEAMS.extraPoint;
      if (good) margin += sign;
      text += good ? ", extra point good" : ", extra point missed";
    }
  }
  return kickoff({ ...g, margin }, clock, text, r);
}

/** After a score by the team with the ball in `scored`: the kickoff, onside if the scorer is still behind late. */
function kickoff(scored: GameState, clock: number, text: string, r: Rng): { state: GameState; text: string } {
  if (clock <= 0) return { state: { ...scored, clock: 0, running: false, over: true }, text: `${text}.` };
  const kickerMargin = marginFor(scored, scored.poss);
  const kickerTimeouts = scored.timeouts[scored.poss];
  const onside = kickerMargin < 0 && (clock <= SPECIAL_TEAMS.onsideWindow
    || (clock <= SPECIAL_TEAMS.onsideWindowShort && kickerTimeouts <= 1)
    || (clock <= SPECIAL_TEAMS.onsideWindowLong && kickerMargin <= -9));
  if (onside) {
    const t = Math.max(0, clock - 3);
    if (r() < SPECIAL_TEAMS.onsideRecovery) {
      // The kicking team recovers around its own 45 and keeps the ball.
      const kept: GameState = { ...scored, clock: t, running: false, down: 1, toGo: 10, ytg: 55 };
      return { state: kept, text: `${text}. Onside kick recovered by the kicking team.` };
    }
    return { state: changePossession(scored, t, 45), text: `${text}. Onside kick fails.` };
  }
  const t = Math.max(0, clock - SPECIAL_TEAMS.kickoffSeconds);
  return {
    state: changePossession(scored, t, 100 - SPECIAL_TEAMS.kickoffStart),
    text: `${text}. Kickoff, returned to their ${SPECIAL_TEAMS.kickoffStart}.`,
  };
}

/**
 * Apply one decision by the team with the ball: the pre-snap clock, the play, the clock after it,
 * a defensive timeout if the defense wants one, and any change of possession.
 */
export function step(
  g: GameState, call: PlayCall, r: Rng, cfg: SimConfig = DEFAULT_SIM, forced?: PlayOutcome, tempo: Tempo = "auto",
): StepResult {
  if (g.over) throw new RangeError("the game is over");
  const team = g.poss;
  const def = other(team);

  if (call === "timeout") {
    if (g.timeouts[team] === 0) throw new RangeError("no timeouts left");
    const timeouts = { ...g.timeouts, [team]: g.timeouts[team] - 1 };
    const wasted = g.running ? "" : " (the clock was already stopped)";
    let clock = g.clock;
    let note = "";
    if (g.running && tempo === "drain") {
      // Let the play clock run down to 1 before calling it, or stop at 0:03 if that comes first, leaving
      // time for a field goal. The two-minute warning stops the clock on its own first.
      const drained = g.clock - BURN;
      if (g.clock > WARNING && drained <= WARNING) {
        clock = WARNING;
        note = " after the two-minute warning";
      } else {
        clock = Math.max(drained, Math.min(g.clock, FG_TIMEOUT_AT));
      }
    } else if (g.running && tempo === "auto" && fgPlay(g) && (g.down === 4 || g.clock <= kickFloor(g, cfg))) {
      // At the kick, a team that only needs the field goal lets the clock run down before taking its last
      // stop, so the kick ends the game rather than handing the other team time to answer: to 0:03, or as
      // far as one play clock allows.
      clock = Math.max(g.clock - BURN, Math.min(g.clock, FG_TIMEOUT_AT));
      if (g.clock > WARNING && clock <= WARNING) {
        clock = WARNING;
        note = " after the two-minute warning";
      }
    }
    return {
      state: { ...g, clock, running: false, timeouts },
      text: `${who(team)} call timeout at ${formatClock(clock)}${note}${wasted}, ${timeouts[team]} left.`,
    };
  }

  // Before the snap. If the clock reaches 2:00 first, the two-minute warning stops it there, before any play:
  // the down stays the same, and the next play is called from 2:00 with the clock stopped.
  let t = g.clock;
  if (g.running) {
    const after = t - preSnapBurn(g, call, cfg, tempo);
    if (t > WARNING && after <= WARNING) {
      return { state: { ...g, clock: WARNING, running: false }, text: "Two-minute warning.", twoMinuteWarning: true };
    } else if (after <= 0) {
      return { state: { ...g, clock: 0, running: false, over: true }, text: "The clock runs out before the snap." };
    } else {
      t = after;
    }
  }
  const snapClock = t;
  const crossed = (after: number) => snapClock > WARNING && after <= WARNING;

  if (call === "fg") {
    const end = Math.max(0, t - 5);
    const distance = fgDistance(g.ytg);
    if (r() < fgMakeChance(g.ytg)) {
      const s = score(g, end, 3, r);
      return { state: s.state, text: `${distance}-yard try: ${s.text}`, driveEnded: "Field goal" };
    }
    // They take over at the spot of the kick, or their 20 if that's farther back.
    const missed = changePossession(g, end, Math.min(80, 100 - (g.ytg + 7)));
    return { state: missed, text: `${distance}-yard field goal: no good.`, driveEnded: "Missed field goal" };
  }

  if (call === "punt") {
    const end = Math.max(0, t - 7);
    const theirYtg = 100 - puntLanding(g.ytg);
    return {
      state: changePossession(g, end, theirYtg),
      text: `Punt at ${formatClock(snapClock)}. They start at their ${100 - theirYtg}.`,
      driveEnded: "Punt",
    };
  }

  const play = forced ?? (call === "pass-deep" && snapClock <= HAIL_MARY.maxClock && g.ytg <= HAIL_MARY.maxYtg
    ? rollHailMary(g.ytg, r)
    : rollPlay(call, r, coverageFor(g, snapClock), g.ytg, sticksRunFor(g), sticksBand(g), mustGetOut(g, snapClock),
      marginFor(g, g.poss) > 0 && snapClock <= EXPECTED_RUN_WINDOW));
  t = Math.max(0, t - play.seconds);
  const what = describePlay(call, play);

  if (play.turnover) {
    const spot = g.ytg - play.yards;
    const theirYtg = spot <= 0 ? 80 : 100 - spot;
    return { state: changePossession(g, t, theirYtg), text: `${what}.`, driveEnded: play.turnover === "fumble" ? "Fumble" : "Interception" };
  }

  if (play.yards >= g.ytg) {
    const s = score(g, t, 6, r);
    return { state: s.state, text: `${describePlay(call, { ...play, yards: g.ytg })}. ${s.text}`, driveEnded: "Touchdown" };
  }

  const ytg = Math.min(99, g.ytg - play.yards);
  const firstDown = play.yards >= g.toGo;
  if (!firstDown && g.down === 4) {
    return { state: changePossession(g, t, 100 - ytg), text: `${what}. Turnover on downs.`, driveEnded: "Turnover on downs" };
  }

  let next: GameState = {
    ...g,
    clock: t,
    ytg,
    down: firstDown ? 1 : NEXT_DOWN[g.down],
    toGo: firstDown ? Math.min(10, ytg) : g.toGo - play.yards,
    running: !play.outOfBounds && !play.incomplete && !crossed(t) && t > 0,
    over: t <= 0,
  };
  let text = `${what} at ${formatClock(t)}.`;
  if (crossed(t)) text += " Two-minute warning.";
  if (next.over) return { state: next, text: `${what}. Time expires.` };

  // A defense that's behind stops the clock, unless the two-minute warning is about to do it for free.
  const defenseBehind = marginFor(next, def) < 0;
  const warningComing = t > WARNING && t - preSnapBurn(next, "run", cfg) <= WARNING;
  if (next.running && (defenseBehind || savesTimeForAnswer(next, def)) && next.timeouts[def] > 0 && t <= DEFENSE_TIMEOUT_WINDOW && !warningComing) {
    const timeouts = { ...next.timeouts, [def]: next.timeouts[def] - 1 };
    next = { ...next, running: false, timeouts };
    text += ` ${who(def)} call timeout (${timeouts[def]} left).`;
  }
  return { state: next, text: `${text} ${firstDown ? "First down. " : ""}${downAndDistance(next)}.` };
}

// ---------------------------------------------------------------------------
// The rule-of-thumb policy both teams follow after your decision

/** Whether kneel-downs end the game from here, given the defense's timeouts. */
function canKneelOut(g: GameState): boolean {
  return simulateRunOut({
    gameClock: g.clock, down: g.down, clockRunning: g.running, defTimeouts: g.timeouts[other(g.poss)],
  }).endsGame;
}

/**
 * 4th down for a team a field goal would put level or ahead (or that's already ahead by a field goal or
 * less): kick, or go for it? Kicking with time left hands them the ball with time to answer; going for it
 * keeps the ball, to kick later or run the clock out. So the rule goes for it when there's time left, the
 * kick is long and the distance is short: with more than `minClock` seconds left, when
 * `perSecond` x seconds (up to `clockCap`) + `perYard` x yards to the goal - `perToGo` x yards to go
 * beats a bar. With 1:15 or more left that's 4th and 2 or less at the 5, 4 or less at the 20 and 7 or less
 * at the 40 for a team that's tied or behind; a team that's ahead goes for it on about 4 yards more.
 *
 * The shape comes from the model's own kick-or-go values over 174 clock, spot and distance combinations;
 * the bars were then set by full games, over 16,632 tied-to-down-3 situations and 12,474 leading ones.
 */
export const FOURTH_DOWN_GO = { minClock: 30, perSecond: 0.4, clockCap: 75, perYard: 0.45, perToGo: 3, tiedOrBehind: 26, ahead: 14 } as const;
function goesForIt(g: GameState, bar: number): boolean {
  const f = FOURTH_DOWN_GO;
  return g.clock > f.minClock && f.perSecond * Math.min(g.clock, f.clockCap) + f.perYard * g.ytg - f.perToGo * g.toGo > bar;
}
/** Needing a touchdown this close to the goal line, on 3 yards to go or less, the rule of thumb runs. */
export const GOAL_LINE_RUN_YTG = 5;
/** Before a winning kick with the clock stopped, a team kneels from this close, and runs from further out. */
export const KNEEL_BEFORE_KICK_YTG = 10;
/** With less than this share of the time a drive needs, under a minute, the rule of thumb throws deep. */
export const DESPERATE_SHARE = 0.5;

/**
 * The rule-of-thumb call. Its late-game rules were tested one at a time and together over 17,820 tied and
 * trailing situations (every down, 0:08 to 2:30 left, the whole field) and 5,940 leading ones. Each was
 * kept only if the team following it won more often against a team that didn't; together they raise that
 * team's average win probability by about 2 points over the rules before them. Late drives still score as
 * often as real ones did (scripts/calibrate/drives.ts).
 */
export function policy(g: GameState, cfg: SimConfig = DEFAULT_SIM): PlayCall {
  const m = marginFor(g, g.poss);
  const to = g.timeouts[g.poss];
  const t = g.clock;

  if (m > 0) {
    if (canKneelOut(g)) return "kneel";
    if (g.down === 4) {
      // Ahead by a field goal or less at midfield or closer, on a short enough 4th down: go for it. A first
      // down all but ends the game, which is worth more than the kick or the punt (see FOURTH_DOWN_GO).
      if (m <= 3 && g.ytg <= 50 && goesForIt(g, FOURTH_DOWN_GO.ahead)) return g.toGo <= 2 ? "run" : "pass-middle";
      return g.ytg <= 35 ? "fg" : g.ytg >= PUNT_MIN_YTG ? "punt" : "run";
    }
    return "run";
  }

  // Tied, out of range, with a minute or less: kneel it out and take it to overtime. A team that starts
  // kneeling keeps kneeling (a kneel at 0:59 leaves 0:57, still under the line), so a kneel is scored as one.
  // Not on 4th down: that hands them the ball in range. The 4th-down rule below punts instead.
  if (m === 0 && g.ytg > 60 && t <= TIED_KNEEL_CLOCK && g.down < 4) return "kneel";

  const hurry = t <= LATE_HURRY_WINDOW ? Math.min(cfg.hurryUp, LATE_HURRY_UP) : cfg.hurryUp;

  if (fgPlay(g)) {
    const floor = kickFloor(g, cfg);
    // 4th and short with time left: kicking now leaves them time to answer, so go for it and kick later.
    if (g.down === 4 && goesForIt(g, FOURTH_DOWN_GO.tiedOrBehind)) return g.toGo <= 2 ? "run" : "pass-middle";
    if (g.down === 4 || t <= floor) {
      if (!g.running) {
        if (g.down <= 3) {
          // Downs to spare with the clock stopped: use the time before the kick instead of leaving it for
          // them. Down 3 with a timeout, that's a shot at the touchdown. Otherwise a kneel from chip-shot
          // range, or a run from further out (a shorter kick is worth the small risk), if the kick still fits
          // after it: a timeout at 0:03, or the time to get the unit on.
          if (m === -3 && to > 0 && t > 9) return g.ytg <= 10 ? "pass-middle" : g.ytg <= 16 ? "pass-play-action" : "pass-deep";
          if (g.ytg <= KNEEL_BEFORE_KICK_YTG) {
            if (to > 0 ? t > FG_TIMEOUT_AT + 3 : t >= cfg.fgRush + 4) return "kneel";
          } else if (to > 0 ? t > FG_TIMEOUT_AT + DRAIN_PLAY_SECONDS + 1 : t >= cfg.fgRush + DRAIN_PLAY_SECONDS + 2) {
            return "run";
          }
        }
        return "fg";
      }
      if (to > 0) return "timeout";
      // No timeout and the clock running: the field-goal unit needs `fgRush` seconds to get on. With less
      // than that, spike it to stop the clock, then kick; on 4th down, all that's left is to run a play.
      if (t <= cfg.fgRush + 1 && g.down <= 3 && t > cfg.spikeLineup + 1) return "spike";
      if (t <= cfg.fgRush) return g.ytg <= 3 ? "run" : "pass-middle";
      return "fg";
    }
    return "run";
  }

  // A field goal ties or wins and it's within reach (a 65-yarder at most), with time for one last play at
  // most: kick. With the clock stopped, that's 12 seconds with a timeout, or enough for a play and a spike
  // without one. With it running: a timeout first; without one, a spike if the unit can't get on in time,
  // and the kick itself (letting the clock run down to it) once a play, a spike and a kick no longer fit.
  if (m >= -3 && g.ytg <= LAST_PLAY_FG_YTG) {
    const playAndSpike = cfg.spikeLineup + DRAIN_PLAY_SECONDS + 1;
    if (!g.running) {
      if (t <= (to > 0 ? 12 : playAndSpike)) return "fg";
    } else if (to > 0) {
      if (t <= 12) return "timeout";
    } else if (t <= playAndSpike) {
      return g.down <= 3 && t > cfg.spikeLineup + 1 ? "spike" : "fg";
    } else if (t <= hurry + playAndSpike + 1) {
      return "fg";
    }
  }

  // The clock is running and time is tight (see DRIVE_TIME): stop it before anything else, whatever the
  // down. With a timeout, call it. Without one, when there isn't even time to snap a play, spike it for one
  // more. Not before a punt, which gives the ball away: the time would be theirs.
  const tight = t <= driveTimeNeeded(g);
  const punting = g.down === 4 && !(m >= -3 && g.ytg <= 42) && ((m === 0 && g.ytg >= PUNT_MIN_YTG) || (t > 300 && g.ytg >= 50));
  if (g.running && !punting) {
    if (to > 0 && tight && t <= 120) return "timeout";
    if (to === 0 && t <= hurry + 1 && g.down <= 3 && t > cfg.spikeLineup + 1) return "spike";
  }

  // Need a touchdown, under 45 seconds, and the end zone within reach: go for it. Inside the 20 an end-zone
  // throw rarely connects, so play the goal line: run from the 1 or 2 with a timeout to stop the clock (or
  // from the 1 with 20 seconds to spare), throw over the middle inside the 10, play-action to the 16, and
  // only throw at the end zone beyond that. Chosen by testing rules over 1,760 late red-zone situations
  // (every distance, clock, timeouts, down, down 4 or 7): 32.1% average win probability against 21.3% for
  // throwing at the end zone every time, with the biggest gains near the goal line (49% against 19% from the 1).
  if (m < -3 && t <= END_ZONE_SHOT_CLOCK && g.ytg <= 40) {
    if (g.ytg > RED_ZONE) return "pass-deep";
    if ((g.ytg <= 2 && to > 0) || (g.ytg <= 1 && t >= 20)) return "run";
    return g.ytg <= 10 ? "pass-middle" : g.ytg <= 16 ? "pass-play-action" : "pass-deep";
  }

  // Last play, or close to it, and a field goal won't do: take a shot at the end zone. A tied team doesn't
  // heave it on 4th down, though: it punts (below) and plays for overtime.
  if (t <= 12 && (m < -3 || g.ytg > 42) && (g.down < 4 || m < 0)) return "pass-deep";

  if (g.down === 4) {
    const goForIt = g.toGo <= 2 ? "run" : g.toGo >= 15 ? "pass-deep" : "pass-middle";
    // A field goal that ties or takes the lead: kick, unless there's time for them to answer it and the
    // distance is short enough to go for it (see FOURTH_DOWN_GO).
    if (m >= -3 && g.ytg <= 42) return goesForIt(g, FOURTH_DOWN_GO.tiedOrBehind) ? goForIt : "fg";
    // Tied and out of range: punt and play for overtime, rather than hand them a short field.
    if (m === 0 && g.ytg >= PUNT_MIN_YTG) return "punt";
    if (t > 300 && g.ytg >= 50) return "punt";
    return goForIt;
  }
  if (g.running && t <= 40 && g.down <= 2 && to === 0 && tight) return "spike";
  // Under a minute with less than half the time the drive needs: short throws can't get there, so throw deep.
  if (t <= 60 && t <= DESPERATE_SHARE * driveTimeNeeded(g)) return "pass-deep";
  // Short yardage near the goal line when a touchdown is needed: run it, if the clock can take a tackle in bounds.
  if (m < -3 && g.ytg <= GOAL_LINE_RUN_YTG && g.toGo <= 3 && (to > 0 || !tight)) return "run";
  // Late passing, chosen by testing rules over 48 trailing situations against the calibrated odds:
  // play-action with field and time to work with, the sideline when there's no timeout and little time,
  // otherwise over the middle (sideline catches only get out of bounds about a quarter of the time).
  if (t <= 120) return g.ytg > 40 && t > 30 ? "pass-play-action" : to === 0 && t <= 40 ? "pass-sideline" : "pass-middle";
  return g.down === 1 && g.toGo >= 10 && t > 150 ? "run" : "pass-middle";
}

/**
 * Play the other team's possessions out with the policy, until the ball comes back to you or the game ends.
 * Returns each play's text, and the situation when it stopped.
 */
export function playUntilYourBall(g: GameState, r: Rng, cfg: SimConfig = DEFAULT_SIM): { lines: string[]; state: GameState } {
  const lines: string[] = [];
  let s = g;
  for (let i = 0; i < 200 && !s.over && s.poss !== "you"; i++) {
    const res = step(s, policy(s, cfg), r, cfg);
    lines.push(res.text);
    s = res.state;
  }
  return { lines, state: s };
}

/** Play the rest of the game out with both teams following the policy. 1 = you win, 0 = you lose. */
export function playOut(
  g: GameState, r: Rng, cfg: SimConfig = DEFAULT_SIM, log?: (s: StepResult, offense: Team) => void,
): number {
  let s = g;
  for (let i = 0; i < 500 && !s.over; i++) {
    const res = step(s, policy(s, cfg), r, cfg);
    log?.(res, s.poss);
    s = res.state;
  }
  return s.margin > 0 ? 1 : s.margin < 0 ? 0 : 0.5;
}

// ---------------------------------------------------------------------------
// Win probability

/**
 * Every situation is simulated on the same random streams (common random numbers), so comparing two
 * situations, e.g. with and without a timeout, compares like with like instead of two independent
 * samples. Estimates stay deterministic.
 */
const SEED = 0x5eed;
const streamFor = (i: number) => seededRng(SEED + i * 7919);

/** Your chance of winning from here, with both teams following the policy. */
export function winProbability(g: GameState, n = 1500, cfg: SimConfig = DEFAULT_SIM): number {
  if (g.over) return g.margin > 0 ? 1 : g.margin < 0 ? 0 : 0.5;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += playOut(g, streamFor(i), cfg);
  return sum / n;
}

/**
 * Your win probability if a play snapped from `g` turns out exactly as `outcome`: the play's result is
 * fixed, and everything after it is simulated, on the same random streams as every other estimate.
 * With no `outcome`, the play's result is rolled as usual. `g.clock` is the clock at the snap, so no
 * time runs off before it.
 */
export function outcomeWinProbability(
  g: GameState, call: PlayCall, outcome: PlayOutcome | undefined, n = 2000, cfg: SimConfig = DEFAULT_SIM,
): number {
  const atSnap = { ...g, running: false };
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const r = streamFor(i);
    sum += playOut(step(atSnap, call, r, cfg, outcome).state, r, cfg);
  }
  return sum / n;
}

/** What one play does to the situation, with its result fixed: the state after it and how it reads. */
export function outcomeStep(g: GameState, call: PlayCall, outcome: PlayOutcome, cfg: SimConfig = DEFAULT_SIM): StepResult {
  return step({ ...g, running: false }, call, streamFor(0), cfg, outcome);
}

/** A call and your win probability if you make it. */
export interface ScoredCall {
  readonly call: PlayCall;
  readonly wp: number;
}

/** Every call available in a situation, scored. Never empty: there's always a play to run. */
export type CallValues = NonEmpty<ScoredCall>;

/**
 * Your win probability for each call available now, followed by the policy afterward.
 * Every call is scored on the same random streams, so differences between calls are less noisy.
 */
/**
 * Your win probability with one call at a tempo you pick for its snap, on the same random streams as
 * `evaluateCalls`, so it compares directly with the values there. A timeout looks ahead to the best play
 * after it, as there.
 */
export function scoreCallAtTempo(g: GameState, call: PlayCall, tempo: Tempo, n = 500, cfg: SimConfig = DEFAULT_SIM): number {
  if (call === "timeout") {
    const after = step(g, "timeout", streamFor(0), cfg, undefined, tempo).state;
    if (g.running) return bestScored(scoreCalls(after, availableCalls(after).filter((c) => c !== "timeout"), n, cfg)).wp;
    return wpOf(evaluateCalls(g, n, cfg), "timeout");
  }
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const r = streamFor(i);
    sum += playOut(step(g, call, r, cfg, undefined, tempo).state, r, cfg);
  }
  return sum / n;
}

export function evaluateCalls(g: GameState, n = 500, cfg: SimConfig = DEFAULT_SIM): CallValues {
  return scoreCalls(g, availableCalls(g), n, cfg);
}

function scoreCalls(g: GameState, calls: readonly PlayCall[], n: number, cfg: SimConfig): CallValues {
  const scorePlay = (state: GameState, call: PlayCall): number => {
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const r = streamFor(i);
      sum += playOut(step(state, call, r, cfg).state, r, cfg);
    }
    return sum / n;
  };

  const plays = calls.filter((c) => c !== "timeout").map((call): ScoredCall => ({ call, wp: scorePlay(g, call) }));
  const [firstPlay, ...otherPlays] = plays;
  if (firstPlay === undefined) throw new RangeError("the game is over: there are no calls to score");
  const scoredPlays: CallValues = [firstPlay, ...otherPlays];
  if (!calls.includes("timeout")) return scoredPlays;

  // A timeout isn't a play: it stops the clock and spends the timeout, then you run a play.
  const after = step(g, "timeout", streamFor(0), cfg).state;
  const timeoutWp = g.running
    // Stopping the clock changes the situation, so look ahead: the best play from there. This is
    // what the win-probability meter shows after you call it. With more time than the drive needs, it
    // can't beat the best play either: you could run that play and call the timeout later, if you need it.
    ? Math.min(
      bestScored(scoreCalls(after, availableCalls(after).filter((c) => c !== "timeout"), n, cfg)).wp,
      earlyTimeout(g) ? bestScored(scoredPlays).wp : 1,
    )
    // With the clock already stopped, nothing changes but the timeout count. Score the same best
    // play with one fewer timeout, on the same random streams, so the difference is what the timeout
    // was worth (taking the best of several noisy estimates would overstate it). It can't beat that
    // play outright: you could always run it and keep the timeout for later.
    : Math.min(scorePlay(after, bestScored(scoredPlays).call), bestScored(scoredPlays).wp);
  return [{ call: "timeout", wp: timeoutWp }, ...scoredPlays];
}

/** How `evaluateCallsPrecisely` spends its games. */
export interface Precision {
  /** Games per call for the first pass over every call. */
  readonly quick: number;
  /** Games per call for the second pass, over the calls close to the leader. */
  readonly precise: number;
  /** A call within this many win-probability points of the first pass's leader gets the second pass. */
  readonly window: number;
}

/**
 * Enough to grade a call: at 500 games an estimate can be off by up to 4 points, enough to call a near-tie
 * a clear mistake; at 3,000 the worst case across late situations is under 2.
 */
export const GRADING: Precision = Object.freeze({ quick: 500, precise: 3000, window: 0.04 });

/** One call's win probability over `n` games, on the shared random streams. */
function scoreOne(g: GameState, call: PlayCall, n: number, cfg: SimConfig, tempo: Tempo = "auto"): number {
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const r = streamFor(i);
    sum += playOut(step(g, call, r, cfg, undefined, tempo).state, r, cfg);
  }
  return sum / n;
}

/** Plays (not a timeout) scored in two passes: all of them quickly, then the close ones precisely. */
function playsPrecisely(g: GameState, calls: readonly PlayCall[], p: Precision, cfg: SimConfig): CallValues {
  const first = calls.map((call): ScoredCall => ({ call, wp: scoreOne(g, call, p.quick, cfg) }));
  const lead = Math.max(...first.map((v) => v.wp));
  const [head, ...rest] = first.map((v) => (lead - v.wp <= p.window ? { call: v.call, wp: scoreOne(g, v.call, p.precise, cfg) } : v));
  if (head === undefined) throw new RangeError("the game is over: there are no calls to score");
  return [head, ...rest];
}

/**
 * Every call available now, scored precisely enough to tell close calls apart without simulating every
 * call at full precision: all of them get a quick pass, and those within `window` of the leader a precise
 * one (calls further back are clearly worse either way). A timeout is scored as in `evaluateCalls`: with
 * the clock running, by the best play from the stopped clock, found the same two-pass way.
 */
export function evaluateCallsPrecisely(g: GameState, cfg: SimConfig = DEFAULT_SIM, p: Precision = GRADING): CallValues {
  const calls = availableCalls(g);
  const plays = playsPrecisely(g, calls.filter((c) => c !== "timeout"), p, cfg);
  if (!calls.includes("timeout")) return plays;
  const after = step(g, "timeout", streamFor(0), cfg).state;
  const best = bestScored(plays);
  const timeoutWp = g.running
    ? Math.min(bestScored(playsPrecisely(after, availableCalls(after).filter((c) => c !== "timeout"), p, cfg)).wp, earlyTimeout(g) ? best.wp : 1)
    : Math.min(scoreOne(after, best.call, p.precise, cfg), best.wp);
  return [{ call: "timeout", wp: timeoutWp }, ...plays];
}

/**
 * Your win probability with one call at a tempo you pick for its snap, as precisely as
 * `evaluateCallsPrecisely` scores the close calls, so it compares directly with the values there.
 */
export function scoreCallPrecisely(g: GameState, call: PlayCall, tempo: Tempo, cfg: SimConfig = DEFAULT_SIM, p: Precision = GRADING): number {
  if (call !== "timeout") return scoreOne(g, call, p.precise, cfg, tempo);
  const after = step(g, "timeout", streamFor(0), cfg, undefined, tempo).state;
  if (g.running) return bestScored(playsPrecisely(after, availableCalls(after).filter((c) => c !== "timeout"), p, cfg)).wp;
  return wpOf(evaluateCallsPrecisely(g, cfg, p), "timeout");
}

/**
 * Score every call but a timeout from a situation at the snap, as play-by-play data records it: the
 * clock is where the snap happened, so no time runs off before it, and a timeout is no longer an option.
 */
export function evaluateCallsAtSnap(g: GameState, n = 1500, cfg: SimConfig = DEFAULT_SIM): CallValues {
  const atSnap = { ...g, running: false };
  return scoreCalls(atSnap, availableCalls(atSnap).filter((c) => c !== "timeout"), n, cfg);
}

/** Your win probability with a call that was scored. */
export function wpOf(values: CallValues, call: PlayCall): number {
  const found = values.find((v) => v.call === call);
  if (found === undefined) throw new RangeError(`${call} wasn't available, so it wasn't scored`);
  return found.wp;
}

/** The calls sorted from best to worst win probability. */
/**
 * The calls from best to worst win probability. Exact ties go to the call that spends the least, so a
 * timeout that only matches a play (it can never beat it with the clock stopped) ranks below that play.
 */
export function ranked(values: CallValues): ScoredCall[] {
  return [...values].sort((a, b) => b.wp - a.wp || TIE_ORDER.indexOf(a.call) - TIE_ORDER.indexOf(b.call));
}

/**
 * Calls that can't help with the clock already stopped: a timeout spends a timeout and a spike a down, and
 * neither gains anything. They can tie the best play at most, never beat it.
 */
export function noGainCalls(g: GameState): readonly PlayCall[] {
  return g.running ? [] : ["timeout", "spike"];
}

/** When calls are this close, they count as tied. */
export const TIE = 0.005;

/**
 * Among tied calls, prefer the one that spends the least: no timeout, no down, no risk. On 4th down,
 * a kick counts as less risky than going for it.
 */
const TIE_ORDER: readonly PlayCall[] = [
  "kneel", "punt", "fg", "run", "pass-screen", "pass-middle", "pass-sideline", "pass-play-action", "pass-deep", "spike", "timeout",
];

/** The best call and its win probability. */
export function bestScored(values: CallValues): ScoredCall {
  const top = Math.max(...values.map((v) => v.wp));
  const preference = (v: ScoredCall) => TIE_ORDER.indexOf(v.call);
  return values.reduce((best, v) =>
    v.wp >= top - TIE && (best.wp < top - TIE || preference(v) < preference(best)) ? v : best);
}

export function bestCall(values: CallValues): PlayCall {
  return bestScored(values).call;
}

// ---------------------------------------------------------------------------
// Explaining a decision

export interface Explanation {
  verdict: "best" | "close" | "worse";
  headline: string;
  /** Why the best call was best, and what was wrong with yours if it wasn't. */
  reasons: Reason[];
}

/** Calls within this many win-probability points of the best count as about as good. */
export const CLOSE = 0.015;
/** Calls at least this far below the best are worth warning against. */
export const AVOID = 0.03;

const pct = (x: number) => `${Math.round(x * 100)}%`;

/** One reason in an explanation, and the guide section that explains it further. */
export interface Reason {
  readonly text: string;
  readonly section: GuideSection;
}

const reason = (text: string, section: GuideSection): Reason => ({ text, section });

/** Why a call is the best one here, and the guide section that explains it. */
export function bestCallReason(g: GameState, call: PlayCall, cfg: SimConfig = DEFAULT_SIM): Reason {
  return whyBest(g, call, cfg);
}

function whyBest(g: GameState, call: PlayCall, cfg: SimConfig): Reason {
  const m = marginFor(g, g.poss);
  // Ahead on 4th down with the clock running: a stop hands them the ball with the clock stopped, so a
  // play's value is the time that comes off before the snap.
  if (m > 0 && g.down === 4 && g.running && (call === "run" || call === "kneel" || call.startsWith("pass-")))
    return reason(`Let the play clock run down first: that takes ${BURN} seconds off before the snap. On 4th down, if you're stopped they get the ball with the clock stopped, so the time has to come off before the snap, not after. Converting ends the game.`, "running-out-the-clock");
  switch (call) {
    case "kneel":
      if (m > 0 && canKneelOut(g))
        return reason("Kneel-downs run out the clock from here, so the opponent never gets the ball back.", "running-out-the-clock");
      if (m > 0) return reason("A kneel keeps the clock running with no risk of a fumble.", "running-out-the-clock");
      if (fgPlay(g))
        return reason("A field goal is enough and you're in range. A kneel drains the clock with no risk of a fumble, at the cost of a yard on the kick.", "field-goal-endgame");
      return reason("With little time and a long field, taking a tie to overtime beats risking a turnover.", "limitations");
    case "run":
      if (m > 0) return reason("A run keeps the clock moving and rarely turns the ball over.", "running-out-the-clock");
      if (fgPlay(g))
        return reason("A field goal is enough and you're in range. A run drains the clock and protects the ball before the kick.", "field-goal-endgame");
      if (g.ytg <= 5)
        return reason(`At the ${g.ytg}, the defense packs the line, but a run still scores about ${pct(goalLineRunChance(g.ytg))} of the time from here, with little risk of a turnover.`, "red-zone");
      if (g.toGo <= 2) return reason(`On ${ordinal(g.down)} & ${g.toGo}, a run is the surest way to move the chains.`, "play-model");
      return reason(
        g.clock > 120 || !g.running
          ? "A run gains steady yards with little risk, and you have time to spare."
          : "A run gains steady yards with little risk of a turnover.",
        "play-model",
      );
    case "timeout":
      if (fgPlay(g)) return reason("Stop the clock now so there's time to get the kick off.", "field-goal-endgame");
      return reason(`The clock was running with ${formatClock(g.clock)} left. A timeout saves the ${g.clock <= LATE_HURRY_WINDOW ? Math.min(cfg.hurryUp, LATE_HURRY_UP) : cfg.hurryUp} seconds a hurry-up snap would burn.`, "timeouts");
    case "spike":
      return reason(`A spike stops the clock about ${cfg.spikeLineup} seconds after the play and saves a timeout, at the cost of a down.`, "spike");
    case "pass-sideline": {
      const odds = passOdds(coverageFor(g, g.clock), "pass-sideline", g.ytg);
      return reason(`A sideline throw completes about ${pct(odds.complete / (1 - odds.sack))} of the time, and about ${pct(odds.outOfBounds)} of catches end out of bounds, stopping the clock without spending a timeout.`, "clock-rules");
    }
    case "pass-middle":
      if (g.ytg <= RED_ZONE)
        return reason(`Coverage tightens in the red zone, but a quick throw over the middle still completes about ${pct(passOdds("normal", "pass-middle", g.ytg).complete)} of the time from the ${g.ytg}.`, "red-zone");
      return coverageFor(g, g.clock) === "prevent"
        ? reason(`They're ahead late, so they're playing prevent: deep and soft. Throws over the middle complete about ${pct(PASS_ODDS.prevent["pass-middle"].complete)} of the time against it.`, "prevent-defense")
        : reason("A pass over the middle is the best mix of yards and safety, and the clock isn't the main problem yet.", "play-model");
    case "pass-deep":
      if (g.clock <= 12)
        return reason(`With ${formatClock(g.clock)} left and ${g.ytg} yards to go, there's time for one shot: throw it to the end zone.${g.ytg <= HAIL_MARY.maxYtg ? ` A heave from here scores about ${pct(hailMaryChance(g.ytg))} of the time.` : ""}`, "hail-mary");
      if (g.ytg <= RED_ZONE)
        return reason(`From the ${g.ytg} there's no deep field: a deep pass is a throw into the end zone, and it scores about ${pct(passOdds("normal", "pass-deep", g.ytg).complete)} of the time.`, "red-zone");
      return reason(`You need a big play: ${g.ytg} yards with ${formatClock(g.clock)} left is too far to go in small chunks. A deep pass completes about ${pct(passOdds(coverageFor(g, g.clock), "pass-deep", g.ytg).complete)} of the time, for 20+ yards.`, coverageFor(g, g.clock) === "prevent" ? "prevent-defense" : "play-model");
    case "pass-play-action": {
      const odds = passOdds(coverageFor(g, g.clock), "pass-play-action", g.ytg);
      return reason(`A run fake pulls the linebackers up and opens the middle of the field: play-action completes about ${pct(odds.complete)} of the time here, for ${odds.minYards}+ yards.`, coverageFor(g, g.clock) === "prevent" ? "prevent-defense" : "play-model");
    }
    case "pass-screen": {
      const coverage = coverageFor(g, g.clock);
      return reason(
        coverage === "prevent"
          ? `They're playing soft, so a screen gets the ball out fast into open space: it completes about ${pct(PASS_ODDS.prevent["pass-screen"].complete)} of the time and is almost never sacked or picked off.`
          : `A screen is the safest throw: it completes about ${pct(passOdds(coverage, "pass-screen", g.ytg).complete)} of the time and is almost never sacked or picked off, for steady yards.`,
        coverage === "prevent" ? "prevent-defense" : "play-model",
      );
    }
    case "fg": {
      const after = m + 3;
      const effect = after > 0 && m <= 0 ? "wins it" : after === 0 ? "ties it" : "adds to the lead";
      return reason(`A ${fgDistance(g.ytg)}-yard try is good about ${pct(fgMakeChance(g.ytg))} of the time, and it ${effect}.`, "field-goal-endgame");
    }
    case "punt":
      return reason("Punting pins them deep. Failing on 4th down here would hand them a short field.", "fourth-down");
  }
}

/** Why a call was worse than the best one, when there's a specific reason (null when it was only a bit worse). */
export function whyNot(g: GameState, call: PlayCall): Reason | null {
  const m = marginFor(g, g.poss);
  if (call === "timeout" && g.running && m > 0)
    return reason(`A timeout stops the clock for them: it gives back the ${BURN} seconds you could have run off before the snap.`, "running-out-the-clock");
  if (call === "timeout" && !g.running)
    return reason("A timeout gains nothing with the clock already stopped, and spends one you may need later.", "timeouts");
  if (call === "timeout" && hasTimeToSpare(g))
    return reason(`You have more time than this drive needs (about ${Math.round(driveTimeNeeded(g))} seconds from here). Stopping the clock now only saves time for their answer if you score.`, "timeouts");
  if (call === "spike" && !g.running) return reason("A spike gains nothing with the clock already stopped, and throws away a down.", "spike");
  if (call === "spike" && m > 0)
    return reason(`A spike stops the clock for them and throws away a down: it gives back the ${BURN} seconds you could have run off before the snap.`, "running-out-the-clock");
  if (m > 0 && canKneelOut(g) && call !== "kneel") {
    return reason(
      call === "run"
        ? "Kneel-downs end the game from here. A run does too, but risks a fumble for nothing."
        : call === "fg"
          ? "Kneel-downs end the game from here. A field-goal try only gives them a chance: a miss or a block hands them the ball with time left."
          : "Kneel-downs end the game from here. A pass only gives them a chance: an incompletion stops the clock, and a turnover hands them the ball.",
      "running-out-the-clock",
    );
  }
  if (call === "kneel" && m <= 0 && !fgPlay(g)) return reason("Kneeling gives up a down and time you need.", "clock-rules");
  if (call === "pass-deep" && m <= 0) {
    const coverage = coverageFor(g, g.clock);
    const odds = passOdds(coverage, "pass-deep", g.ytg);
    if (g.ytg <= RED_ZONE)
      return reason(`From the ${g.ytg} there's no room to throw deep: an end-zone throw scores only about ${pct(odds.complete)} of the time. Shorter throws and runs were the better bet.`, "red-zone");
    return coverage === "prevent"
      ? reason(`They're ahead late and playing prevent, with safeties deep: a deep pass completes only about ${pct(odds.complete)} of the time and is picked off about ${pct(odds.interception)}. Shorter throws were the better bet.`, "prevent-defense")
      : reason(`A deep pass completes only about ${pct(odds.complete)} of the time and risks a sack or an interception; shorter throws were the better bet here.`, "play-model");
  }
  if ((call === "pass-middle" || call === "pass-sideline" || call === "pass-deep" || call === "pass-play-action" || call === "pass-screen") && m > 0)
    return reason("A pass risks an incompletion that stops the clock for them, and an interception.", "running-out-the-clock");
  if (call === "fg" && fgPlay(g) && g.down < 4 && g.clock > 20)
    return reason("Kicking now leaves them time to answer. Drain the clock first.", "field-goal-endgame");
  if (call === "punt") {
    return reason(
      m <= 0 ? "Punting gives the ball back when you need to score." : "Punting gives the ball back when you still had a good chance to keep it.",
      "fourth-down",
    );
  }
  if ((call === "run" || call === "kneel") && m < 0 && !fgPlay(g) && g.clock <= 120)
    return reason("A play that ends in bounds keeps the clock running when you're short on time.", "clock-rules");
  return null;
}

/**
 * With the clock already stopped, a timeout spends a timeout and a spike spends a down, and neither
 * gains anything. Any cost at all makes them a mistake, even one too small to otherwise count.
 */
const wastesResource = (g: GameState, call: PlayCall) =>
  (call === "timeout" || call === "spike") && (!g.running || marginFor(g, g.poss) > 0);

/**
 * A timeout called with the clock running and more time than the drive needs. It isn't wasted the way one
 * with the clock stopped is, but it's never the best call: the time it saves is time for their answer, and
 * you could always call it later.
 */
export const earlyTimeout = (g: GameState) => g.running && hasTimeToSpare(g);

/**
 * The smallest gap between two calls' estimates that's more than the simulation's own noise, for `games`
 * simulated per call: twice the standard error of the difference between two win probabilities near `p`
 * (at least CLOSE). At 3,000 games and 30% that's about 2.4 points; at 20,000 it would be 0.9, under CLOSE.
 */
export function noiseMargin(p: number, games: number): number {
  return Math.max(CLOSE, 2 * Math.sqrt((2 * p * (1 - p)) / games));
}

/**
 * The verdict on a call. A call is only "worse" when it trails the best by more than `closeWithin`: by
 * default CLOSE, or the simulation's noise margin when the values are estimates that can't tell a smaller
 * gap apart (see `noiseMargin`).
 */
export function explain(
  g: GameState, chosen: PlayCall, values: CallValues, cfg: SimConfig = DEFAULT_SIM, closeWithin: number = CLOSE,
): Explanation {
  const { call: best, wp: top } = bestScored(values);
  const mine = wpOf(values, chosen);
  const wasted = wastesResource(g, chosen);
  const early = chosen === "timeout" && earlyTimeout(g);
  // Ahead with kneel-downs that end the game: only the kneel is best. Every other call can only add risk,
  // even when the simulations round them all to 100%.
  if (marginFor(g, g.poss) > 0 && chosen !== "kneel" && canKneelOut(g)) {
    const kneel = values.find((v) => v.call === "kneel")?.wp ?? top;
    const why = [whyBest(g, "kneel", cfg), ...(whyNot(g, chosen) ? [whyNot(g, chosen)!] : [])];
    // When both round to the same percentage, the numbers don't say anything: say what the kneel does instead.
    const odds = pct(kneel) === pct(mine) ? "it ends the game" : `${pct(kneel)} vs your ${pct(mine)}`;
    return chosen === "run"
      ? { verdict: "close", headline: `Close: Kneel was the safer call (${pct(kneel) === pct(mine) ? "it ends the game with no risk" : `${pct(kneel)} vs ${pct(mine)}`})`, reasons: why }
      : { verdict: "worse", headline: `Better call: Kneel (${odds})`, reasons: why };
  }
  if (chosen === best || (top - mine <= TIE && !wasted && !early)) {
    return { verdict: "best", headline: `Best call: ${PLAY_CALLS[chosen]} (${pct(mine)})`, reasons: [whyBest(g, chosen, cfg)] };
  }
  const reasons = [whyBest(g, best, cfg)];
  const not = whyNot(g, chosen);
  if (not) reasons.push(not);
  if (wasted) {
    const what = chosen === "timeout" ? "Your timeout" : "Your spike";
    return {
      verdict: "worse",
      headline: g.running
        ? `Better call: ${PLAY_CALLS[best]} (${pct(top)}). ${what} stopped the clock for them.`
        : `Better call: ${PLAY_CALLS[best]} (${pct(top)}). ${what} gained nothing: the clock was already stopped.`,
      reasons,
    };
  }
  if (top - mine <= closeWithin) {
    return {
      verdict: "close",
      headline: `Close: ${PLAY_CALLS[best]} was about as good (${pct(top)} vs ${pct(mine)})`,
      reasons,
    };
  }
  return {
    verdict: "worse",
    headline: `Better call: ${PLAY_CALLS[best]} (${pct(top)} vs your ${pct(mine)})`,
    reasons,
  };
}

// ---------------------------------------------------------------------------
// Game recap

export interface Swing<T> {
  /** The step, e.g. a call in the play log. */
  item: T;
  /** Its position in the drive, from 0. */
  index: number;
  before: number;
  after: number;
  /** Win probability added (negative if lost). */
  added: number;
}

/**
 * The step with the largest win probability added or lost. Each step runs from its own `before` to
 * the next step's, and the last one runs to `final`.
 */
export function biggestSwing<T>(steps: readonly { item: T; before: number }[], final: number): Swing<T> | null {
  let biggest: Swing<T> | null = null;
  steps.forEach(({ item, before }, index) => {
    const after = steps[index + 1]?.before ?? final;
    const added = after - before;
    if (biggest === null || Math.abs(added) > Math.abs(biggest.added)) biggest = { item, index, before, after, added };
  });
  return biggest;
}

// ---------------------------------------------------------------------------
// Recommending a call without playing it

export interface Recommendation {
  best: PlayCall;
  /** Your win probability with the best call. */
  wp: number;
  /** Why the best call is best. */
  reason: Reason;
  /** The next-best call, and whether it's close enough to be a coin flip. */
  runnerUp: { call: PlayCall; wp: number; close: boolean } | null;
  /** Calls to avoid here, with why, worst first. */
  avoid: Avoid[];
}

export interface Avoid {
  call: PlayCall;
  wp: number;
  reason: Reason;
}

export function recommend(g: GameState, values: CallValues, cfg: SimConfig = DEFAULT_SIM, closeWithin: number = CLOSE): Recommendation {
  const { call: best, wp } = bestScored(values);
  const others = ranked(values).filter((v) => v.call !== best);
  // With the clock already stopped, a timeout or a spike only postpones the decision, so neither is a real alternative.
  const next = others.find((v) => g.running || (v.call !== "timeout" && v.call !== "spike"));
  const runnerUp = next === undefined ? null : { call: next.call, wp: next.wp, close: wp - next.wp <= closeWithin };
  const avoid = others
    // A timeout or spike with the clock stopped is always worth avoiding, even when it costs nothing
    // measurable: it gives up something and can't gain anything.
    .filter((v) => wp - v.wp >= AVOID || wastesResource(g, v.call) || (v.call === "timeout" && earlyTimeout(g)))
    .flatMap((v): Avoid[] => {
      const why = whyNot(g, v.call);
      return why === null ? [] : [{ call: v.call, wp: v.wp, reason: why }];
    })
    .sort((a, b) => a.wp - b.wp);
  return { best, wp, reason: whyBest(g, best, cfg), runnerUp, avoid };
}

// ---------------------------------------------------------------------------
// Kick or go for two

export interface TryValues {
  /** Your win probability if the try adds 0, 1 or 2 points, everything after it simulated. */
  byPoints: readonly [number, number, number];
  /** Your win probability with each call, at the league's success rates. */
  kick: number;
  two: number;
  best: TryCall;
  /** The two-point success rate at which going for two is as good as kicking (null: two never helps). */
  breakEven: number | null;
}

/**
 * Kick or go for two, for you, just after you score a touchdown. `g.margin` already counts the 6
 * points, `g.clock` is the time left after the touchdown, and the kickoff follows the try.
 */
export function evaluateTry(g: GameState, n = 3000, cfg: SimConfig = DEFAULT_SIM): TryValues {
  const scored: GameState = { ...g, poss: "you", running: false };
  const wpAfter = (extra: number) => {
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const r = streamFor(i);
      const s = kickoff({ ...scored, margin: scored.margin + extra }, scored.clock, "", r).state;
      sum += s.over ? (s.margin > 0 ? 1 : s.margin < 0 ? 0 : 0.5) : playOut(s, r, cfg);
    }
    return sum / n;
  };
  const byPoints = [wpAfter(0), wpAfter(1), wpAfter(2)] as const;
  const [w0, w1, w2] = byPoints;
  const kick = w0 + SPECIAL_TEAMS.extraPoint * (w1 - w0);
  const two = w0 + SPECIAL_TEAMS.twoPoint * (w2 - w0);
  const gain = w2 - w0;
  const breakEven = gain > 0 ? Math.min(1, (SPECIAL_TEAMS.extraPoint * (w1 - w0)) / gain) : null;
  return { byPoints, kick, two, best: two > kick ? "two" : "kick", breakEven };
}
