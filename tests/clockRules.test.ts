// The rule-of-thumb clock management: when a team stops the clock, lets it run, kicks, kneels or goes for
// it. Each rule here was kept because the team following it won more often (see the comments in sim.ts).
import { describe, expect, it } from "bun:test";
import {
  availableCalls, bestScored, driveTimeNeeded, earlyTimeout, evaluateCallsPrecisely, explain, hasTimeToSpare, noiseMargin, policy,
  recommend, seededRng, snapDelay, step, winProbability, wpOf, GRADING, type GameState,
} from "../src/sim";

const g0: GameState = { clock: 60, running: true, poss: "you", down: 1, toGo: 10, ytg: 50, margin: -4, timeouts: { you: 2, opp: 2 }, over: false };
const inBounds = { yards: 5, seconds: 6, outOfBounds: false, incomplete: false };

describe("time to spare", () => {
  // Down 7 with 1:01 left, 2nd & 1 at the 5, three timeouts: far more time than four plays need.
  const atTheFive: GameState = { ...g0, clock: 61, down: 2, toGo: 1, ytg: 5, margin: -7, timeouts: { you: 3, opp: 0 } };
  // Down 4 at your own 40 with 1:00 left: 60 yards to go, and every second counts.
  const longWay: GameState = { ...g0, ytg: 60 };

  it("a drive needs about 25 seconds plus 1.5 a yard: to the end zone, or to the 20 when a field goal is enough", () => {
    expect(driveTimeNeeded(atTheFive)).toBe(32.5);
    expect(driveTimeNeeded(longWay)).toBe(115);
    expect(driveTimeNeeded({ ...longWay, margin: -3 })).toBe(85);
    expect(hasTimeToSpare(atTheFive)).toBe(true);
    expect(hasTimeToSpare(longWay)).toBe(false);
  });

  it("with time to spare, the rule of thumb keeps its timeouts and lets the spare time run", () => {
    expect(policy(atTheFive)).not.toBe("timeout");
    expect(snapDelay(atTheFive, "run")).toBe(28); // down to the 33 seconds the drive needs
    expect(policy(longWay)).toBe("timeout");
    expect(snapDelay(longWay, "pass-middle")).toBe(13);
  });

  it("an extra timeout never makes a trailing team worse off", () => {
    // Each of these lost 3 to 5 points with a timeout in hand under the old rule (stop the clock whenever
    // it's running under 2:00).
    for (const s of [
      { ...g0, clock: 50, ytg: 5, margin: -7, timeouts: { you: 0, opp: 3 } },
      { ...g0, clock: 90, ytg: 5, margin: -4, timeouts: { you: 0, opp: 0 } },
      { ...g0, clock: 90, ytg: 15, margin: -4, timeouts: { you: 0, opp: 3 } },
      { ...g0, clock: 119, ytg: 50, margin: -1, timeouts: { you: 0, opp: 3 } },
    ]) {
      const without = winProbability(s, 3000);
      const withOne = winProbability({ ...s, timeouts: { ...s.timeouts, you: 1 } }, 3000);
      expect(withOne).toBeGreaterThan(without - 0.01);
    }
  });

  it("a timeout called with time to spare is never the best call, and says why", () => {
    expect(earlyTimeout(atTheFive)).toBe(true);
    const values = evaluateCallsPrecisely(atTheFive);
    expect(bestScored(values).call).not.toBe("timeout");
    const x = explain(atTheFive, "timeout", values, undefined, noiseMargin(bestScored(values).wp, GRADING.precise));
    expect(x.verdict).not.toBe("best");
    expect(x.reasons.map((r) => r.text).join(" ")).toContain("more time than this drive needs");
    expect(recommend(atTheFive, values).avoid.map((a) => a.call)).toContain("timeout");
    // When time is tight, a timeout is a fine call again.
    expect(earlyTimeout(longWay)).toBe(false);
  });
});

describe("stopping the clock when it's tight", () => {
  it("calls the timeout before anything else when there's no time to snap a play", () => {
    // 12 seconds, the clock running: a hurry-up snap takes 13.
    const late: GameState = { ...g0, clock: 12, ytg: 3, toGo: 3 };
    expect(policy(late)).toBe("timeout");
    expect(policy({ ...late, down: 4 })).toBe("timeout");
    expect(policy({ ...late, timeouts: { you: 0, opp: 2 } })).toBe("spike");
  });

  it("doesn't stop the clock before a punt", () => {
    expect(policy({ ...g0, margin: 0, ytg: 80, clock: 100, down: 4, toGo: 9 })).toBe("punt");
  });
});

describe("kicking last", () => {
  const range: GameState = { ...g0, margin: -1, ytg: 8, toGo: 8, clock: 27, timeouts: { you: 1, opp: 0 } };

  it("on 4th down in range, lets the clock run down to 0:03 before the timeout", () => {
    expect(policy({ ...range, down: 4 })).toBe("timeout");
    expect(step({ ...range, down: 4 }, "timeout", seededRng(1)).state.clock).toBe(3);
    // From 1:00 it can only run one play clock off.
    expect(step({ ...range, down: 4, clock: 60 }, "timeout", seededRng(1)).state.clock).toBe(21);
  });

  it("without a timeout, lets the clock run down before the kick, and the kick ends the game", () => {
    const noTimeout = { ...range, down: 4 as const, timeouts: { you: 0, opp: 0 } };
    expect(policy(noTimeout)).toBe("fg");
    expect(snapDelay(noTimeout, "fg")).toBe(24); // 0:27 down to 0:03
    expect(step(noTimeout, "fg", seededRng(1)).state.over).toBe(true);
  });

  it("with too little time to get the kicking unit on, spikes it first", () => {
    expect(policy({ ...range, down: 2, clock: 14, timeouts: { you: 0, opp: 0 } })).toBe("spike");
  });

  it("with the clock stopped and downs to spare, uses the time before the kick: a kneel in close, a run from further out", () => {
    const stopped = { ...range, running: false, clock: 20, timeouts: { you: 0, opp: 2 } };
    expect(policy(stopped)).toBe("kneel");
    // A run takes longer than a kneel: without a timeout it needs 0:23, the play plus 15 to get the unit on.
    expect(policy({ ...stopped, ytg: 30, clock: 25 })).toBe("run");
    expect(policy({ ...stopped, ytg: 30 })).toBe("fg");
    // No time for it: kick.
    expect(policy({ ...stopped, clock: 10 })).toBe("fg");
  });

  it("paces a kneel in field-goal range like a run, so the kick still fits", () => {
    const g = { ...range, clock: 45 };
    expect(availableCalls(g)).toContain("kneel");
    expect(snapDelay(g, "kneel")).toBe(snapDelay(g, "run"));
    expect(snapDelay(g, "kneel")).toBeLessThan(39);
  });

  it("down 3 inside the 10, plays for the touchdown instead of draining for the tie", () => {
    const down3 = { ...g0, margin: -3, clock: 50, running: false };
    expect(policy({ ...down3, ytg: 6, toGo: 6 })).toBe("pass-middle");
    expect(policy({ ...down3, ytg: 25 })).toBe("run");
  });
});

describe("4th down with a kick that ties or takes the lead", () => {
  const fourth: GameState = { ...g0, margin: 0, down: 4, toGo: 1, ytg: 30, running: false };

  it("goes for it on short yardage when they'd have time to answer the kick, and kicks when they wouldn't", () => {
    expect(policy({ ...fourth, clock: 120 })).toBe("run");
    expect(policy({ ...fourth, clock: 20 })).toBe("fg");
    expect(policy({ ...fourth, clock: 120, toGo: 9 })).toBe("fg");
    // The closer the kick, the shorter the distance worth going for.
    expect(policy({ ...fourth, clock: 120, ytg: 5, toGo: 2 })).toBe("run");
    expect(policy({ ...fourth, clock: 120, ytg: 5, toGo: 3 })).toBe("fg");
  });

  it("ahead by a field goal or less, goes for it to end the game", () => {
    expect(policy({ ...fourth, margin: 3, clock: 90, ytg: 40, toGo: 2, timeouts: { you: 2, opp: 2 } })).toBe("run");
    expect(policy({ ...fourth, margin: 3, clock: 90, ytg: 40, toGo: 12, timeouts: { you: 2, opp: 2 } })).toBe("punt");
  });

  it("a tied team punts rather than heaving it on 4th down from its own end", () => {
    expect(policy({ ...fourth, clock: 12, ytg: 80, toGo: 6 })).toBe("punt");
  });
});

describe("the defense's timeouts", () => {
  it("don't rescue an offense that's out of timeouts and out of time to get its kick off", () => {
    // Down 1 at the 39, no timeouts: an 8-yard catch in bounds leaves 0:14 with the clock running.
    const s: GameState = { ...g0, margin: -1, clock: 20, running: false, ytg: 39, timeouts: { you: 0, opp: 2 } };
    const after = step(s, "pass-middle", seededRng(1), undefined, { ...inBounds, yards: 8 }).state;
    expect(after).toMatchObject({ clock: 14, running: true, timeouts: { you: 0, opp: 2 } });
  });

  it("aren't spent when they can't leave time to answer with", () => {
    // 0:25 left: the offense has a timeout and three downs to let the clock run before its kick.
    const s: GameState = { ...g0, margin: -1, clock: 25, running: false, ytg: 30, timeouts: { you: 3, opp: 3 } };
    expect(step(s, "run", seededRng(1), undefined, inBounds).state.timeouts.opp).toBe(3);
    // With 1:40 left they can: the defense stops the clock after the run.
    const early = step({ ...s, clock: 100 }, "run", seededRng(1), undefined, inBounds).state;
    expect(early).toMatchObject({ running: false, timeouts: { you: 3, opp: 2 } });
  });

  it("save time for an answer when a team that needs a touchdown is in the red zone with time to spare", () => {
    const s: GameState = { ...g0, margin: -4, clock: 100, running: false, ytg: 15, timeouts: { you: 2, opp: 2 } };
    expect(step(s, "run", seededRng(1), undefined, inBounds).state).toMatchObject({ clock: 94, running: false, timeouts: { you: 2, opp: 1 } });
    // Not when the offense is short of time: then the running clock is the defense's friend.
    expect(step({ ...s, clock: 30 }, "run", seededRng(1), undefined, inBounds).state.running).toBe(true);
  });
});

describe("precise scoring", () => {
  it("scores every available call, the same way every time", () => {
    const a = evaluateCallsPrecisely(g0);
    expect(a.map((v) => v.call).sort()).toEqual([...availableCalls(g0)].sort());
    expect(evaluateCallsPrecisely(g0)).toEqual(a);
    expect(wpOf(a, "pass-middle")).toBeGreaterThan(0);
  });
});

describe("grading a call when you're ahead", () => {
  // Up 3 with 1:10 left at the opponent's 45, the clock running, and they're out of timeouts: kneel-downs
  // end the game, so every sensible call rounds to 100%.
  const kneelOut: GameState = { ...g0, clock: 70, ytg: 45, margin: 3, timeouts: { you: 3, opp: 0 } };
  const grade = (g: GameState, call: Parameters<typeof explain>[1]) => {
    const values = evaluateCallsPrecisely(g);
    return explain(g, call, values, undefined, noiseMargin(bestScored(values).wp, GRADING.precise));
  };

  it("when kneels end the game, only the kneel is best: a run is close, anything else a mistake", () => {
    expect(grade(kneelOut, "kneel").verdict).toBe("best");
    expect(grade(kneelOut, "run").verdict).toBe("close");
    for (const call of ["spike", "timeout", "pass-deep", "pass-screen", "fg"] as const) expect(grade(kneelOut, call).verdict).toBe("worse");
    // The headline doesn't claim "100% vs your 100%".
    expect(grade(kneelOut, "spike").headline).toBe("Better call: Kneel (it ends the game)");
  });

  it("a spike or a timeout with the clock running is a mistake, even when it ties", () => {
    const notOver: GameState = { ...kneelOut, clock: 150, timeouts: { you: 3, opp: 3 } };
    for (const call of ["spike", "timeout"] as const) {
      const x = grade(notOver, call);
      expect(x.verdict).toBe("worse");
      expect(x.headline).toContain("stopped the clock for them");
    }
    expect(recommend(kneelOut, evaluateCallsPrecisely(kneelOut)).avoid.map((a) => a.call)).toEqual(expect.arrayContaining(["spike", "timeout"]));
  });
});

it("ahead on 4th down, the reasons for going for it and against punting fit a team that's ahead", () => {
  // Up 1 on 4th & 15 at your own 32 with 3:51 left, 1 timeout to their 3: punt and a deep shot are close.
  const g: GameState = { ...g0, clock: 231, running: false, down: 4, toGo: 15, ytg: 68, margin: 1, timeouts: { you: 1, opp: 3 } };
  const values = evaluateCallsPrecisely(g);
  const x = explain(g, "punt", values, undefined, noiseMargin(bestScored(values).wp, GRADING.precise));
  expect(x.verdict).not.toBe("worse");
  const text = x.reasons.map((r) => r.text).join(" ");
  expect(text).not.toContain("You need a big play");
  expect(text).toContain("3 timeouts to drive for the field goal they need");
  expect(text).toContain("A deep pass completes");
});
