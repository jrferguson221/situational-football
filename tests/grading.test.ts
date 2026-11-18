import { describe, expect, it } from "bun:test";
import { CLOSE, bestScored, evaluateCalls, explain, noiseMargin, scoreCallAtTempo, type CallValues, type GameState } from "../src/sim";

describe("grading a call", () => {
  it("only calls a gap a mistake when it's more than the simulation's noise", () => {
    // At 3,000 games and a 30% win probability, two estimates can differ by about 2.4 points by chance.
    expect(noiseMargin(0.3, 3000)).toBeCloseTo(0.0237, 3);
    // With more games, or near a sure thing, the noise is small and CLOSE is the floor.
    expect(noiseMargin(0.3, 20000)).toBe(CLOSE);
    expect(noiseMargin(0.995, 3000)).toBe(CLOSE);
  });

  it("down 4, 1st & 10 at your 42 with 1:56 left: a pass over the middle is about as good as the run", () => {
    // At 20,000 games the run and the pass are within a tenth of a point; at 3,000 the gap can look like 1.7.
    const g: GameState = { clock: 116, running: true, poss: "you", down: 1, toGo: 10, ytg: 58, margin: -4, timeouts: { you: 3, opp: 3 }, over: false };
    const values = evaluateCalls(g, 500);
    const lead = bestScored(values).wp;
    const graded = values.map((v) => (v.call === "pass-middle" || lead - v.wp <= 0.04 ? { ...v, wp: scoreCallAtTempo(g, v.call, "auto", 3000) } : v));
    const [first, ...rest] = graded;
    const values2: CallValues = [first!, ...rest];
    const x = explain(g, "pass-middle", values2, undefined, noiseMargin(bestScored(values2).wp, 3000));
    expect(x.verdict).not.toBe("worse");
  });

  it("still flags a real mistake: a deep pass while protecting a lead", () => {
    const g: GameState = { clock: 90, running: true, poss: "you", down: 2, toGo: 5, ytg: 60, margin: 3, timeouts: { you: 2, opp: 3 }, over: false };
    const values = evaluateCalls(g, 3000);
    expect(explain(g, "pass-deep", values, undefined, noiseMargin(bestScored(values).wp, 3000)).verdict).toBe("worse");
  });
});
