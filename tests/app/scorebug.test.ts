import { describe, expect, it } from "bun:test";
import { fieldMarks, finalBug, stripPercent } from "../../src/Scorebug";

it("places the ball and the line to gain in yards from your own goal line", () => {
  expect(fieldMarks("own", 30, 10)).toEqual({ ball: 30, lineToGain: 40, goal: false });
  expect(fieldMarks("opp", 35, 10)).toEqual({ ball: 65, lineToGain: 75, goal: false });
  // & Goal: the line to gain is the opponent's goal line.
  expect(fieldMarks("opp", 8, 8)).toEqual({ ball: 92, lineToGain: 100, goal: true });
  expect(fieldMarks("opp", 5, 10)).toEqual({ ball: 95, lineToGain: 100, goal: true });
});

it("maps the field onto the strip between two 10-yard end zones", () => {
  expect(stripPercent(0)).toBeCloseTo(100 / 12);
  expect(stripPercent(50)).toBe(50);
  expect(stripPercent(100)).toBeCloseTo(100 - 100 / 12);
});

describe("the final", () => {
  const end = { clock: 0, running: false, down: 2 as const, toGo: 7, margin: -3, timeouts: { you: 0, opp: 1 }, over: true };
  it("puts the ball where it ended, from your end of the field, and says whose it was", () => {
    // Their ball 67 yards from the goal they attack is at their own 33: 33 yards from the goal you attack.
    const theirs = finalBug({ ...end, poss: "opp", ytg: 67 });
    expect(theirs.situation.ball).toMatchObject({ side: "opp", yardLine: 33 });
    expect(theirs.final.oppBall).toBe(true);
    expect(theirs.margin).toBe(-3);
    const yours = finalBug({ ...end, poss: "you", ytg: 65, margin: 3 });
    expect(yours.situation.ball).toMatchObject({ side: "own", yardLine: 35 });
    expect(yours.final.oppBall).toBe(false);
  });
});
