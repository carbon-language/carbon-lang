// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/13

library "day13_common";

import Core library "io";
import library "io_utils";

// Returns m and n so that am + bn = gcd.
fn Euclid(a: i64, b: i64) -> {.m: i64, .n: i64, .gcd: i64} {
  if (a < b) {
    let reverse: {.m: i64, .n: i64, .gcd: i64} = Euclid(b, a);
    return {.m = reverse.n, .n = reverse.m, .gcd = reverse.gcd};
  }
  if (b == 0) {
    return {.m = 1, .n = 0, .gcd = a};
  }
  let next: {.m: i64, .n: i64, .gcd: i64} = Euclid(b, a % b);
  return {.m = next.n, .n = next.m - next.n * (a / b), .gcd = next.gcd};
}

class Machine {
  fn Read() -> Machine {
    returned var me: Machine;
    // "Button A: X+"
    SkipNChars(12);
    ReadInt64(&me.a.0);
    // ", Y+"
    SkipNChars(4);
    ReadInt64(&me.a.1);
    // "\nButton B: X+"
    SkipNChars(13);
    ReadInt64(&me.b.0);
    // ", Y+"
    SkipNChars(4);
    ReadInt64(&me.b.1);
    // "\nPrize: X="
    SkipNChars(10);
    ReadInt64(&me.prize.0);
    // ", Y="
    SkipNChars(4);
    ReadInt64(&me.prize.1);
    SkipNewline();
    return var;
  }

  var a: (i64, i64);
  var b: (i64, i64);
  var prize: (i64, i64);
}

// Set of solutions to 'm a + n b = c'.
class BezoutSolutionSet {
  fn Make(a: i64, b: i64, c: i64) -> BezoutSolutionSet {
    var e: {.m: i64, .n: i64, .gcd: i64} = Euclid(a, b);
    if (c % e.gcd != 0) {
      // Impossible.
      return {.m0 = -1, .n0 = -1, .m_step = -1, .n_step = -1};
    }

    // Find an initial solution. Note that m and n might be negative.
    let num_gcds: i64 = c / e.gcd;
    e.m = e.m * num_gcds;
    e.n = e.n * num_gcds;

    // Pick the smallest positive m we can.
    let a_over_gcd: i64 = a / e.gcd;
    let b_over_gcd: i64 = b / e.gcd;
    var adj: i64 = 0;
    // This is e.m / b rounded towards -inf.
    // TODO: Should there be a way of expressing this directly?
    if (e.m < 0) {
      adj = (e.m - b_over_gcd + 1) / b_over_gcd;
    } else {
      adj = e.m / b_over_gcd;
    }
    e.m = e.m - adj * b_over_gcd;
    e.n = e.n + adj * a_over_gcd;
    return {.m0 = e.m, .n0 = e.n,
            .m_step = b_over_gcd, .n_step = -a_over_gcd};
  }

  fn Valid[self: Self]() -> bool {
    return self.m_step >= 0;
  }

  fn Solution[self: Self](k: i64) -> (i64, i64) {
    return (self.m0 + k * self.m_step, self.n0 + k * self.n_step);
  }

  // m_k * a + n_k * b == c, where:
  //   m_k = m0 + k * m_step
  //   n_k = n0 * k * n_step
  // m0 is the minimum non-negative m value, and n_step is negative.
  var m0: i64;
  var n0: i64;
  var m_step: i64;
  var n_step: i64;
}

// Given two sets of points s and t, find the intersection in the first quadrant
// with the minimum x coordinate. Returns the intersection point, or (-1, -1) if
// there is no intersection.
fn FirstIntersection(s: BezoutSolutionSet, t: BezoutSolutionSet) -> (i64, i64) {
  if (not s.Valid() or not t.Valid()) {
    // One of the sets is empty.
    return (-1, -1);
  }

  // Easy case: lines meet at a point. This happens unless the lines are
  // parallel.
  let d: i64 = s.m_step * t.n_step - t.m_step * s.n_step;
  if (d != 0) {
    let u: i64 = (t.m0 - s.m0) * t.n_step - (t.n0 - s.n0) * t.m_step;
    if (u % d != 0) {
      // Lines don't meet at an integer point.
      return (-1, -1);
    }
    let j: i64 = u / d;
    if (j < 0 or s.n0 + j * s.n_step < 0) {
      // Lines don't meet in first quadrant.
      return (-1, -1);
    }
    return s.Solution(j);
  }

  // Hard case: lines are parallel. We also get here if either "line" is a
  // point, but that doesn't happen in this exercise. We know all integer points
  // on the line are solutions, so the first intersection is the greater of the
  // first point in s and the first point in t, if that point is on both lines.
  if (s.m0 < t.m0) {
    // (t.m0, t.n0) is the solution if it's in s.
    if ((t.m0 - s.m0) % s.m_step == 0 and
        (t.n0 - s.n0) % s.n_step == 0) {
      return (t.m0, t.n0);
    }
  } else {
    // (s.m0, s.n0) is the solution if it's in t.
    if ((s.m0 - t.m0) % t.m_step == 0 and
        (s.n0 - t.n0) % t.n_step == 0) {
      return (s.m0, s.n0);
    }
  }
  return (-1, -1);
}

fn CostIfPossible(m: Machine) -> i64 {
  let x: BezoutSolutionSet = BezoutSolutionSet.Make(m.a.0, m.b.0, m.prize.0);
  let y: BezoutSolutionSet = BezoutSolutionSet.Make(m.a.1, m.b.1, m.prize.1);
  let ab: (i64, i64) = FirstIntersection(x, y);
  if (ab.0 == -1) { return 0; }
  return ab.0 * 3 + ab.1;
}
