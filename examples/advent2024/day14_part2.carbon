// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/14

import Core library "io";
import Core library "range";

import library "day14_common";
import library "io_utils";

// Returns m and n so that am + bn = gcd.
fn Euclid(a: i32, b: i32) -> {.m: i32, .n: i32, .gcd: i32} {
  if (a < b) {
    let reverse: {.m: i32, .n: i32, .gcd: i32} = Euclid(b, a);
    return {.m = reverse.n, .n = reverse.m, .gcd = reverse.gcd};
  }
  if (b == 0) {
    return {.m = 1, .n = 0, .gcd = a};
  }
  let next: {.m: i32, .n: i32, .gcd: i32} = Euclid(b, a % b);
  return {.m = next.n, .n = next.m - next.n * (a / b), .gcd = next.gcd};
}

// Uses Chinese remainder theorem to find the least positive x that solves
//   x = a mod p
//   x = b mod q
// modulo pq / gcd(p,q), if one exists (if a = b mod gcd(p,q)).
fn CRT(a: i32, b: i32, p: i32, q: i32) -> i32 {
  let e: {.m: i32, .n: i32, .gcd: i32} = Euclid(p, q);
  let x: i32 = a * (q / e.gcd) * e.n + b * (p / e.gcd) * e.m;
  return Mod(x, p * (q / e.gcd));
}

fn PrintState(r: array(Robot, 500), t: i32) {
  var display: array(array(char, 101), 103);

  for (y: i32 in Core.Range(103)) {
    for (x: i32 in Core.Range(101)) {
      display[y][x] = '.';
    }
  }

  for (ri: Robot in r) {
    let xy: (i32, i32) = ri.Pos(t);
    display[xy.1][xy.0] = '#';
  }

  for (y: i32 in Core.Range(103)) {
    for (x: i32 in Core.Range(101)) {
      Core.PrintChar(display[y][x]);
    }
    Core.PrintChar('\n');
  }
}

fn Run() {
  var r: array(Robot, 500);
  for (i: i32 in Core.Range(500)) {
    r[i] = Robot.Read();
  }

  let first_match_x: i32 = 28;
  let first_match_y: i32 = 84;

  var t: i32 = CRT(first_match_x, first_match_y, 101, 103);
  Core.Print(t);
  PrintState(r, t);
}
