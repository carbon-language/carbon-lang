// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/14

import Core library "io";

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

fn Run() {
  var r: [Robot; 500];
  var i: i32 = 0;
  while (i < 500) {
    r[i] = Robot.Read();
    ++i;
  }

  let first_match_x: i32 = 28;
  let first_match_y: i32 = 84;

  var t: i32 = CRT(first_match_x, first_match_y, 101, 103);
  while (true) {
    Core.Print(t);
    var display: [[i32; 101]; 103];

    var y: i32 = 0;
    while (y < 103) {
      var x: i32 = 0;
      while (x < 101) {
        display[y][x] = 0x2E;
        ++x;
      }
      ++y;
    }

    i = 0;
    while (i < 500) {
      let xy: (i32, i32) = r[i].Pos(t);
      display[xy.1][xy.0] = 0x23;
      ++i;
    }

    y = 0;
    while (y < 103) {
      var x: i32 = 0;
      while (x < 101) {
        Core.PrintChar(display[y][x]);
        ++x;
      }
      Core.PrintChar(0xA);
      ++y;
    }
    Core.PrintChar(0xA);

    t += 101 * 103;
  }
}
