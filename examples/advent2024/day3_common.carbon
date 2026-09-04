// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/3

library "day3_common";

import library "io_utils";

// Reads "mul(a,b)", and returns Some((a, b)).
// On error, stops before the first invalid character and returns None.
fn ReadMul() -> Core.Optional((i32, i32)) {
  var a: i32;
  var b: i32;
  if (ConsumeChar('m') and ConsumeChar('u') and ConsumeChar('l') and
      ConsumeChar('(') and ReadInt(ref a) and ConsumeChar(',') and
      ReadInt(ref b) and ConsumeChar(')')) {
    return (a, b);
  }
  return Core.Optional((i32, i32)).None();
}

// Reads "do()" or "don't()", and returns Some(was_do).
// On error, stops before the first invalid character and returns None.
fn ReadDoOrDont() -> Core.Optional(bool) {
  // "do"
  if (not ConsumeChar('d') or not ConsumeChar('o')) {
    return Core.Optional(bool).None();
  }

  var do: bool = true;
  // "n't"
  if (ConsumeChar('n')) {
    if (not ConsumeChar('\'') or not ConsumeChar('t')) {
      return Core.Optional(bool).None();
    }
    do = false;
  }

  // "()"
  if (not ConsumeChar('(') or not ConsumeChar(')')) {
    return Core.Optional(bool).None();
  }
  return do;
}
