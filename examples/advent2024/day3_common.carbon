// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/3

library "day3_common";

import library "io_utils";

// Reads "mul(a,b)", and returns (true, a, b).
// On error, stops before the first invalid character and returns (false, 0, 0).
// TODO: -> Optional((i32, i32))
fn ReadMul() -> (bool, i32, i32) {
  var a: i32;
  var b: i32;
  if (ConsumeChar(0x6D) and ConsumeChar(0x75) and ConsumeChar(0x6C) and
      ConsumeChar(0x28) and ReadInt(&a) and ConsumeChar(0x2C) and
      ReadInt(&b) and ConsumeChar(0x29)) {
    return (true, a, b);
  }
  return (false, 0, 0);
}

// Reads "do()" or "don't()", and returns (true, was_do).
// On error, stops before the first invalid character and returns (false, false).
fn ReadDoOrDont() -> (bool, bool) {
  // "do"
  if (not ConsumeChar(0x64) or not ConsumeChar(0x6F)) {
    return (false, false);
  }

  var do: bool = true;
  // "n't"
  if (ConsumeChar(0x6E)) {
    if (not ConsumeChar(0x27) or not ConsumeChar(0x74)) {
      return (false, false);
    }
    do = false;
  }

  // "()"
  if (not ConsumeChar(0x28) or not ConsumeChar(0x29)) {
    return (false, false);
  }
  return (true, do);
}
