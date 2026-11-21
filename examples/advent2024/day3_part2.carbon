// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/3

import Core library "io";

import library "day3_common";
import library "io_utils";

fn Run() {
  var total: i32 = 0;
  var enabled: bool = true;
  while (PeekChar() != CharOrEOF.EOF()) {
    if (PeekChar() == 'm') {
      // TODO: Use `if let` when available.
      let result: Core.Optional((i32, i32)) = ReadMul();
      if (result.HasValue() and enabled) {
        total += result.Get().0 * result.Get().1;
      }
    } else if (PeekChar() == 'd') {
      // TODO: Use `if let` when available.
      let result: Core.Optional(bool) = ReadDoOrDont();
      if (result.HasValue()) {
        enabled = result.Get();
      }
    } else {
      ReadChar();
    }
  }
  Core.Print(total);
}
