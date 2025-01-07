// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/3

import Core library "io";

import library "day3_common";
import library "io_utils";

fn Run() {
  var total: i32 = 0;
  while (PeekChar() != Core.EOF()) {
    if (PeekChar() == 0x6D) {
      // TODO: Use `if let` when available.
      let result: (bool, i32, i32) = ReadMul();
      if (result.0) {
        total += result.1 * result.2;
      }
    } else {
      ReadChar();
    }
  }
  Core.Print(total);
}
