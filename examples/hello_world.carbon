// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import Cpp library "cstdio";

fn Run() {
  // TODO: This should be a string, and just print directly.
  let msg: array(Core.Char, 13) =
      ('H', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd', '!', '\n');
  for (c: Core.Char in msg) {
    // TODO: u8 should probably have an implicit cast to i32.
    Cpp.putchar((c as u8) as i32);
  }
}
