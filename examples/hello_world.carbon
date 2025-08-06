// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import Cpp library "cstdio";

fn Run() {
  var msg: array(Core.Char, 13) =
      ('H', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd', '!', '\n');
  for (c: Core.Char in msg) {
    Cpp.putchar((c as u8) as i32);
  }
}
