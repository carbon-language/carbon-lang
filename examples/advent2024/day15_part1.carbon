// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import Core library "io";

import library "day15_common";
import library "io_utils";

fn Run() {
  var g: Grid = Grid.Read();

  while (true) {
    var m: i32 = ReadChar();
    if (m == 0x5E) {
      g.Move((0, -1));
    } else if (m == 0x3C) {
      g.Move((-1, 0));
    } else if (m == 0x3E) {
      g.Move((1, 0));
    } else if (m == 0x76) {
      g.Move((0, 1));
    } else if (m == -1) {
      break;
    }
  }
  Core.Print(g.Signature());
}
