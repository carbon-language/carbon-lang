// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/15

import Core library "io";

import library "day15_common";

fn Run() {
  var g: Grid = Grid.Read();
  ReadAndExecuteActions(ref g);
  Core.Print(g.Signature());
}
