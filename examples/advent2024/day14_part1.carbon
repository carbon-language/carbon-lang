// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/14

import Core library "io";

import library "day14_common";
import library "io_utils";

fn Run() {
  var q: array(array(i32, 3), 3) = ((0,0,0), (0,0,0), (0,0,0));
  while (PeekChar() != CharOrEOF.EOF()) {
    var r: Robot = Robot.Read();
    let xy: (i32, i32) = r.Pos(100);
    ++q[if xy.0 < 50 then 0 else if xy.0 > 50 then 2 else 1]
       [if xy.1 < 51 then 0 else if xy.1 > 51 then 2 else 1];
  }
  Core.Print(q[0][0] * q[0][2] * q[2][0] * q[2][2]);
}
