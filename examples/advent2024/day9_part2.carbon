// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/9

import Core library "io";

import library "day9_common";
import library "io_utils";

fn Run() {
  var list: SectorList = SectorList.Read();
  list.DefragFiles();
  PrintInt64(list.Checksum());
}
