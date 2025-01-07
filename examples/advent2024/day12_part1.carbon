// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/12

import Core library "io";

import library "day12_common";
import library "io_utils";

fn Run() {
  var map: Map = Map.Read();
  var regions: DisjointSetForest = MakeRegions(map);

  var total: i32 = 0;

  var i: i32 = 0;
  while (i < 140 * 140) {
    if (regions.Lookup(i) == i) {
      let area: i32 = regions.Weight(i);
      let internal_edges: i32 = regions.Unions(i);
      let perimeter: i32 = area * 4 - internal_edges * 2;
      total += area * perimeter;
    }
    ++i;
  }

  Core.Print(total);
}
