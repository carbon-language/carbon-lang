// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/12

import Core library "io";
import Core library "range";

import library "day12_common";
import library "io_utils";

fn CountExtensions(map: Map, regions: DisjointSetForest*) -> array(i32, 140 * 140) {
  returned var extensions: array(i32, 140 * 140);
  for (i: i32 in Core.Range(140 * 140)) {
    extensions[i] = 0;
  }

  var ext: array({.same: (i32, i32), .adj: (i32, i32)}, 4) = (
    {.same = (-1, 0), .adj = (0, -1)},
    {.same = (-1, 0), .adj = (0, 1)},
    {.same = (0, -1), .adj = (-1, 0)},
    {.same = (0, -1), .adj = (1, 0)},
  );

  for (x: i32 in Core.Range(140)) {
    for (y: i32 in Core.Range(140)) {
      let kind: i32 = map.At(x, y);
      for (e: i32 in Core.Range(4)) {
        if (map.At(x + ext[e].same.0, y + ext[e].same.1) == kind and
            map.At(x + ext[e].adj.0, y + ext[e].adj.1) != kind and
            map.At(x + ext[e].same.0 + ext[e].adj.0,
                   y + ext[e].same.1 + ext[e].adj.1) != kind) {
          ++extensions[regions->Lookup(y * 140 + x)];
        }
      }
    }
  }
  return var;
}

fn Run() {
  var map: Map = Map.Read();
  var regions: DisjointSetForest = MakeRegions(map);
  var ext: array(i32, 140 * 140) = CountExtensions(map, &regions);

  var total: i32 = 0;

  for (i: i32 in Core.Range(140 * 140)) {
    if (regions.Lookup(i) == i) {
      let area: i32 = regions.Weight(i);
      let internal_edges: i32 = regions.Unions(i);
      let extensions: i32 = ext[i];
      let perimeter: i32 = area * 4 - internal_edges * 2 - extensions;
      total += area * perimeter;
    }
  }

  Core.Print(total);
}
