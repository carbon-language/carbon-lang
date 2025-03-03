// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/10

import Core library "io";

import library "day10_common";
import library "io_utils";

class PathsToTop {
  fn Make(terrain: Terrain) -> PathsToTop {
    returned var me: PathsToTop;
    var y: i32 = 0;
    while (y < 43) {
      var x: i32 = 0;
      while (x < 43) {
        // TODO: We shouldn't need an explicit cast here.
        me.paths[x][y] =
          if terrain.height[x][y] == 9 then 1 as i64 else 0;
        ++x;
      }
      ++y;
    }
    return var;
  }

  fn AddLevel[addr self: Self*](terrain: Terrain, level: i32) -> i64 {
    var total: i64 = 0;
    let adj: array((i32, i32), 4) = ((-1, 0), (0, -1), (1, 0), (0, 1));
    var y: i32 = 0;
    while (y < 43) {
      var x: i32 = 0;
      while (x < 43) {
        if (terrain.height[x][y] == level) {
          var paths: i64 = 0;
          var i: i32 = 0;
          while (i < 4) {
            let adj_x: i32 = x + adj[i].0;
            let adj_y: i32 = y + adj[i].1;
            if (adj_x >= 0 and adj_x < 43 and
                adj_y >= 0 and adj_y < 43 and
                terrain.height[adj_x][adj_y] == level + 1) {
              paths = paths + self->paths[adj_x][adj_y];
            }
            ++i;
          }
          self->paths[x][y] = paths;
          total = total + paths;
        }
        ++x;
      }
      ++y;
    }
    return total;
  }

  var paths: array(array(i64, 43), 43);
}

fn Run() {
  var terrain: Terrain = Terrain.Read();
  var paths: PathsToTop = PathsToTop.Make(terrain);
  var i: i32 = 8;
  var total: i64;
  while (i >= 0) {
    total = paths.AddLevel(terrain, i);
    --i;
  }
  PrintInt64(total);
}
