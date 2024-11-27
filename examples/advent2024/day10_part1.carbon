// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/10

import Core library "io";

import library "day10_common";
import library "io_utils";

// TODO: Add this to the prelude.
fn PopCount(n: u256) -> i32 {
  var bit: u256 = 1;
  var total: i32 = 0;
  while (bit != 0) {
    if (n & bit != 0) {
      ++total;
    }
    bit = bit << 1;
  }
  return total;
}

class Reachable {
  fn Make(terrain: Terrain) -> Reachable {
    returned var me: Reachable;
    var next: u256 = 1;
    var y: i32 = 0;
    while (y < 43) {
      var x: i32 = 0;
      while (x < 43) {
        if (terrain.height[x][y] == 0) {
          me.trailheads[x][y] = next;
          next = next << 1;
        }
        ++x;
      }
      ++y;
    }
    return var;
  }

  fn AddLevel[addr self: Self*](terrain: Terrain, level: i32) {
    let adj: array((i32, i32), 4) = ((-1, 0), (0, -1), (1, 0), (0, 1));
    var y: i32 = 0;
    while (y < 43) {
      var x: i32 = 0;
      while (x < 43) {
        if (terrain.height[x][y] == level) {
          var reach: u256 = 0;
          var i: i32 = 0;
          while (i < 4) {
            let adj_x: i32 = x + adj[i].0;
            let adj_y: i32 = y + adj[i].1;
            if (adj_x >= 0 and adj_x < 43 and
                adj_y >= 0 and adj_y < 43 and
                terrain.height[adj_x][adj_y] == level - 1) {
              reach = reach | self->trailheads[adj_x][adj_y];
            }
            ++i;
          }
          self->trailheads[x][y] = reach;
        }
        ++x;
      }
      ++y;
    }
  }

  fn Count[self: Self](terrain: Terrain, level: i32) -> i32 {
    var total: i32 = 0;
    var y: i32 = 0;
    while (y < 43) {
      var x: i32 = 0;
      while (x < 43) {
        if (terrain.height[x][y] == level) {
          total += PopCount(self.trailheads[x][y]);
        }
        ++x;
      }
      ++y;
    }
    return total;
  }

  var trailheads: array(array(u256, 43), 43);
}

fn Run() {
  var terrain: Terrain = Terrain.Read();
  var reachable: Reachable = Reachable.Make(terrain);
  var i: i32 = 1;
  while (i <= 9) {
    reachable.AddLevel(terrain, i);
    ++i;
  }
  Core.Print(reachable.Count(terrain, 9));
}
