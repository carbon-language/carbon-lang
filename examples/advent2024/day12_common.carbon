// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/12

library "day12_common";

import Core library "io";
import library "io_utils";

class Map {
  fn Read() -> Map {
    returned var me: Self;
    var y: i32 = 0;
    while (y < 140) {
      var x: i32 = 0;
      while (x < 140) {
        me.kind[x][y] = ReadChar();
        ++x;
      }
      SkipNewline();
      ++y;
    }
    return var;
  }

  fn At[self: Self](x: i32, y: i32) -> i32 {
    return if x < 0 or x >= 140 or y < 0 or y >= 140
           then -1
           else self.kind[x][y];
  }

  var kind: array(array(i32, 140), 140);
}

class DisjointSetForest {
  fn Make() -> DisjointSetForest {
    returned var me: Self;
    var i: i32 = 0;
    while (i < 140 * 140) {
      me.nodes[i].next = i;
      me.nodes[i].weight = 1;
      me.nodes[i].unions = 0;
      ++i;
    }
    return var;
  }

  fn Lookup[addr self: Self*](a: i32) -> i32 {
    let next: i32 = self->nodes[a].next;
    if (next == a) {
      return next;
    }
    let resolved: i32 = self->Lookup(next);
    self->nodes[a].next = resolved;
    return resolved;
  }

  fn Unions[self: Self](canon_a: i32) -> i32 {
    return self.nodes[canon_a].unions;
  }

  fn Weight[self: Self](canon_a: i32) -> i32 {
    return self.nodes[canon_a].weight;
  }

  fn Union[addr self: Self*](a: i32, b: i32) {
    let canon_b: i32 = self->Lookup(b);
    self->Set(a, canon_b);
    ++self->nodes[canon_b].unions;
  }

  private fn Set[addr self: Self*](a: i32, canon_b: i32) {
    let next: i32 = self->nodes[a].next;
    self->nodes[a].next = canon_b;
    if (next == a) {
      if (a != canon_b) {
        self->nodes[canon_b].weight += self->nodes[a].weight;
        self->nodes[canon_b].unions += self->nodes[a].unions;
      }
    } else {
      self->Set(next, canon_b);
    }
  }

  // TODO: Consider adding ranked choice.
  // TODO: Make this generic in the payload data.
  var nodes: array({.next: i32, .weight: i32, .unions: i32}, 140 * 140);
}

fn MakeRegions(map: Map) -> DisjointSetForest {
  returned var forest: DisjointSetForest =
    DisjointSetForest.Make();

  var x: i32 = 0;
  while (x < 140) {
    var y: i32 = 0;
    while (y < 140) {
      var a: i32 = 0;
      while (a < 2) {
        // TODO: Crashes toolchain:
        // let adj: (i32, i32) = if a == 0 then (x - 1, y) else (x, y - 1);
        // if (map.At(adj.0, adj.1) == map.At(x, y)) {
        //   forest.Union(y * 140 + x, adj.1 * 140 + adj.0);
        // }
        let adj_x: i32 = if a == 0 then x - 1 else x;
        let adj_y: i32 = if a == 0 then y else y - 1;
        if (map.At(adj_x, adj_y) == map.At(x, y)) {
          forest.Union(y * 140 + x, adj_y * 140 + adj_x);
        }
        ++a;
      }
      ++y;
    }
    ++x;
  }

  return var;
}
