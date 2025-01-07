// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/6

import Core library "io";

import library "day6_common";

class LoopDetector {
  fn Make() -> LoopDetector {
    return {.last = ((-1, -1), (-1, -1)), .steps = 1, .next_steps = 1};
  }

  fn Check[addr self: Self*](next: ((i32, i32), (i32, i32))) -> bool {
    // TODO: if (next == self->last) {
    // TODO: The lexer mishandles `next.0.0` as `next` `.` `0.0`.
    if (next.0 .0 == self->last.0 .0 and next.0 .1 == self->last.0 .1 and
        next.1 .0 == self->last.1 .0 and next.1 .1 == self->last.1 .1) {
      return true;
    }
    --self->steps;
    if (self->steps == 0) {
      self->steps = self->next_steps;
      self->next_steps <<= 1;
      self->last = next;
    }
    return false;
  }

  var last: ((i32, i32), (i32, i32));
  var steps: i32;
  var next_steps: i32;
}

fn AddingObstacleMakesALoop(position: Maze) -> bool {
  var maze: Maze = position.Copy();
  var loop: LoopDetector = LoopDetector.Make();
  if (not maze.AddObstacle()) {
    return false;
  }
  while (maze.Step()) {
    if (loop.Check((maze.loc, maze.dir))) {
      return true;
    }
  }
  return false;
}

fn Run() {
  var maze: Maze = Maze.Read();
  var loops: i32 = 0;
  while (true) {
    if (AddingObstacleMakesALoop(maze)) {
      ++loops;
    }
    if (not maze.Step()) {
      break;
    }
  }
  Core.Print(loops);
}
