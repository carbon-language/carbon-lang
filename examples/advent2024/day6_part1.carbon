// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/6

import Core library "io";

import library "day6_common";

fn Run() {
  var maze: Maze = Maze.Read();
  while (maze.Step()) {
  }
  Core.Print(maze.CountVisited());
}
