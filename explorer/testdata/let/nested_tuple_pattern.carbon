// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 0

package ExplorerTest api;

choice Ints {
  None,
  One(i32),
  Two(i32, i32)
}

fn Main() -> i32 {
  let (var Ints.Two(a1: auto, var a2: auto),
       ((b: auto, var c: auto), var (d: auto, e: auto))) =
      (Ints.Two(1, 10), ((2, 3), (4, 5)));
  a1 = 0;
  a2 = 0;
  c = 0;
  d = 0;
  e = 0;
  return 0;
}
