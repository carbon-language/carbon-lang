// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 0

package ExplorerTest api;

// The lifetime of a heap-allocated object is until it is explicitly
// deleted. Its lifetime is unrelated to the procedure call stack.

fn AllocateInteger() -> i32* {
  return heap.New(0);
}

fn Main() -> i32 {
  var p: i32* = AllocateInteger();
  var y: i32 = *p;
  heap.Delete(p);
  return y;
}
