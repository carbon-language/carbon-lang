// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: result: 0

package ExplorerTest api;

// The lifetime of a heap-allocated object is until it is explicitely
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
