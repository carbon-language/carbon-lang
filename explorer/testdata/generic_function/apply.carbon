// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: result: -2

package ExplorerTest api;

fn apply[T:! Type, U:! Type](f: __Fn (T) -> U, x: T) -> U {
  return f(x);
}

fn positive(x: bool) -> i32 {
  if (x) {
    return 2;
  } else {
    return -2;
  }
}

fn Main() -> i32 {
  return apply(positive, false);
}
