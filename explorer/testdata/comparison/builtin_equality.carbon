// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: bool: 1
// CHECK: bool: 0
// CHECK: bool: 0
// CHECK: bool: 1
// CHECK: bool: 0
// CHECK: bool: 1
// CHECK: bool: 1
// CHECK: bool: 0
// CHECK: string: 1
// CHECK: string: 0
// CHECK: string: 0
// CHECK: string: 1
// CHECK: string: 0
// CHECK: string: 1
// CHECK: string: 1
// CHECK: string: 0
// CHECK: int: 1
// CHECK: int: 0
// CHECK: int: 0
// CHECK: int: 1
// CHECK: int: 0
// CHECK: int: 1
// CHECK: int: 1
// CHECK: int: 0
// CHECK: result: 0

package ExplorerTest api;

fn Compare[T:! EqWith(.Self)](format: String, a: T, b: T) {
  Print(format, if a == b then 1 else 0);
  Print(format, if a != b then 1 else 0);
}

fn CompareAll[T:! EqWith(.Self)](format: String, a: T, b: T) {
  Compare(format, a, a);
  Compare(format, a, b);
  Compare(format, b, a);
  Compare(format, b, b);
}

fn Main() -> i32 {
  CompareAll("bool: {0}", false, true);
  CompareAll("string: {0}", "hello", "world");
  CompareAll("int: {0}", 1, 2);
  return 0;
}
