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
// CHECK: bool: 1
// CHECK: bool: 1
// CHECK: bool: 1
// CHECK: bool: 1
// CHECK: bool: 1
// CHECK: bool: 1
// CHECK: bool: 1
// CHECK: string: 1
// CHECK: string: 1
// CHECK: string: 1
// CHECK: string: 1
// CHECK: string: 1
// CHECK: string: 1
// CHECK: string: 1
// CHECK: string: 1
// CHECK: int: 1
// CHECK: int: 1
// CHECK: int: 1
// CHECK: int: 1
// CHECK: int: 1
// CHECK: int: 1
// CHECK: int: 1
// CHECK: int: 1
// CHECK: result: 0

package ExplorerTest api;

fn CompareEqualValues[T:! EqWith(.Self)](format: String, a: T, b: T) {
  Print(format, if a == b then 1 else 0);
  Print(format, if a != b then 0 else 1);
}

fn CompareDifferentValues[T:! EqWith(.Self)](format: String, a: T, b: T) {
  Print(format, if a == b then 0 else 1);
  Print(format, if a != b then 1 else 0);
}

fn CompareAll[T:! EqWith(.Self)](format: String, a: T, b: T) {
  CompareEqualValues(format, a, a);
  CompareEqualValues(format, b, b);
  CompareDifferentValues(format, a, b);
  CompareDifferentValues(format, b, a);
}

fn Main() -> i32 {
  CompareAll("bool: {0}", false, true);
  CompareAll("string: {0}", "hello", "world");
  CompareAll("int: {0}", 1, 2);
  return 0;
}
