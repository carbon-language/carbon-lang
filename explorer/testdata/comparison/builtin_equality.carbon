// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: bool: 1
// CHECK:STDOUT: bool: 1
// CHECK:STDOUT: bool: 1
// CHECK:STDOUT: bool: 1
// CHECK:STDOUT: bool: 1
// CHECK:STDOUT: bool: 1
// CHECK:STDOUT: bool: 1
// CHECK:STDOUT: bool: 1
// CHECK:STDOUT: string: 1
// CHECK:STDOUT: string: 1
// CHECK:STDOUT: string: 1
// CHECK:STDOUT: string: 1
// CHECK:STDOUT: string: 1
// CHECK:STDOUT: string: 1
// CHECK:STDOUT: string: 1
// CHECK:STDOUT: string: 1
// CHECK:STDOUT: int: 1
// CHECK:STDOUT: int: 1
// CHECK:STDOUT: int: 1
// CHECK:STDOUT: int: 1
// CHECK:STDOUT: int: 1
// CHECK:STDOUT: int: 1
// CHECK:STDOUT: int: 1
// CHECK:STDOUT: int: 1
// CHECK:STDOUT: result: 0

package ExplorerTest api;

fn CompareEqualValues[T:! EqWith(.Self)](format: String, a: T, b: T) {
  Print(format, if a == b then 1 else 0);
  Print(format, if a != b then 0 else 1);
}

fn CompareDifferentValues[U:! EqWith(.Self)](format: String, a: U, b: U) {
  Print(format, if a == b then 0 else 1);
  Print(format, if a != b then 1 else 0);
}

fn CompareAll[V:! EqWith(.Self)](format: String, a: V, b: V) {
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
