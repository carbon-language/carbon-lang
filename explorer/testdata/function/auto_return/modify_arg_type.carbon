// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Id(t: type) -> auto { return t; }

// Test non-trivial type expression in parameter type.

fn f(x: Id(i32)) -> i32 {
  return x;
}

fn Main() -> i32 {
  return f(0);
}

// CHECK:STDOUT: result: 0
