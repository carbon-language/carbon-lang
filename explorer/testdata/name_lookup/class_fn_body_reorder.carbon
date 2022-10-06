// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s
// CHECK:result: 3

package ExplorerTest api;

// The bodies of member functions are processed after all immediately enclosing
// classes, impls, and interfaces.
class A {
  fn F[me: Self]() -> i32 {
    return G() + me.H();
  }
  fn G() -> i32 { return 1; }
  fn H[me: Self]() -> i32 { return 2; }
}

fn Main() -> i32 {
  var a: A = {};
  return a.F();
}
