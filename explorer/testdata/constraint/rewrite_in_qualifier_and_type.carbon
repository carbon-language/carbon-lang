// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s
// CHECK:result: 3

package ExplorerTest api;

interface HasAssoc {
  let Assoc:! Type;
}
class X {
  // TODO: Replace `==` with `=`.
  external impl as HasAssoc where .Assoc == i32 {}
}

fn H[T:! HasAssoc where .Assoc = i32, U:! Type where .Self == i32](a: T, b: U) -> i32 {
  var a: T.((HasAssoc where .Assoc = U).Assoc) = 3;
  return a;
}

fn Main() -> i32 {
  var x: X = {};
  var y: i32 = 0;
  return H(x, y);
}
