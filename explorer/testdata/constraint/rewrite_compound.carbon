// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s
// CHECK:result: 1

package ExplorerTest api;

interface HasAssoc {
  let Assoc:! Type;
}
class X {
  external impl as HasAssoc where .Assoc = i32 {}
}

alias WithoutRewrite = HasAssoc where .Assoc == i32;
alias WithRewrite = HasAssoc where .Assoc = i32;

fn F[T:! WithRewrite](x: T) -> i32 {
  var a: T.(WithoutRewrite.Assoc) = 1;
  return a;
}

fn Main() -> i32 {
  var x: X = {};
  return F(x);
}
