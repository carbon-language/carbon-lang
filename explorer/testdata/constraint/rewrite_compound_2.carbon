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
  let Assoc:! i32;
}
class X {
  // TODO: Replace `==` with `=`.
  external impl as HasAssoc where .Assoc == 1 {}
}

alias WithoutRewrite = HasAssoc where .Assoc == 1;
alias WithRewrite = HasAssoc where .Assoc = 1;

fn F[T:! WithRewrite](x: T) -> i32 {
  return x.(WithoutRewrite.Assoc);
}

fn Main() -> i32 {
  var x: X = {};
  return F(x);
}
