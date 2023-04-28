// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 2

package ExplorerTest api;

interface HasAssoc {
  let Assoc:! type;
}
class X {
  external impl as HasAssoc where .Assoc = i32 {}
}

alias WithoutRewrite = HasAssoc where .Assoc == i32;
alias WithRewrite = HasAssoc where .Assoc = i32;

fn G[T:! WithoutRewrite](x: T) -> i32 {
  // TODO: We should reject this once `==` isn't applied automatically.
  var a: T.(WithRewrite.Assoc) = 2;
  return a;
}

fn Main() -> i32 {
  var x: X = {};
  return G(x);
}
