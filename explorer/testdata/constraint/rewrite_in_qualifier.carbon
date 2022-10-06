// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: result: 2

package ExplorerTest api;

interface HasAssoc {
  let Assoc:! Type;
}
class X {
  // TODO: Replace `==` with `=`.
  external impl as HasAssoc where .Assoc == i32 {}
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
