// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK:result: 12

package ExplorerTest api;

interface HasAssoc {
  let Assoc:! Type;
  let AssocVal:! i32;
}
class X {
  // TODO: Replace `==` with `=`.
  external impl as HasAssoc where .Assoc == i32 and .AssocVal == 2 {}
}

fn F[T:! HasAssoc where .Assoc = i32](x: T) -> i32 {
  var a: T.Assoc = 1;
  return a;
}

fn G[T:! HasAssoc where .AssocVal = 2](x: T) -> i32 {
  return x.AssocVal;
}

fn Main() -> i32 {
  var x: X = {};
  return F(x) * 10 + G(x);
}
