// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: result: 1

package ExplorerTest api;

interface A(T:! Type) {}
interface B {
  fn F() -> i32;
}

class C(T:! Type) {}

external impl forall [T:! Type] C(T) as A(T) & B {
  fn F() -> i32 { return 1; }
}

fn Main() -> i32 {
  var v: C(i32) = {};
  return v.(B.F)();
}
