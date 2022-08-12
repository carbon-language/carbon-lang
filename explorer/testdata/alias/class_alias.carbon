// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: b.n: 1
// CHECK: d.Get(0): 2
// CHECK: e.Get(1): 3
// CHECK: result: 0

package ExplorerTest api;

interface Addable {
  fn Add[me: Self](k: i32) -> Self;
}
impl i32 as Addable {
  fn Add[me: Self](k: i32) -> Self { return me + k; }
}

class Class { var n: i32; }
class GenericClass(T:! Addable) {
  var m: T;
  fn Get[me: Self](n: i32) -> T { return me.m.Add(n); }
}

alias ClassAlias = Class;
alias GenericClassAlias = GenericClass;
alias ClassSpecializationAlias = GenericClassAlias(i32);

fn Main() -> i32 {
  var a: Class = {.n = 1};
  var b: ClassAlias = a;

  var c: GenericClass(i32) = {.m = 2};
  var d: GenericClassAlias(i32) = c;
  var e: ClassSpecializationAlias = c;

  Print("b.n: {0}", b.n);
  Print("d.Get(0): {0}", d.Get(0));
  Print("e.Get(1): {0}", e.Get(1));
  return 0;
}
