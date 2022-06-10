// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

interface A { fn F() -> i32; }
interface B { fn G() -> i32; }

// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/constraint/missing_member.carbon:[[@LINE+1]]: member access, H not in constraint interface A & interface B where .Self:! Type is interface A, .Self:! Type is interface B
fn Get[T:! A & B](n: T) -> i32 { return n.H(); }

impl i32 as A {
  fn F() -> i32 { return 1; }
}
impl i32 as B {
  fn G() -> i32 { return 2; }
}

fn Main() -> i32 {
  var z: i32 = 0;
  return Get(z);
}
