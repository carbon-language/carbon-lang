// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: result: 122

package ExplorerTest api;

interface A { fn F() -> i32; }
interface B { fn F() -> i32; }

impl i32 as A { fn F() -> i32 { return 1; } }
impl i32 as B { fn F() -> i32 { return 2; } }

fn GetAB(T:! B where .Self is A) -> i32 {
  return 100 * T.(A.F)() + 10 * T.(B.F)() + T.F();
}

fn Main() -> i32 {
  return GetAB(i32);
}
