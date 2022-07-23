// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: result: 3

package ExplorerTest api;

interface Frob {
  let Result:! Type;
  fn F[me: Self]() -> Result;
}

fn Use[T:! Frob where .Result == .Self](x: T) -> T {
  var v: T = x.F();
  return v;
}

impl i32 as Frob where .Result == i32 {
  fn F[me: Self]() -> i32 { return me + 1; }
}

fn Main() -> i32 {
  return Use(2);
}
