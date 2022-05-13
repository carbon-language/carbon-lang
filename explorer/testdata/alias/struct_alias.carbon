// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: result: 2121

package ExplorerTest api;

alias AB = {.a: i32, .b: i32};
alias BA = {.b: i32, .a: i32};

fn Main() -> i32 {
  var ab: AB = {.b = 1, .a = 2};
  var ba: BA = ab;
  return 1000 * ab.a + 100 * ab.b + 10 * ba.a + ba.b;
}
