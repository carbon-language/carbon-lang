// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: ab.a: 2
// CHECK: ab.b: 1
// CHECK: ba.a: 2
// CHECK: ba.b: 1
// CHECK: result: 0

package ExplorerTest api;

alias AB = {.a: i32, .b: i32};
alias BA = {.b: i32, .a: i32};

fn Main() -> i32 {
  var ab: AB = {.b = 1, .a = 2};
  var ba: BA = ab;
  Print("ab.a: {0}", ab.a);
  Print("ab.b: {0}", ab.b);
  Print("ba.a: {0}", ba.a);
  Print("ba.b: {0}", ba.b);
  return 0;
}
