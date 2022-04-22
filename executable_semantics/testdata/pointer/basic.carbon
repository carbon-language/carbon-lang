// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{executable_semantics} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{executable_semantics} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{executable_semantics} %s
// CHECK: result: 0

package ExecutableSemanticsTest api;

fn Main() -> i32 {
  var x: i32 = 5;
  x = 10; // changes x to 10
  var p: i32* = &x;
  *p = 7; // changes x to 7
  var q: i32* = &*p;
  *q = 0; // changes x to 0
  var y: i32 = *p;

  return y;
}
