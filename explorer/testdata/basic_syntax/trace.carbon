// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
//
// A lot of output is elided: this is only checking for a few things for simple
// sanity checking on --parser_debug --trace_file=- output.
//
// NOAUTOUPDATE
// CHECK: ********** source program **********
// CHECK: fn Print (format_str: String) {
// CHECK: ********** type checking **********
// CHECK: checking tuple pattern (format_str: String)
// CHECK: ********** type checking complete **********
// CHECK: fn Print (format_str: String) {
// CHECK: ********** starting execution **********
// CHECK: ********** initializing globals **********
// CHECK: ********** calling main function **********
// CHECK: {
// CHECK: stack:
// CHECK: heap:
// CHECK: }
// CHECK: --- step exp Main() (<Main()>:0) --->
// CHECK: result: 0

package ExplorerTest api;

fn Main() -> i32 {
  return 0;
}
