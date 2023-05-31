// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// A lot of output is elided: this is only checking for a few things for simple
// sanity checking on --parser_debug --trace_file=- -trace_name_resolution output.
//
// NOAUTOUPDATE
// RUN: %{explorer-run} --parser_debug --trace_file=- -trace_name_resolution | %{FileCheck-allow-unmatched}
// CHECK-NOT:STDOUT: ********** source program **********
// CHECK-NOT:STDOUT: interface TestInterface {
// CHECK-NOT:STDOUT: interface ImplicitAs {
// CHECK:STDOUT: ********** resolving names **********
// CHECK-NOT:STDOUT: ********** resolving control flow **********
// CHECK-NOT:STDOUT: ********** type checking **********
// CHECK-NOT:STDOUT: ** declaring interface TestInterface
// CHECK-NOT:STDOUT: ********** resolving unformed variables **********
// CHECK-NOT:STDOUT: ********** printing declarations **********
// CHECK-NOT:STDOUT: interface TestInterface {
// CHECK-NOT:STDOUT: ********** starting execution **********
// CHECK-NOT:STDOUT: ********** initializing globals **********
// CHECK-NOT:STDOUT: ********** calling main function **********
// CHECK-NOT:STDOUT: --- step exp Main() .0. (<Main()>:0) --->
// CHECK-NOT:STDOUT: ********** printing timing **********
// CHECK:STDOUT: result: 0

package ExplorerTest api;

interface TestInterface {}

fn Main() -> i32 {
  return 0;
}
