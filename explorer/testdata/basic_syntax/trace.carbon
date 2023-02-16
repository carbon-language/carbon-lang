// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// A lot of output is elided: this is only checking for a few things for simple
// sanity checking on --parser_debug --trace_file=- output.
//
// NOAUTOUPDATE
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: ********** source program **********
// CHECK:STDOUT: interface ImplicitAs {
// CHECK:STDOUT: ********** type checking **********
// CHECK:STDOUT: ** declaring interface ImplicitAs
// CHECK:STDOUT: ********** resolving unformed variables **********
// CHECK:STDOUT: ********** printing declarations **********
// CHECK:STDOUT: interface ImplicitAs {
// CHECK:STDOUT: ********** starting execution **********
// CHECK:STDOUT: ********** initializing globals **********
// CHECK:STDOUT: ********** calling main function **********
// CHECK:STDOUT: --- step exp Main() .0. (<Main()>:0) --->
// CHECK:STDOUT: result: 0

package ExplorerTest api;

fn Main() -> i32 {
  return 0;
}
