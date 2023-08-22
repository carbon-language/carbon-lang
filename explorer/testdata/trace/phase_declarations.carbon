// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package ExplorerTest api;

interface TestInterface {}

fn Main() -> i32 {
  return 0;
}

// Place checks after code so that line numbers are stable, reducing merge
// conflicts.
// ARGS: --trace_file=- --trace_phase=declarations %s
// AUTOUPDATE

// CHECK:STDOUT: * * * * * * * * * *  printing declarations  * * * * * * * * * *
// CHECK:STDOUT: ---------------------------------------------------------------
// CHECK:STDOUT: interface TestInterface {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn Main () -> i32
// CHECK:STDOUT: {
// CHECK:STDOUT:   return 0;
// CHECK:STDOUT: }
// CHECK:STDOUT: result: 0
