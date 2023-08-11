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
// ARGS: --trace_file=- --trace_phase=timing %s
// AUTOUPDATE

// CHECK:STDOUT: * * * * * * * * * *  printing timing  * * * * * * * * * *
// CHECK:STDOUT: ---------------------------------------------------------
// CHECK:STDOUT: Time elapsed in ExecProgram: {{\d+}}ms
// CHECK:STDOUT: Time elapsed in AnalyzeProgram: {{\d+}}ms
// CHECK:STDOUT: Time elapsed in AddPrelude: {{\d+}}ms
// CHECK:STDOUT: Time elapsed in Parse: {{\d+}}ms
// CHECK:STDOUT: result: 0
