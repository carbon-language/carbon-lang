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
// ARGS: --trace_file=- --trace_phase=control_flow_resolution %s
// AUTOUPDATE

// CHECK:STDOUT: * * * * * * * * * *  resolving control flow  * * * * * * * * * *
// CHECK:STDOUT: ----------------------------------------------------------------
// CHECK:STDOUT: ==> flow-resolved return statement `return 0;` in `fn Main` (phase_control_flow_resolution.carbon:10)
// CHECK:STDOUT: result: 0
