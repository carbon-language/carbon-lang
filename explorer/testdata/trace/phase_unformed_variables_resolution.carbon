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
// ARGS: --trace_file=- --trace_phase=unformed_variables_resolution %s
// AUTOUPDATE

// CHECK:STDOUT: * * * * * * * * * *  resolving unformed variables  * * * * * * * * * *
// CHECK:STDOUT: ----------------------------------------------------------------------
// CHECK:STDOUT: ->> resolving-unformed in decl `interface TestInterface` (phase_unformed_variables_resolution.carbon:7)
// CHECK:STDOUT: ->> resolving-unformed in decl `fn Main` (phase_unformed_variables_resolution.carbon:11)
// CHECK:STDOUT: ->> resolving-unformed in stmt `{ ... }` (phase_unformed_variables_resolution.carbon:11)
// CHECK:STDOUT: ->> resolving-unformed in stmt `return ...;` (phase_unformed_variables_resolution.carbon:10)
// CHECK:STDOUT: result: 0
