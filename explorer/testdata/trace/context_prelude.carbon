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
// ARGS: --trace_file=- --trace_phase=all --trace_file_context=prelude %s
// NOAUTOUPDATE
// SET-CHECK-SUBSET
// CHECK:STDOUT: ->> declaring `interface As` (prelude.carbon:14)
// CHECK:STDOUT: interface As {
