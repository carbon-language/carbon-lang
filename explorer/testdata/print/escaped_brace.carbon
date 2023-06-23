// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: Print: {0}
// CHECK:STDOUT: result: 0

package ExplorerTest api;

fn Main() -> i32 {
  Print("Print: {{0}");
  return 0;
}
