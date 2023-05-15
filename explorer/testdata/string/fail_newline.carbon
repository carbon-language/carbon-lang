// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  // CHECK:STDERR: SYNTAX ERROR: explorer/testdata/string/fail_newline.carbon:[[@LINE+2]]: missing closing quote in single-line string: "new
  // CHECK:STDERR:
  Print("new
line");
  return 0;
}
