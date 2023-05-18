// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  // CHECK:STDERR: SYNTAX ERROR: explorer/testdata/string/fail_invalid_escape.carbon:[[@LINE+1]]: Invalid escaping in string: "str\e"
  Print("str\e");
  return 0;
}
