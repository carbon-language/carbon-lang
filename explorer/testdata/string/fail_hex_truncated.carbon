// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  // CHECK:STDERR: SYNTAX ERROR: explorer/testdata/string/fail_hex_truncated.carbon:[[@LINE+1]]: Invalid escaping in string: "str\x"
  Print("str\x");
  return 0;
}
