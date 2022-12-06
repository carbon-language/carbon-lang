// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

fn Main() -> i32 {
  // CHECK:STDERR: SYNTAX ERROR: {{.*}}/explorer/testdata/string/fail_hex_lower.carbon:[[@LINE+1]]: Invalid escaping in string: "str\xaa"
  Print("str\xaa");
  return 0;
}
