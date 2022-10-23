// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

fn Main() -> i32 {
  // CHECK:STDERR: SYNTAX ERROR: {{.*}}/explorer/testdata/string/fail_raw_block_quotes_not_on_own_line.carbon:[[@LINE+1]]: Invalid block string: Should end with triple quotes: error: closing """
  var s: String = #"""
    error: closing """# is not on its own line.
  """#;

  return 0;
}
