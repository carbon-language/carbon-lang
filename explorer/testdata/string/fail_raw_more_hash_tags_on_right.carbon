// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

fn CompareStr(s: String) -> i32 {
  // CHECK:STDERR: SYNTAX ERROR: {{.*}}/explorer/testdata/string/fail_raw_more_hash_tags_on_right.carbon:[[@LINE+1]]: invalid character '\x23' in source file.
  if (s == "str"#) {
    return 0;
  }
  return 1;
}

fn Main() -> i32 {
  return CompareStr("str");
}
