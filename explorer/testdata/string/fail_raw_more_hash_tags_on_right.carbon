// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

fn CompareStr(s: String) -> i32 {
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/string/fail_raw_more_hash_tags_on_right.carbon:[[@LINE+1]]: invalid character '\x23' in source file.
  if (s == "str"#) {
    return 0;
  }
  return 1;
}

fn Main() -> i32 {
  return CompareStr("str");
}
