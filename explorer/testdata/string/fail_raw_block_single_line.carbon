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
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/string/fail_raw_block_single_line.carbon:[[@LINE+1]]: Invalid block string: Too few lines
  if (s == #"""raw string literal starting with """#) {
    return 0;
  }
  return 1;
}

fn Main() -> i32 {
  return CompareStr("\"\"raw string literal starting with \"\"");
}
