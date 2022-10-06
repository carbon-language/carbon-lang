// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

fn CompareStr(s: String) -> i32 {
  // CHECK:SYNTAX ERROR: {{.*}}/explorer/testdata/string/fail_raw_more_hash_tags_on_left.carbon:[[@LINE+2]]: missing closing quote in single-line string: ##"str"#) {
  // CHECK-EMPTY:
  if (s == ##"str"#) {
    return 0;
  }
  return 1;
}

fn Main() -> i32 {
  return CompareStr("str");
}
