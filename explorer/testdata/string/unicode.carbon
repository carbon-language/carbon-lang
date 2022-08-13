// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: result: 0

package ExplorerTest api;

fn CompareStr(s: String) -> i32 {
  if (s == "str;\u0021") {
    return 0;
  }
  return 1;
}

fn Main() -> i32 {
  return CompareStr("\u0048\u0045\u004C\u004C\u004F \u0057\u004F\u0052\u004C\u0044\u0021");
}
