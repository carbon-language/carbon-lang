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

interface Bad {
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/interface/fail_bad_member_kind.carbon:[[@LINE+1]]: syntax error, unexpected VAR, expecting FN or RIGHT_CURLY_BRACE
  var V: i32;
}

fn Main() -> i32 { return 0; }
