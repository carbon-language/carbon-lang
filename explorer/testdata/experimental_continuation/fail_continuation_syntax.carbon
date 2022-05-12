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

fn Main() -> i32 {
  var x: i32 = 0;
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/experimental_continuation/fail_continuation_syntax.carbon:[[@LINE+1]]: syntax error, unexpected identifier, expecting LEFT_CURLY_BRACE
  __continuation k x = 3;
  __run k;
  return x;
}
