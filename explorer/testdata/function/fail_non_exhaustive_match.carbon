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
  var n: i32 = 0;
  match (n) {
    case 1 => { return 0; }
  }
// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/function/fail_non_exhaustive_match.carbon:[[@LINE+1]]: non-exhaustive match may allow control-flow to reach the end of a function that provides a `->` return type
}
