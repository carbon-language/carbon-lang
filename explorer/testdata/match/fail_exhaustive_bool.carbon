// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: not %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: not %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

fn A(b: bool) -> i32 {
  // TODO: This should be considered an exhaustive match, but that's not
  // implemented yet.
  match (b) {
    case false => { return 1; }
    case true => { return 2; }
  }
// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/match/fail_exhaustive_bool.carbon:[[@LINE+1]]: non-exhaustive match may allow control-flow to reach the end of a function that provides a `->` return type
}

fn Main() -> i32 {
  return A(0);
}
