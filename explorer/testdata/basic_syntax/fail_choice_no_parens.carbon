// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

choice Ints {
  None,
  One(i32),
  Two(i32, i32)
}

fn Main() -> i32 {
  match (Ints.None()) {
    case Ints.None => { return 0; }
    // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/basic_syntax/fail_choice_no_parens.carbon:[[@LINE+3]]: type error in `match` pattern type
    // CHECK:STDERR: expected: fn () -> choice Ints
    // CHECK:STDERR: actual: choice Ints
    default => { return 1; }
  }
}
