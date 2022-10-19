// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

choice Ints {
  None,
  One(i32),
  Two(i32, i32)
}

fn Main() -> i32 {
  match (Ints.None()) {
    case Ints.None => { return 0; }
    // CHECK:COMPILATION ERROR: {{.*}}/explorer/testdata/basic_syntax/fail_choice_no_parens.carbon:[[@LINE+3]]: type error in `match` pattern type
    // CHECK:expected: fn () -> choice Ints
    // CHECK:actual: choice Ints
    default => { return 1; }
  }
}
