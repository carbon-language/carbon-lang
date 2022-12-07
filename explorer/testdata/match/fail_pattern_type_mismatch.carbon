// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

class A {}
impl A as ImplicitAs(i32) {
  fn Convert[self: Self]() -> i32 { return 1; }
}

fn Main() -> i32 {
  var a: A = {};

  // TODO: It's not clear whether this should be valid. The patterns here have
  // different types, but we could perform different conversions on the source
  // expression when checking each pattern.
  match ((a, a)) {
    case (0, n: A) => { return 1; }
    // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/match/fail_pattern_type_mismatch.carbon:[[@LINE+3]]: type error in `match` pattern type
    // CHECK:STDERR: expected: (i32, class A)
    // CHECK:STDERR: actual: (class A, i32)
    case (n: A, 0) => { return 1; }
    case (1, 1) => { return 0; }
  }

  return 1;
}
