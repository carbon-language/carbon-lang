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

class A {}
impl A as ImplicitAs(i32) {
  fn Convert[me: Self]() -> i32 { return 1; }
}

fn Main() -> i32 {
  var a: A = {};

  // FIXME: It's not clear whether this should be valid. The patterns here have
  // different types, but we could perform different conversions on the source
  // expression when checking each pattern.
  match ((a, a)) {
    case (0, n: A) => { return 1; }
    // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/match/fail_pattern_type_mismatch.carbon:[[@LINE+3]]: type error in `match` pattern type
    // CHECK: expected: (i32, class A)
    // CHECK: actual: (class A, i32)
    case (n: A, 0) => { return 1; }
    case (1, 1) => { return 0; }
  }

  return 1;
}
