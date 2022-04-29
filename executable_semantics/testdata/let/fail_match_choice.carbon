// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{executable_semantics} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{executable_semantics} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{executable_semantics} %s
// CHECK: COMPILATION ERROR: {{.*}}/executable_semantics/testdata/let/fail_match_choice.carbon:29: Cannot assign to rvalue 'x'

package ExecutableSemanticsTest api;

choice Ints {
  None,
  One(i32),
  Two(i32, i32)
}

fn Main() -> i32 {
  var x: auto = Ints.None();
  var n: auto = 0;
  match (x) {
    case Ints.One(var x: auto) =>
      x = 2;
    case Ints.None() =>
      n = n - 1;
    case Ints.Two(x: auto, y: auto) => {
      x = 0;
    }
  }
  return 0;
}
