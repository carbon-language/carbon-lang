// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{executable_semantics} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{executable_semantics} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{executable_semantics} %s

package ExecutableSemanticsTest api;

fn Main() -> i32 {
  var t: auto = 5;
  match (t) {
    // CHECK: COMPILATION ERROR: {{.*}}/executable_semantics/testdata/generic_class/fail_generic_in_pattern.carbon:[[@LINE+1]]: Generic binding may not occur in pattern with expected type: T:! i32
    case T:! i32 =>
      return 0;
    default =>
      return 1;
  }
}
