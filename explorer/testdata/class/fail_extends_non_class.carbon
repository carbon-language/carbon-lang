// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

// CHECK:SYNTAX ERROR: {{.*}}/explorer/testdata/class/fail_extends_non_class.carbon:[[@LINE+1]]: syntax error, unexpected sized_type_literal, expecting identifier
class C extends i32 {
  var x: i32;
  var y: i32;
}

fn Main() -> i32 {
  return 0;
}
