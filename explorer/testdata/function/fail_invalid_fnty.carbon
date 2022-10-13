// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/function/fail_invalid_fnty.carbon:[[@LINE+2]]: type error in `-`:
// CHECK:STDERR: could not find implementation of interface Negate for bool
fn f(g: __Fn(-true) -> true) {
}

fn Main() -> i32 {
  return 0;
}
