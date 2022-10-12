// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

class C(T:! Type) {}

fn F(T:! Type) -> Type {
  return C(T);
}

fn Main() -> i32 {
  // CHECK:COMPILATION ERROR: {{.*}}/explorer/testdata/generic_class/fail_instantiation_at_compile_time.carbon:[[@LINE+1]]:  function call not allowed here
  var v: F(i32) = {};
  var w: C(i32) = v;
  return 0;
}
