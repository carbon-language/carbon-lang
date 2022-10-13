// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

interface Has(T:! Type) {
  fn Get() -> T;
}
impl i32 as Has(Type) {
  fn Get() -> Type { return Self; }
}

class WithType(T:! Has(Type)) {
  fn Get() -> Type { return T.Get(); }
}

fn Main() -> i32 {
  // CHECK:COMPILATION ERROR: {{.*}}/explorer/testdata/generic_class/fail_use_at_compile_time.carbon:[[@LINE+1]]: Function would be called at compile-time
  var v: WithType(i32).Get() = 0;
  return v;
}
