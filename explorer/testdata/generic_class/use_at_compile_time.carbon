// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: result: 0

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
  var v: WithType(i32).Get() = 0;
  return v;
}
