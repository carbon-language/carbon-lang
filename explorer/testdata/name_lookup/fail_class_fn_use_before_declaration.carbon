// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

// The bodies of member functions are processed after all immediately enclosing
// classes, impls, and interfaces.
class A {
  fn F() -> Type {
    return i32;
  }
  // OK, resolves to `A.F`.
  fn G() -> F() {
    return 0;
  }

  // CHECK:COMPILATION ERROR: {{.*}}/explorer/testdata/name_lookup/fail_class_fn_use_before_declaration.carbon:[[@LINE+1]]: 'I' has not been declared yet
  fn H() -> I() {
    return 0;
  }
  fn I() -> Type {
    return i32;
  }
}

fn Main() -> i32 {
  return 0;
}
