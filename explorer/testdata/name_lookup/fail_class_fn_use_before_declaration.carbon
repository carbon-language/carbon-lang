// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

// The bodies of member functions are processed after all immediately enclosing
// classes, impls, and interfaces.
class A {
  fn F() -> type {
    return i32;
  }
  // OK, resolves to `A.F`.
  fn G() -> F() {
    return 0;
  }

  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/name_lookup/fail_class_fn_use_before_declaration.carbon:[[@LINE+1]]: 'I' has not been declared yet
  fn H() -> I() {
    return 0;
  }
  fn I() -> type {
    return i32;
  }
}

fn Main() -> i32 {
  return 0;
}
