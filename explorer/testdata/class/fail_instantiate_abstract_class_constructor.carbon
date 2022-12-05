// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

abstract class C {
  fn Create() -> Self {
    return { .a = 1 };
  }
  var a: i32;
}

fn Main() -> i32 {
  // CHECK:STDERR: RUNTIME ERROR: {{.*}}/explorer/testdata/class/fail_instantiate_abstract_class_constructor.carbon:[[@LINE+1]]: Cannot instantiate abstract class C
  var c: C = C.Create();
  return 0;
}
