// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

abstract class C {
  var a: i32;
}

class D extends C {
  var b: i32;
}

fn Main() -> i32 {
  var d: D = { .base = {.a = 1}, .b = 2 };
  // Not supported yet.
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/class/fail_abstract_class_subtyping.carbon:[[@LINE+1]]: type error in name binding: 'class D*' is not implicitly convertible to 'class C*'
  var c: C* = &d;
  return 0;
}
