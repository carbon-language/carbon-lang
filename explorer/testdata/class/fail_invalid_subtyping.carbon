// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

base class C {
}

class D {
}

fn Main() -> i32 {
  var d: D = {};
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/class/fail_invalid_subtyping.carbon:[[@LINE+1]]: type error in name binding: 'class D*' is not implicitly convertible to 'class C*'
  var c: C* = &d;
  return 0;
}
