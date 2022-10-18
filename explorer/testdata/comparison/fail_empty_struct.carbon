// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

// TODO: This should work
// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/comparison/fail_empty_struct.carbon:[[@LINE+1]]: type error in call: '({})' is not implicitly convertible to '(Type)'
external impl {} as EqWith({}) {
  fn Equal[me: Self](other: Self) -> bool {
    return true;
  }
  fn NotEqual[me: Self](other: Self) -> bool {
    return false;
  }
}

fn Main() -> i32 {
  var empty: auto = {};
  if (empty == {}) {
    return 0;
  } else {
    return 1;
  }
}
