// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 2>&1 | %{FileCheck} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

// TODO: This should work
// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/comparison/fail_empty_struct.carbon:[[@LINE+1]]: type error in call: '{}' is not implicitly convertible to 'Type'
external impl {} as EqWith({}) {
  fn Equal[me: Self](other: Self) -> Bool {
    return true;
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
