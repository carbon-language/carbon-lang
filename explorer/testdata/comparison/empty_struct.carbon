// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 0

package ExplorerTest api;

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
