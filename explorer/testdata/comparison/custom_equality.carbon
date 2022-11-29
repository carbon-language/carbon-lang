// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: structs equal: 0
// CHECK:STDOUT: structs not equal: 1
// CHECK:STDOUT: result: 0

package ExplorerTest api;

class MyType {
  var value: i32;

  impl as EqWith(Self) {
    fn Equal[me: Self](other: Self) -> bool {
      return me.value == other.value;
    }
    fn NotEqual[me: Self](other: Self) -> bool{
      return me.value != other.value;
    }
  }
}

fn Main() -> i32 {
  let x: MyType = {.value = 1};
  let y: MyType = {.value = 2};
  Print("structs equal: {0}", if x == y then 1 else 0);
  Print("structs not equal: {0}", if x != y then 1 else 0);
  return 0;
}
