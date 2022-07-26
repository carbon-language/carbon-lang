// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: structs equal: 0
// CHECK: structs equal (w/ implicit conversion of rhs): 1
// CHECK: result: 0

package ExplorerTest api;

class MyType {
  var value: i32;

  impl as Eq {
    fn Equals[me: Self](rhs: Self) -> Bool {
      return me.value == rhs.value;
    }
  }
}

fn Main() -> i32 {
  let x: MyType = {.value = 1};
  let y: MyType = {.value = 2};
  Print("structs equal: {0}", if x == y then 1 else 0);
  Print("structs equal (w/ implicit conversion of rhs): {0}", if x == {.value = 1} then 1 else 0);
  return 0;
}
