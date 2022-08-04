// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: result: 1

package ExplorerTest api;

class LazyEq {
  var v1: i32;
  var v2: i32;
  impl as ImplicitAs(bool) {
    fn Convert[me: Self]() -> bool {
      return me.v1 == me.v2;
    }
  }
  fn Make(v1: i32, v2: i32) -> Self {
    return {.v1 = v1, .v2 = v2};
  }
}

fn Main() -> i32 {
  return if LazyEq.Make(2, 1 + 1) then 1 else 2;
}
