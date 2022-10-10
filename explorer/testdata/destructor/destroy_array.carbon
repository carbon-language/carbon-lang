// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: DESTRUCTOR A 0
// CHECK: DESTRUCTOR A 1
// CHECK: result: 1

package ExplorerTest api;


class A{
    destructor[me: Self]{
        Print("DESTRUCTOR A {0}",me.n);
    }
    var n: i32;
}

fn Main() -> i32 {
  var a: [A; 2] = ({.n = 6},{.n = 5});
  var b: [[A; 2]; 2] = (({.n = 4},{.n = 3}), ({.n = 2},{.n = 1}));
  return 1;
}
