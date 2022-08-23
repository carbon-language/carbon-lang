// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: result: 22

package ExplorerTest api;


choice MyOptionalElement(ZZ:! Type) {
  None(ZZ),
  Element(ZZ)
}


fn Main() -> i32 {
  var f: MyOptionalElement(i32);
  f = MyOptionalElement(i32).None(22);
  match(f) {
    case MyOptionalElement(i32).None(var x: i32) => {
      return x;
    }
  }
  return 0;
}
