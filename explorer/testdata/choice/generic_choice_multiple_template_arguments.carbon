// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s
// CHECK:result: 22

package ExplorerTest api;


choice MyOptionalElement(ZZ:! Type, YY:! Type) {
  None(YY),
  Element(ZZ)
}


fn Main() -> i32 {
  var f: MyOptionalElement(String,i32);
  f = MyOptionalElement(String,i32).None(22);
  match(f) {
    case MyOptionalElement(String,i32).None(var x: i32) => {
      return x;
    }
  }
  return 0;
}
