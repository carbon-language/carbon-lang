// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: result: 0

package ExplorerTest api;


choice OptionalElement {
  None(i32),
  Element(i32)
}


fn Main() -> i32 {
  var f: OptionalElement; //= OptionalElement(i32).Create(22);
  f = OptionalElement.None(22);
  match(f){
    case OptionalElement.None( var x: i32) => {
      return 1;
    }
  }
  return 0;
}