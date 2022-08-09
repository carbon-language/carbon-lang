// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: Nice!
// CHECK: result: 0

package ExplorerTest api;
fn test(para: i32)->bool{
  return true;
}
fn Main() -> i32 {
  var i : i32 = Rand(0,100);
  var j : i32 = Rand(0,100);
  if(i == j){
    Print("HALLO WELT");
  }else{
    Print("Nice!");
  }
  return 0;
}
