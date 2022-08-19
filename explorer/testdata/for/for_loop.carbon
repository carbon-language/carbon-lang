// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: HALLO WELT
// CHECK: HALLO WELT
// CHECK: HALLO WELT
// CHECK: HALLO WELT
// CHECK: HALLO WELT
// CHECK: HALLO WELT
// CHECK: HALLO WELT
// CHECK: HALLO WELT
// CHECK: result: 8

package ExplorerTest api;




fn Main() -> i32 {
  var ar: [i32; 4] = (0, 1,2,3);
  var count : i32 = 0;
  for( x: i32 in ar){
      Print("HALLO WELT ", x);
      count = count +1;
  }
  for( x: i32 in ar){
      Print("HALLO WELT ", x);
      count = count +1;
  }
  return count;
}
