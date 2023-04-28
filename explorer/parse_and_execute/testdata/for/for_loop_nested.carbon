// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 20

package ExplorerTest api;

fn Main() -> i32 {
  var ar: [i32; 4] = (0, 1,2,3);
   var count : i32 = 0;
   for( x: i32 in ar){
       count = count +1;
        for( x: i32 in ar){
            count = count +1;
        }
   }
   return count;
}
