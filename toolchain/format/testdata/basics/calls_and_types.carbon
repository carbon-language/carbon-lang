// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/basics/calls_and_types.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/basics/calls_and_types.carbon

// --- test.carbon

fn F ( a : i32 , b : i32 ) -> i32 {
  var v: array ( bool , 1000 ) ;
  if ( a < b ) {
    for ( x : i32 in Core . Range ( 10 ) ) {
      Print ( x ) ;
    }
  }
  while ( a > 0 ) {
    return F ( a - 1 , b ) ;
  }
  return 0 ;
}

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: fn F(a: i32, b: i32) -> i32 {
// CHECK:STDOUT:   var v: array(bool, 1000);
// CHECK:STDOUT:   if (a < b) {
// CHECK:STDOUT:     for (x: i32 in Core.Range(10)) {
// CHECK:STDOUT:       Print(x);
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   while (a > 0) {
// CHECK:STDOUT:     return F(a - 1, b);
// CHECK:STDOUT:   }
// CHECK:STDOUT:   return 0;
// CHECK:STDOUT: }
