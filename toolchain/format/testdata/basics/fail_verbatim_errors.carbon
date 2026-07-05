// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/basics/fail_verbatim_errors.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/basics/fail_verbatim_errors.carbon

// An error subtree keeps its original source text, including its odd spacing
// and comments, while the surrounding well-formed code still reformats.

// --- fail_verbatim_region.carbon

fn  F ( )  {
  var  ok : i32  =  1 ;
  var broken  :  = 2  +  ;
  var  also_ok : i32  =  3 ;
}

fn  G ( )  {
  var broken  :  =  // odd   comment
      2  +  ;
}

// --- AUTOUPDATE-SPLIT

// CHECK:STDERR: fail_verbatim_region.carbon:4:18: error: expected expression [ExpectedExpr]
// CHECK:STDERR:   var broken  :  = 2  +  ;
// CHECK:STDERR:                  ^
// CHECK:STDERR:
// CHECK:STDERR: fail_verbatim_region.carbon:4:26: error: expected expression [ExpectedExpr]
// CHECK:STDERR:   var broken  :  = 2  +  ;
// CHECK:STDERR:                          ^
// CHECK:STDERR:
// CHECK:STDERR: fail_verbatim_region.carbon:9:18: error: expected expression [ExpectedExpr]
// CHECK:STDERR:   var broken  :  =  // odd   comment
// CHECK:STDERR:                  ^
// CHECK:STDERR:
// CHECK:STDERR: fail_verbatim_region.carbon:10:13: error: expected expression [ExpectedExpr]
// CHECK:STDERR:       2  +  ;
// CHECK:STDERR:             ^
// CHECK:STDERR:
// CHECK:STDOUT: fn F() {
// CHECK:STDOUT:   var ok: i32 = 1;
// CHECK:STDOUT:   var broken  :  = 2  +  ;
// CHECK:STDOUT:   var also_ok: i32 = 3;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn G() {
// CHECK:STDOUT:   var broken  :  =  // odd   comment
// CHECK:STDOUT:       2  +  ;
// CHECK:STDOUT: }
