// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/basics/simple.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/basics/simple.carbon

// --- test.carbon

fn  F  (  x  :  i32  )  ->  i32  {  return  x  ;  }

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: fn F (x: i32) -> i32 {
// CHECK:STDOUT:   return x;
// CHECK:STDOUT: }
