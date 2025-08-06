// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_init.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_init.carbon

class Class {
  var a: i32;
  var b: i32;
}

fn F() {
  // CHECK:STDERR: fail_init.carbon:[[@LINE+4]]:3: error: cannot initialize class with 2 fields from struct with 1 field [StructInitElementCountMismatch]
  // CHECK:STDERR:   {.a = 1} as Class;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  {.a = 1} as Class;
  // CHECK:STDERR: fail_init.carbon:[[@LINE+4]]:3: error: missing value for field `b` in struct initialization [StructInitMissingFieldInLiteral]
  // CHECK:STDERR:   {.a = 1, .c = 2} as Class;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  {.a = 1, .c = 2} as Class;
  // CHECK:STDERR: fail_init.carbon:[[@LINE+4]]:3: error: cannot initialize class with 2 fields from struct with 3 fields [StructInitElementCountMismatch]
  // CHECK:STDERR:   {.a = 1, .b = 2, .c = 3} as Class;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  {.a = 1, .b = 2, .c = 3} as Class;
}
