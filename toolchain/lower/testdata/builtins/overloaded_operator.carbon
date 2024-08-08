// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/builtins/overloaded_operator.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/builtins/overloaded_operator.carbon

// TODO: Move this into prelude.carbon when possible. For now, impl lookup only
// looks in the current file.
impl i32 as Core.Add {
  fn Op[self: i32](other: i32) -> i32 = "int.sadd";
}

fn AddThreeIntegers(a: i32, b: i32, c: i32) -> i32 {
  return a + b + c;
}

// CHECK:STDOUT: ; ModuleID = 'overloaded_operator.carbon'
// CHECK:STDOUT: source_filename = "overloaded_operator.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @AddThreeIntegers(i32 %a, i32 %b, i32 %c) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.sadd.loc18_12 = add i32 %a, %b
// CHECK:STDOUT:   %.loc18_12.3.temp = alloca i32, align 4
// CHECK:STDOUT:   store i32 %int.sadd.loc18_12, ptr %.loc18_12.3.temp, align 4
// CHECK:STDOUT:   %.loc18_12.5 = load i32, ptr %.loc18_12.3.temp, align 4
// CHECK:STDOUT:   %int.sadd.loc18_16 = add i32 %.loc18_12.5, %c
// CHECK:STDOUT:   ret i32 %int.sadd.loc18_16
// CHECK:STDOUT: }
