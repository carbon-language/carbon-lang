// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

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
// CHECK:STDOUT:   %int.sadd.loc14_12 = add i32 %a, %b
// CHECK:STDOUT:   %.loc14_12.2.temp = alloca i32, align 4
// CHECK:STDOUT:   store i32 %int.sadd.loc14_12, ptr %.loc14_12.2.temp, align 4
// CHECK:STDOUT:   %.loc14_12.4 = load i32, ptr %.loc14_12.2.temp, align 4
// CHECK:STDOUT:   %int.sadd.loc14_16 = add i32 %.loc14_12.4, %c
// CHECK:STDOUT:   ret i32 %int.sadd.loc14_16
// CHECK:STDOUT: }
