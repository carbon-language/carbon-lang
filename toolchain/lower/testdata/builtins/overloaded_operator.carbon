// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- prelude.carbon

package Core api;

interface Add {
  fn Op[self: Self](other: Self) -> Self;
}

// --- user.carbon

import Core;

// TODO: Move this into prelude.carbon when possible. For now, impl lookup only
// looks in the current file.
impl i32 as Core.Add {
  fn Op[self: i32](other: i32) -> i32 = "int.add";
}

fn AddThreeIntegers(a: i32, b: i32, c: i32) -> i32 {
  return a + b + c;
}

// CHECK:STDOUT: ; ModuleID = 'prelude.carbon'
// CHECK:STDOUT: source_filename = "prelude.carbon"
// CHECK:STDOUT: ; ModuleID = 'user.carbon'
// CHECK:STDOUT: source_filename = "user.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @AddThreeIntegers(i32 %a, i32 %b, i32 %c) {
// CHECK:STDOUT:   %add = add i32 %a, %b
// CHECK:STDOUT:   %temp = alloca i32, align 4
// CHECK:STDOUT:   store i32 %add, ptr %temp, align 4
// CHECK:STDOUT:   %1 = load i32, ptr %temp, align 4
// CHECK:STDOUT:   %add1 = add i32 %1, %c
// CHECK:STDOUT:   ret i32 %add1
// CHECK:STDOUT: }
