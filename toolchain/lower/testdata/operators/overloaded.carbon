// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- prelude.carbon

package Core api;

interface Negate {
  fn Op[self: Self]() -> Self;
}
interface Mul {
  fn Op[self: Self](other: Self) -> Self;
}

// --- overloaded.carbon

import Core;

class Number {
  var is_positive: bool;
}

impl Number as Core.Negate {
  fn Op[self: Number]() -> Number {
    return {.is_positive = not self.is_positive};
  }
}
impl Number as Core.Mul {
  fn Op[self: Number](other: Number) -> Number {
    return {.is_positive = (self.is_positive and other.is_positive) or
                            (not self.is_positive and not other.is_positive)};
  }
}

fn Calculate(a: Number, b: Number) -> Number {
  return -a * b;
}

// CHECK:STDOUT: ; ModuleID = 'prelude.carbon'
// CHECK:STDOUT: source_filename = "prelude.carbon"
// CHECK:STDOUT: ; ModuleID = 'overloaded.carbon'
// CHECK:STDOUT: source_filename = "overloaded.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Op(ptr sret({ i1 }) %return, ptr %self) {
// CHECK:STDOUT:   %is_positive = getelementptr inbounds { i1 }, ptr %self, i32 0, i32 0
// CHECK:STDOUT:   %1 = load i1, ptr %is_positive, align 1
// CHECK:STDOUT:   %2 = xor i1 %1, true
// CHECK:STDOUT:   %is_positive1 = getelementptr inbounds { i1 }, ptr %return, i32 0, i32 0
// CHECK:STDOUT:   store i1 %2, ptr %is_positive1, align 1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Op.1(ptr sret({ i1 }) %return, ptr %self, ptr %other) {
// CHECK:STDOUT:   %is_positive = getelementptr inbounds { i1 }, ptr %self, i32 0, i32 0
// CHECK:STDOUT:   %1 = load i1, ptr %is_positive, align 1
// CHECK:STDOUT:   br i1 %1, label %2, label %4
// CHECK:STDOUT:
// CHECK:STDOUT: 2:                                                ; preds = %0
// CHECK:STDOUT:   %is_positive1 = getelementptr inbounds { i1 }, ptr %other, i32 0, i32 0
// CHECK:STDOUT:   %3 = load i1, ptr %is_positive1, align 1
// CHECK:STDOUT:   br label %4
// CHECK:STDOUT:
// CHECK:STDOUT: 4:                                                ; preds = %2, %0
// CHECK:STDOUT:   %5 = phi i1 [ false, %0 ], [ %3, %2 ]
// CHECK:STDOUT:   %6 = xor i1 %5, true
// CHECK:STDOUT:   br i1 %6, label %7, label %15
// CHECK:STDOUT:
// CHECK:STDOUT: 7:                                                ; preds = %4
// CHECK:STDOUT:   %is_positive2 = getelementptr inbounds { i1 }, ptr %self, i32 0, i32 0
// CHECK:STDOUT:   %8 = load i1, ptr %is_positive2, align 1
// CHECK:STDOUT:   %9 = xor i1 %8, true
// CHECK:STDOUT:   br i1 %9, label %10, label %13
// CHECK:STDOUT:
// CHECK:STDOUT: 10:                                               ; preds = %7
// CHECK:STDOUT:   %is_positive3 = getelementptr inbounds { i1 }, ptr %other, i32 0, i32 0
// CHECK:STDOUT:   %11 = load i1, ptr %is_positive3, align 1
// CHECK:STDOUT:   %12 = xor i1 %11, true
// CHECK:STDOUT:   br label %13
// CHECK:STDOUT:
// CHECK:STDOUT: 13:                                               ; preds = %10, %7
// CHECK:STDOUT:   %14 = phi i1 [ false, %7 ], [ %12, %10 ]
// CHECK:STDOUT:   br label %15
// CHECK:STDOUT:
// CHECK:STDOUT: 15:                                               ; preds = %13, %4
// CHECK:STDOUT:   %16 = phi i1 [ true, %4 ], [ %14, %13 ]
// CHECK:STDOUT:   %is_positive4 = getelementptr inbounds { i1 }, ptr %return, i32 0, i32 0
// CHECK:STDOUT:   store i1 %16, ptr %is_positive4, align 1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Calculate(ptr sret({ i1 }) %return, ptr %a, ptr %b) {
// CHECK:STDOUT:   %temp = alloca { i1 }, align 8
// CHECK:STDOUT:   call void @Op(ptr %temp, ptr %a)
// CHECK:STDOUT:   call void @Op.1(ptr %return, ptr %temp, ptr %b)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder i1 true, { 2, 0, 1, 3, 4 }
