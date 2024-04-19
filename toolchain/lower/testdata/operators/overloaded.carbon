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
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_36.1.is_positive = getelementptr inbounds { i1 }, ptr %self, i32 0, i32 0
// CHECK:STDOUT:   %.loc10_36.2 = load i1, ptr %.loc10_36.1.is_positive, align 1
// CHECK:STDOUT:   %.loc10_28 = xor i1 %.loc10_36.2, true
// CHECK:STDOUT:   %.loc10_48.2.is_positive = getelementptr inbounds { i1 }, ptr %return, i32 0, i32 0
// CHECK:STDOUT:   store i1 %.loc10_28, ptr %.loc10_48.2.is_positive, align 1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Op.1(ptr sret({ i1 }) %return, ptr %self, ptr %other) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_33.1.is_positive = getelementptr inbounds { i1 }, ptr %self, i32 0, i32 0
// CHECK:STDOUT:   %.loc15_33.2 = load i1, ptr %.loc15_33.1.is_positive, align 1
// CHECK:STDOUT:   br i1 %.loc15_33.2, label %and.rhs.loc15, label %and.result.loc15
// CHECK:STDOUT:
// CHECK:STDOUT: and.rhs.loc15:                                    ; preds = %entry
// CHECK:STDOUT:   %.loc15_55.1.is_positive = getelementptr inbounds { i1 }, ptr %other, i32 0, i32 0
// CHECK:STDOUT:   %.loc15_55.2 = load i1, ptr %.loc15_55.1.is_positive, align 1
// CHECK:STDOUT:   br label %and.result.loc15
// CHECK:STDOUT:
// CHECK:STDOUT: and.result.loc15:                                 ; preds = %and.rhs.loc15, %entry
// CHECK:STDOUT:   %0 = phi i1 [ false, %entry ], [ %.loc15_55.2, %and.rhs.loc15 ]
// CHECK:STDOUT:   %.loc15_69.1 = xor i1 %0, true
// CHECK:STDOUT:   br i1 %.loc15_69.1, label %or.rhs, label %or.result
// CHECK:STDOUT:
// CHECK:STDOUT: or.rhs:                                           ; preds = %and.result.loc15
// CHECK:STDOUT:   %.loc16_38.1.is_positive = getelementptr inbounds { i1 }, ptr %self, i32 0, i32 0
// CHECK:STDOUT:   %.loc16_38.2 = load i1, ptr %.loc16_38.1.is_positive, align 1
// CHECK:STDOUT:   %.loc16_30 = xor i1 %.loc16_38.2, true
// CHECK:STDOUT:   br i1 %.loc16_30, label %and.rhs.loc16, label %and.result.loc16
// CHECK:STDOUT:
// CHECK:STDOUT: and.rhs.loc16:                                    ; preds = %or.rhs
// CHECK:STDOUT:   %.loc16_64.1.is_positive = getelementptr inbounds { i1 }, ptr %other, i32 0, i32 0
// CHECK:STDOUT:   %.loc16_64.2 = load i1, ptr %.loc16_64.1.is_positive, align 1
// CHECK:STDOUT:   %.loc16_55 = xor i1 %.loc16_64.2, true
// CHECK:STDOUT:   br label %and.result.loc16
// CHECK:STDOUT:
// CHECK:STDOUT: and.result.loc16:                                 ; preds = %and.rhs.loc16, %or.rhs
// CHECK:STDOUT:   %1 = phi i1 [ false, %or.rhs ], [ %.loc16_55, %and.rhs.loc16 ]
// CHECK:STDOUT:   br label %or.result
// CHECK:STDOUT:
// CHECK:STDOUT: or.result:                                        ; preds = %and.result.loc16, %and.result.loc15
// CHECK:STDOUT:   %2 = phi i1 [ true, %and.result.loc15 ], [ %1, %and.result.loc16 ]
// CHECK:STDOUT:   %.loc16_77.2.is_positive = getelementptr inbounds { i1 }, ptr %return, i32 0, i32 0
// CHECK:STDOUT:   store i1 %2, ptr %.loc16_77.2.is_positive, align 1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Calculate(ptr sret({ i1 }) %return, ptr %a, ptr %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc21_10.2.temp = alloca { i1 }, align 8
// CHECK:STDOUT:   call void @Op(ptr %.loc21_10.2.temp, ptr %a)
// CHECK:STDOUT:   call void @Op.1(ptr %return, ptr %.loc21_10.2.temp, ptr %b)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder i1 true, { 2, 0, 1, 3, 4 }
