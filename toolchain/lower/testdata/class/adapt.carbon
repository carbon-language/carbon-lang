// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- adapt_class.carbon

library "adapt_class" api;

class PairOfInts {
  var a: i32;
  var b: i32;

  fn Make() -> Self {
    return {.a = 1, .b = 2};
  }
}

class PairAdapter {
  adapt PairOfInts;

  fn Make() -> Self {
    return PairOfInts.Make() as Self;
  }

  fn GetB[self: Self]() -> i32 {
    let pi: PairOfInts = self as PairOfInts;
    return pi.b;
  }
}

fn Use() -> i32 {
  var pa: PairAdapter = PairAdapter.Make();
  return pa.GetB();
}

// --- adapt_int.carbon

library "adapt_int" api;

class Int {
  adapt i32;
}

fn DoStuff(a: Int) -> Int {
  return a;
}

// CHECK:STDOUT: ; ModuleID = 'adapt_class.carbon'
// CHECK:STDOUT: source_filename = "adapt_class.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Make(ptr sret({ i32, i32 }) %return) {
// CHECK:STDOUT:   %a = getelementptr inbounds { i32, i32 }, ptr %return, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %a, align 4
// CHECK:STDOUT:   %b = getelementptr inbounds { i32, i32 }, ptr %return, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %b, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Make.1(ptr sret({ i32, i32 }) %return) {
// CHECK:STDOUT:   call void @Make(ptr %return)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @GetB(ptr %self) {
// CHECK:STDOUT:   %b = getelementptr inbounds { i32, i32 }, ptr %self, i32 0, i32 1
// CHECK:STDOUT:   %1 = load i32, ptr %b, align 4
// CHECK:STDOUT:   ret i32 %1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @Use() {
// CHECK:STDOUT:   %pa = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   call void @Make.1(ptr %pa)
// CHECK:STDOUT:   %GetB = call i32 @GetB(ptr %pa)
// CHECK:STDOUT:   ret i32 %GetB
// CHECK:STDOUT: }
// CHECK:STDOUT: ; ModuleID = 'adapt_int.carbon'
// CHECK:STDOUT: source_filename = "adapt_int.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @DoStuff(i32 %a) {
// CHECK:STDOUT:   ret i32 %a
// CHECK:STDOUT: }
