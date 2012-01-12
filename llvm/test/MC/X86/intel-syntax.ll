; RUN: llc %s -o %t1.s
; RUN: llvm-mc %t1.s -filetype=obj -o %t1.o
; RUN: llc %s -x86-asm-syntax=intel -o %t2.s
; RUN: llvm-mc %t2.s -x86-asm-syntax=intel -filetype=obj -o %t2.o
; RUN: diff %t1.o %t2.o

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.7.2"

define i32 @foo() nounwind uwtable ssp {
entry:
  %i = alloca i32, align 4
  %l = alloca i64, align 8
  %c = alloca i8, align 1
  store i32 3, i32* %i, align 4
  store i64 123, i64* %l, align 8
  store i8 97, i8* %c, align 1
  %0 = load i32* %i, align 4
  %add = add nsw i32 %0, 1
  store i32 %add, i32* %i, align 4
  %1 = load i32* %i, align 4
  ret i32 %1
}
