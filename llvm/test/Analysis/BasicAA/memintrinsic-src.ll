; RUN: opt -S -gvn -dse < %s | FileCheck %s

declare void @llvm.memcpy.i8(i8*, i8*, i8, i32)

define i8 @test1() {
; CHECK: @test1
  %A = alloca i8
  %B = alloca i8

  store i8 2, i8* %B  ;; Not written to by memcpy

  call void @llvm.memcpy.i8(i8* %A, i8* %B, i8 -1, i32 0)

  %C = load i8* %B
  ret i8 %C
; CHECK-NEXT: ret i8 2
}
