; RUN: llc < %s -mtriple=armv7-none-linux-gnueabi | FileCheck %s
; Test that we correctly align elements when using va_arg

; CHECK: add	r0, r0, #7
; CHECK: bfc	r0, #0, #3

define i64 @f8(i32 %i, ...) nounwind optsize {
entry:
  %g = alloca i8*, align 4
  %g1 = bitcast i8** %g to i8*
  call void @llvm.va_start(i8* %g1)
  %0 = va_arg i8** %g, i64
  call void @llvm.va_end(i8* %g1)
  ret i64 %0
}

declare void @llvm.va_start(i8*) nounwind

declare void @llvm.va_end(i8*) nounwind
