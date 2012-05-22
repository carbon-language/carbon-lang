; Test that the StrCpyOptimizer works correctly
; RUN: opt < %s -simplify-libcalls -S | FileCheck %s

; This transformation requires the pointer size, as it assumes that size_t is
; the size of a pointer.
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"

@hello = constant [6 x i8] c"hello\00"

declare i8* @strcpy(i8*, i8*)

declare i8* @__strcpy_chk(i8*, i8*, i32) nounwind

declare i32 @llvm.objectsize.i32(i8*, i1) nounwind readonly

; rdar://6839935

define i32 @t1() {
; CHECK: @t1
  %target = alloca [1024 x i8]
  %arg1 = getelementptr [1024 x i8]* %target, i32 0, i32 0
  %arg2 = getelementptr [6 x i8]* @hello, i32 0, i32 0
  %rslt1 = call i8* @strcpy( i8* %arg1, i8* %arg2 )
; CHECK: @llvm.memcpy.p0i8.p0i8.i32
  ret i32 0
}

define i32 @t2() {
; CHECK: @t2
  %target = alloca [1024 x i8]
  %arg1 = getelementptr [1024 x i8]* %target, i32 0, i32 0
  %arg2 = getelementptr [6 x i8]* @hello, i32 0, i32 0
  %tmp1 = call i32 @llvm.objectsize.i32(i8* %arg1, i1 false)
  %rslt1 = call i8* @__strcpy_chk(i8* %arg1, i8* %arg2, i32 %tmp1)
; CHECK: @__memcpy_chk
  ret i32 0
}
