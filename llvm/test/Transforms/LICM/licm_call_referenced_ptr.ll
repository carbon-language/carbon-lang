; RUN: opt < %s -basicaa -licm -S | FileCheck %s

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) nounwind

define i8 @"main"() {
entry:
  %A = alloca [64 x i8]
  %B = alloca [4 x i8]
  %A0 = getelementptr [64 x i8], [64 x i8]* %A, i32 0, i32 0
  %B0 = getelementptr [4 x i8], [4 x i8]* %B, i32 0, i32 0
  %B1 = getelementptr [4 x i8], [4 x i8]* %B, i32 0, i32 1
  %B2 = getelementptr [4 x i8], [4 x i8]* %B, i32 0, i32 2
  %B3 = getelementptr [4 x i8], [4 x i8]* %B, i32 0, i32 3
  store i8 0, i8* %A0
  store i8 32, i8* %B0
  store i8 73, i8* %B1
  store i8 74, i8* %B2
  store i8 75, i8* %B3
  br label %loop_begin

loop_begin:
; CHECK: loop_begin:
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* %A0, i8* %B0, i64 4, i32 4, i1 false)

  %b_val = load i8, i8* %B0
 
  ; *B is invariant in loop and limit_val must be hoisted
  %limit_val_1 = mul i8 %b_val, 3
  %limit_val = add i8 %limit_val_1, 67
  
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %A0, i8* %B0, i64 4, i32 4, i1 false)
  
  %exitcond = icmp ugt i8 164, %limit_val
  br i1 %exitcond, label %after_loop, label %loop_begin

after_loop:
  %b_val_result = load i8, i8* %B0
  ret i8 %b_val_result
}

