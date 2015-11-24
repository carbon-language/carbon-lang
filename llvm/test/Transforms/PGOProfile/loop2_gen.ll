; RUN: opt < %s -pgo-instr-gen -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__llvm_profile_name__Z13test_do_whilei = private constant [18 x i8] c"_Z13test_do_whilei"

define i32 @_Z13test_do_whilei(i32 %n) {
entry:
  br label %do.body

do.body:
  %i.0 = phi i32 [ 0, %entry ], [ %inc1, %do.cond ]
  %sum = phi i32 [ 1, %entry ], [ %inc, %do.cond ]
; CHECK: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([18 x i8], [18 x i8]* @__llvm_profile_name__Z13test_do_whilei, i32 0, i32 0), i64 29706172832, i32 2, i32 0)
  %inc = add nsw i32 %sum, 1
  br label %do.cond

do.cond:
  %inc1 = add nsw i32 %i.0, 1
  %cmp = icmp slt i32 %i.0, %n
  br i1 %cmp, label %do.body, label %do.end

do.end:
; CHECK: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([18 x i8], [18 x i8]* @__llvm_profile_name__Z13test_do_whilei, i32 0, i32 0), i64 29706172832, i32 2, i32 1)
  ret i32 %inc
}
