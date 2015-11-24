; RUN: opt < %s -pgo-instr-gen -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__llvm_profile_name__Z15test_simple_fori = private constant [20 x i8] c"_Z15test_simple_fori"

define i32 @_Z15test_simple_fori(i32 %n) {
entry:
  br label %for.cond

for.cond:
  %i = phi i32 [ 0, %entry ], [ %inc1, %for.inc ]
  %sum = phi i32 [ 1, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i, %n
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %inc = add nsw i32 %sum, 1
  br label %for.inc

for.inc:
; CHECK: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([20 x i8], [20 x i8]* @__llvm_profile_name__Z15test_simple_fori, i32 0, i32 0), i64 32052181608, i32 2, i32 0)
  %inc1 = add nsw i32 %i, 1
  br label %for.cond

for.end:
; CHECK: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([20 x i8], [20 x i8]* @__llvm_profile_name__Z15test_simple_fori, i32 0, i32 0), i64 32052181608, i32 2, i32 1)
  ret i32 %sum
}
