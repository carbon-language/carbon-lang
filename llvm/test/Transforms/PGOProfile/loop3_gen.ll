; RUN: opt < %s -pgo-instr-gen -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__llvm_profile_name__Z15test_nested_foriii = private constant [22 x i8] c"_Z15test_nested_foriii"

define i32 @_Z15test_nested_foriii(i32 %r, i32 %s, i32 %t) {
entry:
; CHECK: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @__llvm_profile_name__Z15test_nested_foriii, i32 0, i32 0), i64 75296580464, i32 4, i32 3)
  br label %for.cond

for.cond:
  %i.0 = phi i32 [ 0, %entry ], [ %inc12, %for.inc11 ]
  %nested_for_sum.0 = phi i32 [ 1, %entry ], [ %nested_for_sum.1, %for.inc11 ]
  %cmp = icmp slt i32 %i.0, %r
  br i1 %cmp, label %for.body, label %for.end13

for.body:
  br label %for.cond1

for.cond1:
  %j.0 = phi i32 [ 0, %for.body ], [ %inc9, %for.inc8 ]
  %nested_for_sum.1 = phi i32 [ %nested_for_sum.0, %for.body ], [ %nested_for_sum.2, %for.inc8 ]
  %cmp2 = icmp slt i32 %j.0, %s
  br i1 %cmp2, label %for.body3, label %for.end10

for.body3:
  br label %for.cond4

for.cond4:
  %k.0 = phi i32 [ 0, %for.body3 ], [ %inc7, %for.inc ]
  %nested_for_sum.2 = phi i32 [ %nested_for_sum.1, %for.body3 ], [ %inc, %for.inc ]
  %cmp5 = icmp slt i32 %k.0, %t
  br i1 %cmp5, label %for.body6, label %for.end

for.body6:
  %inc = add nsw i32 %nested_for_sum.2, 1
  br label %for.inc

for.inc:
; CHECK: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @__llvm_profile_name__Z15test_nested_foriii, i32 0, i32 0), i64 75296580464, i32 4, i32 0)
  %inc7 = add nsw i32 %k.0, 1
  br label %for.cond4

for.end:
  br label %for.inc8

for.inc8:
; CHECK: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @__llvm_profile_name__Z15test_nested_foriii, i32 0, i32 0), i64 75296580464, i32 4, i32 1)
  %inc9 = add nsw i32 %j.0, 1
  br label %for.cond1

for.end10:
; CHECK: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @__llvm_profile_name__Z15test_nested_foriii, i32 0, i32 0), i64 75296580464, i32 4, i32 2)
  br label %for.inc11

for.inc11:
  %inc12 = add nsw i32 %i.0, 1
  br label %for.cond

for.end13:
  ret i32 %nested_for_sum.0
}
