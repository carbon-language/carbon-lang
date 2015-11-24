; RUN: opt < %s -pgo-instr-gen -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__llvm_profile_name__Z9test_br_1i = private constant [13 x i8] c"_Z9test_br_1i"

define i32 @_Z9test_br_1i(i32 %i) {
entry:
  %cmp = icmp sgt i32 %i, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
; CHECK: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @__llvm_profile_name__Z9test_br_1i, i32 0, i32 0), i64 23925403969, i32 2, i32 1)
  %add = add nsw i32 %i, 2
  br label %if.end

if.end:
  %retv = phi i32 [ %add, %if.then ], [ %i, %entry ]
; CHECK: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @__llvm_profile_name__Z9test_br_1i, i32 0, i32 0), i64 23925403969, i32 2, i32 0)
  ret i32 %retv
}
