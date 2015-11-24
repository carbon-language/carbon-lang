; RUN: opt < %s -pgo-instr-gen -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__llvm_profile_name__Z9single_bbv = private constant [13 x i8] c"_Z9single_bbv"

define i32 @_Z9single_bbv() {
entry:
; CHECK: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @__llvm_profile_name__Z9single_bbv, i32 0, i32 0), i64 12884901887, i32 1, i32 0)
  ret i32 0
}
