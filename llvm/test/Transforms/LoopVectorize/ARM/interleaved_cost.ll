; RUN: opt -loop-vectorize -force-vector-width=8 -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s --check-prefix=VF_8
; RUN: opt -loop-vectorize -force-vector-width=16 -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s --check-prefix=VF_16
; REQUIRES: asserts

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "armv8--linux-gnueabihf"

%i8.2 = type {i8, i8}
define void @i8_factor_2(%i8.2* %data, i64 %n) {
entry:
  br label %for.body

; VF_8-LABEL:  Checking a loop in "i8_factor_2"
; VF_8:          Found an estimated cost of 2 for VF 8 For instruction: %tmp2 = load i8, i8* %tmp0, align 1
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: %tmp3 = load i8, i8* %tmp1, align 1
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: store i8 0, i8* %tmp0, align 1
; VF_8-NEXT:     Found an estimated cost of 2 for VF 8 For instruction: store i8 0, i8* %tmp1, align 1
; VF_16-LABEL: Checking a loop in "i8_factor_2"
; VF_16:         Found an estimated cost of 2 for VF 16 For instruction: %tmp2 = load i8, i8* %tmp0, align 1
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: %tmp3 = load i8, i8* %tmp1, align 1
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: store i8 0, i8* %tmp0, align 1
; VF_16-NEXT:    Found an estimated cost of 2 for VF 16 For instruction: store i8 0, i8* %tmp1, align 1
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr inbounds %i8.2, %i8.2* %data, i64 %i, i32 0
  %tmp1 = getelementptr inbounds %i8.2, %i8.2* %data, i64 %i, i32 1
  %tmp2 = load i8, i8* %tmp0, align 1
  %tmp3 = load i8, i8* %tmp1, align 1
  store i8 0, i8* %tmp0, align 1
  store i8 0, i8* %tmp1, align 1
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}
