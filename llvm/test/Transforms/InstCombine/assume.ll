; RUN: opt < %s -instcombine -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind
declare void @llvm.assume(i1) #1

; Function Attrs: nounwind uwtable
define i32 @can1(i1 %a, i1 %b, i1 %c) {
entry:
  %and1 = and i1 %a, %b
  %and  = and i1 %and1, %c
  tail call void @llvm.assume(i1 %and)

; CHECK-LABEL: @can1
; CHECK: call void @llvm.assume(i1 %a)
; CHECK: call void @llvm.assume(i1 %b)
; CHECK: call void @llvm.assume(i1 %c)
; CHECK: ret i32

  ret i32 5
}

; Function Attrs: nounwind uwtable
define i32 @can2(i1 %a, i1 %b, i1 %c) {
entry:
  %v = or i1 %a, %b
  %w = xor i1 %v, 1
  tail call void @llvm.assume(i1 %w)

; CHECK-LABEL: @can2
; CHECK: %[[V1:[^ ]+]] = xor i1 %a, true
; CHECK: call void @llvm.assume(i1 %[[V1]])
; CHECK: %[[V2:[^ ]+]] = xor i1 %b, true
; CHECK: call void @llvm.assume(i1 %[[V2]])
; CHECK: ret i32

  ret i32 5
}

attributes #0 = { nounwind uwtable }
attributes #1 = { nounwind }

