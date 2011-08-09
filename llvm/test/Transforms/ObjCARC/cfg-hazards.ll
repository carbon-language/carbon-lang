; RUN: opt -S -objc-arc < %s | FileCheck %s
; rdar://9503416

; Detect loop boundaries and don't move retains and releases
; across them.

declare void @use_pointer(i8*)
declare i8* @objc_retain(i8*)
declare void @objc_release(i8*)

; CHECK: define void @test0(
; CHECK:   call i8* @objc_retain(
; CHECK: for.body:
; CHECK-NOT: @objc
; CHECK: for.end:
; CHECK:   call void @objc_release(
; CHECK: }
define void @test0(i8* %digits) {
entry:
  %tmp1 = call i8* @objc_retain(i8* %digits) nounwind
  call void @use_pointer(i8* %digits)
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %upcDigitIndex.01 = phi i64 [ 2, %entry ], [ %inc, %for.body ]
  call void @use_pointer(i8* %digits)
  %inc = add i64 %upcDigitIndex.01, 1
  %cmp = icmp ult i64 %inc, 12
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  call void @objc_release(i8* %digits) nounwind, !clang.imprecise_release !0
  ret void
}

; CHECK: define void @test1(
; CHECK:   call i8* @objc_retain(
; CHECK: for.body:
; CHECK-NOT: @objc
; CHECK: for.end:
; CHECK:   void @objc_release(
; CHECK: }
define void @test1(i8* %digits) {
entry:
  %tmp1 = call i8* @objc_retain(i8* %digits) nounwind
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %upcDigitIndex.01 = phi i64 [ 2, %entry ], [ %inc, %for.body ]
  call void @use_pointer(i8* %digits)
  call void @use_pointer(i8* %digits)
  %inc = add i64 %upcDigitIndex.01, 1
  %cmp = icmp ult i64 %inc, 12
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  call void @objc_release(i8* %digits) nounwind, !clang.imprecise_release !0
  ret void
}

; CHECK: define void @test2(
; CHECK:   call i8* @objc_retain(
; CHECK: for.body:
; CHECK-NOT: @objc
; CHECK: for.end:
; CHECK:   void @objc_release(
; CHECK: }
define void @test2(i8* %digits) {
entry:
  %tmp1 = call i8* @objc_retain(i8* %digits) nounwind
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %upcDigitIndex.01 = phi i64 [ 2, %entry ], [ %inc, %for.body ]
  call void @use_pointer(i8* %digits)
  %inc = add i64 %upcDigitIndex.01, 1
  %cmp = icmp ult i64 %inc, 12
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  call void @use_pointer(i8* %digits)
  call void @objc_release(i8* %digits) nounwind, !clang.imprecise_release !0
  ret void
}

!0 = metadata !{}
