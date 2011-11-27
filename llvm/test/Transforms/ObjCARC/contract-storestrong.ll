; RUN: opt -objc-arc-contract -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64"

declare i8* @objc_retain(i8*)
declare void @objc_release(i8*)

@x = external global i8*

; CHECK: define void @test0(
; CHECK: entry:
; CHECK-NEXT: call void @objc_storeStrong(i8** @x, i8* %p) nounwind
; CHECK-NEXT: ret void
define void @test0(i8* %p) {
entry:
  %0 = tail call i8* @objc_retain(i8* %p) nounwind
  %tmp = load i8** @x, align 8
  store i8* %0, i8** @x, align 8
  tail call void @objc_release(i8* %tmp) nounwind
  ret void
}

; Don't do this if the load is volatile.

;      CHECK: define void @test1(i8* %p) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = tail call i8* @objc_retain(i8* %p) nounwind
; CHECK-NEXT:   %tmp = load volatile i8** @x, align 8
; CHECK-NEXT:   store i8* %0, i8** @x, align 8
; CHECK-NEXT:   tail call void @objc_release(i8* %tmp) nounwind
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test1(i8* %p) {
entry:
  %0 = tail call i8* @objc_retain(i8* %p) nounwind
  %tmp = load volatile i8** @x, align 8
  store i8* %0, i8** @x, align 8
  tail call void @objc_release(i8* %tmp) nounwind
  ret void
}

; Don't do this if the store is volatile.

;      CHECK: define void @test2(i8* %p) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = tail call i8* @objc_retain(i8* %p) nounwind
; CHECK-NEXT:   %tmp = load i8** @x, align 8
; CHECK-NEXT:   store volatile i8* %0, i8** @x, align 8
; CHECK-NEXT:   tail call void @objc_release(i8* %tmp) nounwind
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test2(i8* %p) {
entry:
  %0 = tail call i8* @objc_retain(i8* %p) nounwind
  %tmp = load i8** @x, align 8
  store volatile i8* %0, i8** @x, align 8
  tail call void @objc_release(i8* %tmp) nounwind
  ret void
}
