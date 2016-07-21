; RUN: opt -S -gvn-hoist < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @test1(i1 %b, i32* %x) {
entry:
  br i1 %b, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  store i32 2, i32* %x, align 4, !tbaa !1
  br label %if.end

if.else:                                          ; preds = %entry
  store i32 2, i32* %x, align 4, !tbaa !5
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}
; CHECK-LABEL: define void @test1(
; CHECK: store i32 2, i32* %x, align 4
; CHECK-NEXT: br i1 %b

!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C++ TBAA"}
!5 = !{!6, !6, i64 0}
!6 = !{!"_ZTS1e", !3, i64 0}
