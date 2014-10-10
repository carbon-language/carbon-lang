; RUN: opt < %s -simplifycfg -S | FileCheck %s

define void @test1(i1 %C, i1* %BP) {
; CHECK-LABEL: @test1(
; CHECK: entry:
; CHECK-NEXT: ret void
entry:
        br i1 %C, label %T, label %F
T:
        store i1 %C, i1* %BP
        unreachable
F:
        ret void
}

define void @test2() {
; CHECK-LABEL: @test2(
; CHECK: entry:
; CHECK-NEXT: call void @test2()
; CHECK-NEXT: ret void
entry:
        invoke void @test2( )
                        to label %N unwind label %U
U:
        unreachable
N:
        ret void
}

define i32 @test3(i32 %v) {
; CHECK-LABEL: @test3(
; CHECK: entry:
; CHECK-NEXT: [[CMP:%[A-Za-z0-9]+]] = icmp eq i32 %v, 2
; CHECK-NEXT: select i1 [[CMP]], i32 2, i32 1
; CHECK-NEXT: ret
entry:
        switch i32 %v, label %default [
                 i32 1, label %U
                 i32 2, label %T
        ]
default:
        ret i32 1
U:
        unreachable
T:
        ret i32 2
}

; PR9450
define i32 @test4(i32 %v, i32 %w) {
; CHECK: entry:
; CHECK-NEXT:  switch i32 %v, label %T [
; CHECK-NEXT:    i32 3, label %V
; CHECK-NEXT:    i32 2, label %U
; CHECK-NEXT:  ]

entry:
        br label %SWITCH
V:
        ret i32 7
SWITCH:
        switch i32 %v, label %default [
                 i32 1, label %T
                 i32 2, label %U
                 i32 3, label %V
        ]
default:
        unreachable
U:
        ret i32 %w
T:
        ret i32 2
}


;; We can either convert the following control-flow to a select or remove the
;; unreachable control flow because of the undef store of null. Make sure we do
;; the latter.

define void @test5(i1 %cond, i8* %ptr) {

; CHECK-LABEL: test5
; CHECK: entry:
; CHECK-NOT: select
; CHECK:  store i8 2, i8* %ptr
; CHECK:  ret

entry:
  br i1 %cond, label %bb1, label %bb3

bb3:
 br label %bb2

bb1:
 br label %bb2

bb2:
  %ptr.2 = phi i8* [ %ptr, %bb3 ], [ null, %bb1 ]
  store i8 2, i8* %ptr.2, align 8
  ret void
}

; CHECK-LABEL: test6
; CHECK: entry:
; CHECK-NOT: select
; CHECK:  store i8 2, i8* %ptr
; CHECK:  ret

define void @test6(i1 %cond, i8* %ptr) {
entry:
  br i1 %cond, label %bb1, label %bb2

bb1:
  br label %bb2

bb2:
  %ptr.2 = phi i8* [ %ptr, %entry ], [ null, %bb1 ]
  store i8 2, i8* %ptr.2, align 8
  ret void
}
