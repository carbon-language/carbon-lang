; RUN: opt -ipsccp -S %s | FileCheck %s

; Test 1.
; Both arguments and return value of @callee can be tracked. The inferred range
; can be added to call sites.
define internal noundef i32 @callee(i32 %x) {
; CHECK-LABEL: @callee(
; CHECK-NEXT:    ret i32 [[X:%.*]]
;
  ret i32 %x
}

define i32 @caller1() {
; CHECK-LABEL: @caller1(
; CHECK-NEXT:    [[C1:%.*]] = call i32 @callee(i32 10), !range [[RANGE_10_21:![0-9]+]]
; CHECK-NEXT:    [[C2:%.*]] = call i32 @callee(i32 20), !range [[RANGE_10_21]]
; CHECK-NEXT:    [[A:%.*]] = add i32 [[C1]], [[C2]]
; CHECK-NEXT:    ret i32 [[A]]
;
  %c1 = call i32 @callee(i32 10)
  %c2 = call i32 @callee(i32 20)
  %a = add i32 %c1, %c2
  ret i32 %a
}

define i32 @caller2(i32 %x) {
; CHECK-LABEL: @caller2(
; CHECK-NEXT:    [[X_15:%.*]] = and i32 [[X:%.*]], 15
; CHECK-NEXT:    [[C:%.*]] = call i32 @callee(i32 [[X_15]]), !range [[RANGE_10_21]]
; CHECK-NEXT:    ret i32 [[C]]
;
  %x.15 = and i32 %x, 15
  %c = call i32 @callee(i32 %x.15)
  ret i32 %c
}

; Test 2.
; The return value of @callee2 can be tracked, but arguments cannot, because
; it is passed to @use_cb1. We cannot infer a range for the return value, no
; metadata should be added.

declare void @use_cb1(i32 (i32)*)

define internal noundef i32 @callee2(i32 %x) {
; CHECK-LABEL: @callee2(
; CHECK-NEXT:    ret i32 [[X:%.*]]
;
  ret i32 %x
}

define void @caller_cb1() {
; CHECK-LABEL: @caller_cb1(
; CHECK-NEXT:    [[C1:%.*]] = call i32 @callee2(i32 9)
; CHECK-NOT:   !range
; CHECK-NEXT:    [[C2:%.*]] = call i32 @callee2(i32 10)
; CHECK-NOT:   !range
; CHECK-NEXT:    call void @use_cb1(i32 (i32)* @callee2)
; CHECK-NEXT:    ret void
;
  %c1 = call i32 @callee2(i32 9)
  %c2 = call i32 @callee2(i32 10)
  call void @use_cb1(i32 (i32)* @callee2)
  ret void
}

; Test 3.
; The return value can be tracked and it the result range ([500, 601) does not
; depend on the arguments, which cannot be tracked because @callee3 is passed
; to @use_cb2. The result range can be added to the call sites of @callee.

declare void @use_cb2(i32 (i32)*)

define internal noundef i32 @callee3(i32 %x) {
; CHECK-LABEL: @callee3(
; CHECK-NEXT:    [[C:%.*]] = icmp eq i32 [[X:%.*]], 10
; CHECK-NEXT:    [[S:%.*]] = select i1 [[C]], i32 500, i32 600
; CHECK-NEXT:    ret i32 [[S]]
;
  %c = icmp eq i32 %x, 10
  %s = select i1 %c, i32 500, i32 600
  ret i32 %s
}

define void @caller_cb2() {
; CHECK-LABEL: @caller_cb2(
; CHECK-NEXT:    [[C1:%.*]] = call i32 @callee3(i32 9), !range [[RANGE_500_601:![0-9]+]]
; CHECK-NEXT:    [[C2:%.*]] = call i32 @callee3(i32 10), !range [[RANGE_500_601]]
; CHECK-NEXT:    call void @use_cb2(i32 (i32)* @callee3)
; CHECK-NEXT:    ret void
;
  %c1 = call i32 @callee3(i32 9)
  %c2 = call i32 @callee3(i32 10)
  call void @use_cb2(i32 (i32)* @callee3)
  ret void
}

; Test 4.
; The return value of @callee4 can be tracked, but depends on an argument which
; cannot be tracked. No result range can be inferred.

declare void @use_cb3(i32 (i32, i32)*)

define internal noundef i32 @callee4(i32 %x, i32 %y) {
; CHECK-LABEL: @callee4(
; CHECK-NEXT:    [[C:%.*]] = icmp eq i32 [[X:%.*]], 10
; CHECK-NEXT:    [[S:%.*]] = select i1 [[C]], i32 500, i32 [[Y:%.*]]
; CHECK-NEXT:    ret i32 [[S]]
;
  %c = icmp eq i32 %x, 10
  %s = select i1 %c, i32 500, i32 %y
  ret i32 %s
}

define void @caller_cb3() {
; CHECK-LABEL: @caller_cb3(
; CHECK-NEXT:    [[C1:%.*]] = call i32 @callee4(i32 11, i32 30)
; CHECK-NOT:   !range
; CHECK-NEXT:    [[C2:%.*]] = call i32 @callee4(i32 12, i32 40)
; CHECK-NOT:   !range
; CHECK-NEXT:    call void @use_cb3(i32 (i32, i32)* @callee4)
; CHECK-NEXT:    ret void
;
  %c1 = call i32 @callee4(i32 11, i32 30)
  %c2 = call i32 @callee4(i32 12, i32 40)
  call void @use_cb3(i32 (i32, i32)* @callee4)
  ret void
}

; Test 5.
; Range for the return value of callee5 includes undef. No range metadata
; should be added at call sites.
define internal noundef i32 @callee5(i32 %x, i32 %y) {
; CHECK-LABEL: @callee5(
; CHECK-NEXT:    [[C:%.*]] = icmp slt i32 [[X:%.*]], 15
; CHECK-NEXT:    br i1 [[C]], label [[BB1:%.*]], label [[BB2:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    br label [[EXIT:%.*]]
; CHECK:       bb2:
; CHECK-NEXT:    br label [[EXIT]]
; CHECK:       exit:
; CHECK-NEXT:    [[RES:%.*]] = phi i32 [ [[Y:%.*]], [[BB1]] ], [ undef, [[BB2]] ]
; CHECK-NEXT:    ret i32 [[RES]]
;
  %c = icmp slt i32 %x, 15
  br i1 %c, label %bb1, label %bb2

bb1:
  br label %exit

bb2:
  br label %exit

exit:
  %res = phi i32 [ %y, %bb1 ], [ undef, %bb2]
  ret i32 %res
}

define i32 @caller5() {
; CHECK-LABEL: @caller5(
; CHECK-NEXT:    [[C1:%.*]] = call i32 @callee5(i32 10, i32 100)
; CHECK-NOT:   !range
; CHECK-NEXT:    [[C2:%.*]] = call i32 @callee5(i32 20, i32 200)
; CHECK-NOT:   !range
; CHECK-NEXT:    [[A:%.*]] = add i32 [[C1]], [[C2]]
; CHECK-NEXT:    ret i32 [[A]]
;
  %c1 = call i32 @callee5(i32 10, i32 100)
  %c2 = call i32 @callee5(i32 20, i32 200)
  %a = add i32 %c1, %c2
  ret i32 %a
}

; CHECK: [[RANGE_10_21]] = !{i32 0, i32 21}
; CHECK: [[RANGE_500_601]] = !{i32 500, i32 601}
