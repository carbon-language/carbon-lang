; RUN: opt -gvn -S < %s | FileCheck %s

define i32 @f1(i32 %x) {
  ; CHECK-LABEL: define i32 @f1(
bb0:
  %cmp = icmp eq i32 %x, 0
  br i1 %cmp, label %bb2, label %bb1
bb1:
  br label %bb2
bb2:
  %cond = phi i32 [ %x, %bb0 ], [ 0, %bb1 ]
  %foo = add i32 %cond, %x
  ret i32 %foo
  ; CHECK: bb2:
  ; CHECK: ret i32 %x
}

define i32 @f2(i32 %x) {
  ; CHECK-LABEL: define i32 @f2(
bb0:
  %cmp = icmp ne i32 %x, 0
  br i1 %cmp, label %bb1, label %bb2
bb1:
  br label %bb2
bb2:
  %cond = phi i32 [ %x, %bb0 ], [ 0, %bb1 ]
  %foo = add i32 %cond, %x
  ret i32 %foo
  ; CHECK: bb2:
  ; CHECK: ret i32 %x
}

define i32 @f3(i32 %x) {
  ; CHECK-LABEL: define i32 @f3(
bb0:
  switch i32 %x, label %bb1 [ i32 0, label %bb2]
bb1:
  br label %bb2
bb2:
  %cond = phi i32 [ %x, %bb0 ], [ 0, %bb1 ]
  %foo = add i32 %cond, %x
  ret i32 %foo
  ; CHECK: bb2:
  ; CHECK: ret i32 %x
}

declare void @g(i1)
define void @f4(i8 * %x)  {
; CHECK-LABEL: define void @f4(
bb0:
  %y = icmp eq i8* null, %x
  br i1 %y, label %bb2, label %bb1
bb1:
  br label %bb2
bb2:
  %zed = icmp eq i8* null, %x
  call void @g(i1 %zed)
; CHECK: call void @g(i1 %y)
  ret void
}

define double @fcmp_oeq(double %x, double %y) {
entry:
  %cmp = fcmp oeq double %y, 2.0
  br i1 %cmp, label %if, label %return

if:
  %div = fdiv double %x, %y
  br label %return

return:
  %retval.0 = phi double [ %div, %if ], [ %x, %entry ]
  ret double %retval.0

; CHECK-LABEL: define double @fcmp_oeq(
; CHECK: %div = fdiv double %x, 2.000000e+00
}

define double @fcmp_une(double %x, double %y) {
entry:
  %cmp = fcmp une double %y, 2.0
  br i1 %cmp, label %return, label %else

else:
  %div = fdiv double %x, %y
  br label %return

return:
  %retval.0 = phi double [ %div, %else ], [ %x, %entry ]
  ret double %retval.0

; CHECK-LABEL: define double @fcmp_une(
; CHECK: %div = fdiv double %x, 2.000000e+00
}

