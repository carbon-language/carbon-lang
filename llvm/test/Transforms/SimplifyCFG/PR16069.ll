; RUN: opt < %s -simplifycfg -S | FileCheck %s

@b = extern_weak global i32

define i32 @foo(i1 %y) {
; CHECK: define i32 @foo(i1 %y) {
  br i1 %y, label %bb1, label %bb2
bb1:
  br label %bb3
bb2:
  br label %bb3
bb3:
  %cond.i = phi i32 [ 0, %bb1 ], [ srem (i32 1, i32 zext (i1 icmp eq (i32* @b, i32* null) to i32)), %bb2 ]
; CHECK: phi i32 {{.*}} srem (i32 1, i32 zext (i1 icmp eq (i32* @b, i32* null) to i32)), %bb2
  ret i32 %cond.i
}

define i32 @foo2(i1 %x) {
; CHECK: define i32 @foo2(i1 %x) {
bb0:
  br i1 %x, label %bb1, label %bb2
bb1:
  br label %bb2
bb2:
  %cond = phi i32 [ 0, %bb1 ], [ srem (i32 1, i32 zext (i1 icmp eq (i32* @b, i32* null) to i32)), %bb0 ]
; CHECK:  %cond = phi i32 [ 0, %bb1 ], [ srem (i32 1, i32 zext (i1 icmp eq (i32* @b, i32* null) to i32)), %bb0 ]
  ret i32 %cond
}
