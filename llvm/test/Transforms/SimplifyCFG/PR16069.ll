; RUN: opt < %s -simplifycfg -S | FileCheck %s

; CHECK-NOT: select
@b = extern_weak global i32
define i32 @foo(i1 %y) {
  br i1 %y, label %bb1, label %bb2
bb1:
  br label %bb3
bb2:
  br label %bb3
bb3:
  %cond.i = phi i32 [ 0, %bb1 ], [ srem (i32 1, i32 zext (i1 icmp eq (i32* @b, i32* null) to i32)), %bb2 ]
  ret i32 %cond.i
}
