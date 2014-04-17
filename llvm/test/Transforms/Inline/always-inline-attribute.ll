; RUN: opt < %s -O3 -S | FileCheck %s
@gv = external global i32

define i32 @main() nounwind {
; CHECK-NOT: call i32 @foo
  %1 = load i32* @gv, align 4
  %2 = tail call i32 @foo(i32 %1)
  unreachable
}

define internal i32 @foo(i32) alwaysinline {
  br label %2

; <label>:2                                       ; preds = %8, %1
  %3 = phi i32 [ %0, %1 ], [ %10, %8 ]
  %4 = phi i8* [ blockaddress(@foo, %2), %1 ], [ %6, %8 ]
  %5 = icmp eq i32 %3, 1
  %6 = select i1 %5, i8* blockaddress(@foo, %8), i8* %4
  %7 = add nsw i32 %3, -1
  br label %8

; <label>:8                                       ; preds = %8, %2
  %9 = phi i32 [ %7, %2 ], [ %10, %8 ]
  %10 = add nsw i32 %9, -1
  indirectbr i8* %6, [label %2, label %8]
}
