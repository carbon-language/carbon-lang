; RUN: opt < %s -jump-threading -S | FileCheck %s

define i32 @test_jump_threading(i32* %arg1, i32 %arg2) {
entry:
  %cmp = icmp slt i32 %arg2, 0
  br i1 %cmp, label %land.lhs.true, label %lor.rhs

land.lhs.true:
  %ident = getelementptr inbounds i32 * %arg1, i64 0
  %0 = load i32* %ident, align 4
  %cmp1 = icmp eq i32 %0, 1
  br i1 %cmp1, label %lor.end, label %lor.rhs

; CHECK: br i1 %cmp1, label %lor.end, label %lor.rhs.thread

; CHECK: lor.rhs.thread:
; CHECK-NEXT: br label %lor.end

lor.rhs:
  %cmp2 = icmp sgt i32 %arg2, 0
  br i1 %cmp2, label %land.rhs, label %lor.end

land.rhs:
  %ident3 = getelementptr inbounds i32 * %arg1, i64 0
  %1 = load i32* %ident3, align 4
  %cmp4 = icmp eq i32 %1, 2
  br label %lor.end

lor.end:
  %2 = phi i1 [ true, %land.lhs.true ], [ false, %lor.rhs ], [ %cmp4, %land.rhs ]
  %lor.ext = zext i1 %2 to i32
  ret i32 %lor.ext
}
