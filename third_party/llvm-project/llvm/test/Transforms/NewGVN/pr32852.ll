; Make sure GVN doesn't incorrectly think the branch terminating
; bb2 has a constant condition.
; RUN: opt -S -newgvn %s | FileCheck %s

@a = common global i32 0
@patatino = private unnamed_addr constant [3 x i8] c"0\0A\00"

define void @tinkywinky() {
bb:
  %tmp = load i32, i32* @a
  %tmp1 = icmp sge i32 %tmp, 0
  br i1 %tmp1, label %bb2, label %bb7
bb2:
  %tmp4 = icmp sgt i32 %tmp, 0
; CHECK: br i1 %tmp4, label %bb5, label %bb7
  br i1 %tmp4, label %bb5, label %bb7
bb5:
  %tmp6 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @patatino, i32 0, i32 0))
  br label %bb7
bb7:
  ret void
}

declare i32 @printf(i8*, ...)
