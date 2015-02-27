; RUN: opt %s -simplifycfg -S | FileCheck %s --check-prefix=NORMAL
; RUN: opt %s -simplifycfg -S -bonus-inst-threshold=2 | FileCheck %s --check-prefix=AGGRESSIVE

define i32 @foo(i32 %a, i32 %b, i32 %c, i32 %d, i32* %input) {
; NORMAL-LABEL: @foo(
; AGGRESSIVE-LABEL: @foo(
entry:
  %cmp = icmp sgt i32 %d, 3
  br i1 %cmp, label %cond.end, label %lor.lhs.false
; NORMAL: br i1
; AGGRESSIVE: br i1

lor.lhs.false:
  %mul = shl i32 %c, 1
  %add = add nsw i32 %mul, %a
  %cmp1 = icmp slt i32 %add, %b
  br i1 %cmp1, label %cond.false, label %cond.end
; NORMAL: br i1
; AGGRESSIVE-NOT: br i1

cond.false:
  %0 = load i32, i32* %input, align 4
  br label %cond.end

cond.end:
  %cond = phi i32 [ %0, %cond.false ], [ 0, %lor.lhs.false ], [ 0, %entry ]
  ret i32 %cond
}
