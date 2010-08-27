; RUN: opt < %s -value-propagation -S | FileCheck %s
; PR4420

declare i1 @ext()
; CHECK: @foo
define i1 @foo() {
entry:
        %cond = tail call i1 @ext()             ; <i1> [#uses=2]
        br i1 %cond, label %bb1, label %bb2

bb1:            ; preds = %entry
        %cond2 = tail call i1 @ext()            ; <i1> [#uses=1]
        br i1 %cond2, label %bb3, label %bb2

bb2:            ; preds = %bb1, %entry
; CHECK-NOT: phi i1
        %cond_merge = phi i1 [ %cond, %entry ], [ false, %bb1 ]         ; <i1> [#uses=1]
; CHECK: ret i1 false
        ret i1 %cond_merge

bb3:            ; preds = %bb1
        %res = tail call i1 @ext()              ; <i1> [#uses=1]
; CHECK: ret i1 %res
        ret i1 %res
}