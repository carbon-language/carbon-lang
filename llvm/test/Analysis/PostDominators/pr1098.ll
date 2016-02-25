; RUN: opt < %s -postdomtree -analyze | FileCheck %s
; PR932

define void @foo(i1 %x) {
; CHECK: entry
entry:
        br i1 %x, label %bb1, label %bb0
bb0:            ; preds = %entry, bb0
        br label %bb0
bb1:            ; preds = %entry
        br label %bb2
bb2:            ; preds = %bb1
        ret void
}

