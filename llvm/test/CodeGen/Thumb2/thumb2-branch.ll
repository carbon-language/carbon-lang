; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mattr=+thumb2 | FileCheck %s

define void @f1(i32 %a, i32 %b, i32* %v) {
entry:
; CHECK: f1:
; CHECK: bne LBB
        %tmp = icmp eq i32 %a, %b               ; <i1> [#uses=1]
        br i1 %tmp, label %cond_true, label %return

cond_true:              ; preds = %entry
        store i32 0, i32* %v
        ret void

return:         ; preds = %entry
        ret void
}

define void @f2(i32 %a, i32 %b, i32* %v) {
entry:
; CHECK: f2:
; CHECK: bge LBB
        %tmp = icmp slt i32 %a, %b              ; <i1> [#uses=1]
        br i1 %tmp, label %cond_true, label %return

cond_true:              ; preds = %entry
        store i32 0, i32* %v
        ret void

return:         ; preds = %entry
        ret void
}

define void @f3(i32 %a, i32 %b, i32* %v) {
entry:
; CHECK: f3:
; CHECK: bhs LBB
        %tmp = icmp ult i32 %a, %b              ; <i1> [#uses=1]
        br i1 %tmp, label %cond_true, label %return

cond_true:              ; preds = %entry
        store i32 0, i32* %v
        ret void

return:         ; preds = %entry
        ret void
}

define void @f4(i32 %a, i32 %b, i32* %v) {
entry:
; CHECK: f4:
; CHECK: blo LBB
        %tmp = icmp ult i32 %a, %b              ; <i1> [#uses=1]
        br i1 %tmp, label %return, label %cond_true

cond_true:              ; preds = %entry
        store i32 0, i32* %v
        ret void

return:         ; preds = %entry
        ret void
}
