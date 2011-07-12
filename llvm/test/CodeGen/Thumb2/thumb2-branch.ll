; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mattr=+thumb2 | FileCheck %s
; If-conversion defeats the purpose of this test, which is to check conditional
; branch generation, so use memory barrier instruction to make sure it doesn't
; happen and we get actual branches.

define i32 @f1(i32 %a, i32 %b, i32* %v) {
entry:
; CHECK: f1:
; CHECK: bne LBB
        %tmp = icmp eq i32 %a, %b               ; <i1> [#uses=1]
        br i1 %tmp, label %cond_true, label %return

cond_true:              ; preds = %entry
        call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 false)
        store i32 0, i32* %v
        ret i32 0

return:         ; preds = %entry
        call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 false)
        ret i32 1
}

define i32 @f2(i32 %a, i32 %b, i32* %v) {
entry:
; CHECK: f2:
; CHECK: bge LBB
        %tmp = icmp slt i32 %a, %b              ; <i1> [#uses=1]
        br i1 %tmp, label %cond_true, label %return

cond_true:              ; preds = %entry
        call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 false)
        store i32 0, i32* %v
        ret i32 0

return:         ; preds = %entry
        call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 false)
        ret i32 1
}

define i32 @f3(i32 %a, i32 %b, i32* %v) {
entry:
; CHECK: f3:
; CHECK: bhs LBB
        %tmp = icmp ult i32 %a, %b              ; <i1> [#uses=1]
        br i1 %tmp, label %cond_true, label %return

cond_true:              ; preds = %entry
        call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 false)
        store i32 0, i32* %v
        ret i32 0

return:         ; preds = %entry
        call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 false)
        ret i32 1
}

define i32 @f4(i32 %a, i32 %b, i32* %v) {
entry:
; CHECK: f4:
; CHECK: blo LBB
        %tmp = icmp ult i32 %a, %b              ; <i1> [#uses=1]
        br i1 %tmp, label %return, label %cond_true

cond_true:              ; preds = %entry
        call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 false)
        store i32 0, i32* %v
        ret i32 0

return:         ; preds = %entry
        call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 false)
        ret i32 1
}

declare void @llvm.memory.barrier(i1, i1, i1, i1, i1) nounwind
