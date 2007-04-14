; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | \
; RUN:   grep icmp | wc -l | grep 1
; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | \
; RUN:   grep {icmp ugt} | wc -l | grep 1
; END.

; ModuleID = 'bugpoint-tooptimize.bc'
target datalayout = "e-p:32:32"
target endian = little
target pointersize = 32
target triple = "i686-pc-linux-gnu"
%r = external global [17 x int]         ; <[17 x int]*> [#uses=1]

implementation   ; Functions:

bool %print_pgm_cond_true(int %tmp12.reload, int* %tmp16.out) {
newFuncRoot:
        br label %cond_true

bb27.exitStub:          ; preds = %cond_true
        store int %tmp16, int* %tmp16.out
        ret bool true

cond_next23.exitStub:           ; preds = %cond_true
        store int %tmp16, int* %tmp16.out
        ret bool false

cond_true:              ; preds = %newFuncRoot
        %tmp15 = getelementptr [17 x int]* %r, int 0, int %tmp12.reload         ; <int*> [#uses=1]
        %tmp16 = load int* %tmp15               ; <int> [#uses=4]
        %tmp18 = icmp slt int %tmp16, -31               ; <bool> [#uses=1]
        %tmp21 = icmp sgt int %tmp16, 31                ; <bool> [#uses=1]
        %bothcond = or bool %tmp18, %tmp21              ; <bool> [#uses=1]
        br bool %bothcond, label %bb27.exitStub, label %cond_next23.exitStub
}

