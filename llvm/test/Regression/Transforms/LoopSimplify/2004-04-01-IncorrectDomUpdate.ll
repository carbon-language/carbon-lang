; RUN: llvm-as < %s | opt -loopsimplify -licm -disable-output

; This is PR306

void %NormalizeCoeffsVecFFE() {
entry:
        br label %loopentry.0

loopentry.0:            ; preds = %entry, %no_exit.0
        br bool false, label %loopentry.1, label %no_exit.0

no_exit.0:              ; preds = %loopentry.0
        br bool false, label %loopentry.0, label %loopentry.1

loopentry.1:            ; preds = %loopentry.0, %no_exit.0, %no_exit.1
        br bool false, label %no_exit.1, label %loopexit.1

no_exit.1:              ; preds = %loopentry.1
        %tmp.43 = seteq ushort 0, 0             ; <bool> [#uses=1]
        br bool %tmp.43, label %loopentry.1, label %loopexit.1

loopexit.1:             ; preds = %loopentry.1, %no_exit.1
        ret void
}
