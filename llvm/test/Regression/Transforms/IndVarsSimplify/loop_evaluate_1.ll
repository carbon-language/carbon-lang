; RUN: llvm-as < %s | opt -indvars -adce -simplifycfg | llvm-dis | not grep br
;
; Testcase distilled from 256.bzip2

int %main() {
entry:
        br label %loopentry

loopentry:              ; preds = %entry, %loopentry
        %indvar1 = phi uint [ 0, %entry ], [ %indvar.next2, %loopentry ]                ; <uint> [#uses=1]
        %h.0 = phi int [ %tmp.2, %loopentry ], [ 4, %entry ]            ; <int> [#uses=1]
        %tmp.1 = mul int %h.0, 3                ; <int> [#uses=1]
        %tmp.2 = add int %tmp.1, 1              ; <int> [#uses=1]
        %indvar.next2 = add uint %indvar1, 1            ; <uint> [#uses=2]
        %exitcond3 = setne uint %indvar.next2, 4                ; <bool> [#uses=1]
        br bool %exitcond3, label %loopentry, label %loopexit

loopexit:               ; preds = %loopentry
        ret int %tmp.2
}

