; RUN: llvm-as < %s | opt -loop-unroll -disable-output

int %main() {
entry:
        br label %no_exit

no_exit:                ; preds = %entry, %no_exit
        %indvar = phi uint [ 0, %entry ], [ %indvar.next, %no_exit ]            ; <uint> [#uses=1]
        %indvar.next = add uint %indvar, 1              ; <uint> [#uses=2]
        %exitcond = setne uint %indvar.next, 2147483648         ; <bool> [#uses=1]
        br bool %exitcond, label %no_exit, label %loopexit

loopexit:               ; preds = %no_exit
        ret int 0
}
