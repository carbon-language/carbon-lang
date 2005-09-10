; RUN: llvm-as < %s | opt -loop-reduce | llvm-dis | not grep 'cast uint 1 to uint'
; The setlt wants to use a value that is incremented one more than the dominant
; IV.  Don't insert the 1 outside the loop, preventing folding it into the add.

implementation   ; Functions:

void %test([700 x int]* %nbeaux_.0__558, int* %i_.16574) {
then.0:
        br label %no_exit.2

no_exit.2:              ; preds = %no_exit.2, %then.0
        %indvar630 = phi uint [ 0, %then.0 ], [ %indvar.next631, %no_exit.2 ]           ; <uint> [#uses=3]
        %indvar630 = cast uint %indvar630 to int                ; <int> [#uses=1]
        %tmp.38 = getelementptr [700 x int]* %nbeaux_.0__558, int 0, uint %indvar630            ; <int*> [#uses=1]
        store int 0, int* %tmp.38
        %inc.2 = add int %indvar630, 2          ; <int> [#uses=2]
        %tmp.34 = setlt int %inc.2, 701         ; <bool> [#uses=1]
        %indvar.next631 = add uint %indvar630, 1                ; <uint> [#uses=1]
        br bool %tmp.34, label %no_exit.2, label %loopexit.2.loopexit

loopexit.2.loopexit:            ; preds = %no_exit.2
        store int %inc.2, int* %i_.16574
        ret void
}
