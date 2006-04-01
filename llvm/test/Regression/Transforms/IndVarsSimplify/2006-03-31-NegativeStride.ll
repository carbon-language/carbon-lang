; RUN: llvm-as < %s | opt -indvars | llvm-dis | grep 'ret int 27'

; Make sure to compute the right exit value based on negative strides.
; PR726

int %test() {
entry:
        br label %cond_true

cond_true:              ; preds = %cond_true, %entry
        %a.0.0 = phi int [ 10, %entry ], [ %tmp4, %cond_true ]          ; <int> [#uses=2]
        %b.0.0 = phi int [ 0, %entry ], [ %tmp2, %cond_true ]           ; <int> [#uses=1]
        %tmp2 = add int %b.0.0, %a.0.0          ; <int> [#uses=2]
        %tmp4 = add int %a.0.0, -1              ; <int> [#uses=2]
        %tmp = setgt int %tmp4, 7               ; <bool> [#uses=1]
        br bool %tmp, label %cond_true, label %bb7

bb7:            ; preds = %cond_true
        ret int %tmp2
}

