; RUN: llvm-as < %s | opt -predsimplify -verify

void %dgefa() {
entry:
        br label %cond_true96

cond_true:              ; preds = %cond_true96
        %tmp19 = seteq int %tmp10, %k.0         ; <bool> [#uses=1]
        br bool %tmp19, label %cond_next, label %cond_true20

cond_true20:            ; preds = %cond_true
        br label %cond_next

cond_next:              ; preds = %cond_true20, %cond_true
        %tmp84 = setgt int %tmp3, 1999          ; <bool> [#uses=0]
        ret void

cond_true96:            ; preds = %cond_true96, %entry
        %k.0 = phi int [ 0, %entry ], [ 0, %cond_true96 ]               ; <int> [#uses=3]
        %tmp3 = add int %k.0, 1         ; <int> [#uses=1]
        %tmp10 = add int 0, %k.0                ; <int> [#uses=1]
        br bool false, label %cond_true96, label %cond_true
}
