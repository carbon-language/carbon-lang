; RUN: llvm-as < %s | opt -indvars -instcombine | llvm-dis | grep 'store int 0'
; Test that -indvars can reduce variable stride IVs.  If it can reduce variable
; stride iv's, it will make %iv. and %m.0.0 isomorphic to each other without 
; cycles, allowing the tmp.21 subtraction to be eliminated.

void %vnum_test8(int* %data) {
entry:
        %tmp.1 = getelementptr int* %data, int 3                ; <int*> [#uses=1]
        %tmp.2 = load int* %tmp.1               ; <int> [#uses=2]
        %tmp.4 = getelementptr int* %data, int 4                ; <int*> [#uses=1]
        %tmp.5 = load int* %tmp.4               ; <int> [#uses=2]
        %tmp.8 = getelementptr int* %data, int 2                ; <int*> [#uses=1]
        %tmp.9 = load int* %tmp.8               ; <int> [#uses=3]
        %tmp.125 = setgt int %tmp.2, 0          ; <bool> [#uses=1]
        br bool %tmp.125, label %no_exit.preheader, label %return

no_exit.preheader:              ; preds = %entry
        %tmp.16 = getelementptr int* %data, int %tmp.9          ; <int*> [#uses=1]
        br label %no_exit

no_exit:                ; preds = %no_exit, %no_exit.preheader
        %iv. = phi uint [ 0, %no_exit.preheader ], [ %iv..inc, %no_exit ]               ; <uint> [#uses=1]
        %iv. = phi int [ %tmp.5, %no_exit.preheader ], [ %iv..inc, %no_exit ]           ; <int> [#uses=2]
        %m.0.0 = phi int [ %tmp.5, %no_exit.preheader ], [ %tmp.24, %no_exit ]          ; <int> [#uses=2]
        store int 2, int* %tmp.16
        %tmp.21 = sub int %m.0.0, %iv.          ; <int> [#uses=1]
        store int %tmp.21, int* %data
        %tmp.24 = add int %m.0.0, %tmp.9                ; <int> [#uses=1]
        %iv..inc = add int %tmp.9, %iv.         ; <int> [#uses=1]
        %iv..inc = add uint %iv., 1             ; <uint> [#uses=2]
        %iv..inc1 = cast uint %iv..inc to int           ; <int> [#uses=1]
        %tmp.12 = setlt int %iv..inc1, %tmp.2           ; <bool> [#uses=1]
        br bool %tmp.12, label %no_exit, label %return.loopexit

return.loopexit:                ; preds = %no_exit
        br label %return

return:         ; preds = %return.loopexit, %entry
        ret void
}

