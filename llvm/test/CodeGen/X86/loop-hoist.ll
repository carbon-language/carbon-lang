; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -relocation-model=dynamic-no-pic -mtriple=i686-apple-darwin8.7.2 |\
; RUN:   grep L_Arr.non_lazy_ptr
; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -relocation-model=dynamic-no-pic -mtriple=i686-apple-darwin8.7.2 |\
; RUN:   %prcontext L_Arr.non_lazy_ptr 1 | grep {4(%esp)}

%Arr = external global [0 x int]                ; <[0 x int]*> [#uses=2]

implementation   ; Functions:

void %foo(int %N.in) {
entry:
        %N = cast int %N.in to uint                ; <uint> [#uses=1]
        br label %cond_true

cond_true:              ; preds = %cond_true, %entry
        %indvar = phi uint [ 0, %entry ], [ %indvar.next, %cond_true ]          ; <uint> [#uses=3]
        %i.0.0 = cast uint %indvar to int               ; <int> [#uses=1]
        %tmp = getelementptr [0 x int]* %Arr, int 0, int %i.0.0
        store int %i.0.0, int* %tmp
        %indvar.next = add uint %indvar, 1              ; <uint> [#uses=2]
        %exitcond = seteq uint %indvar.next, %N         ; <bool> [#uses=1]
        br bool %exitcond, label %return, label %cond_true

return:         ; preds = %cond_true, %entry
        ret void
}

