;
; RUN: llvm-as < %s | opt -analyze -tddatastructure

%G = global int 2		; <int*> [#uses=1]
%H = global int* null

%I = global int** null
%J = global int** null

implementation   ; Functions:

void %foo1() {
	store int* %G, int** %H
        store int** %H, int ***%I
	ret void
}

void %foo2() {		; No predecessors!
	store int 7, int* %G
	store int** %H, int ***%J
	ret void
}

void %test(bool %cond) {
; <label>:0		; No predecessors!
	br bool %cond, label %call, label %F

F:		; preds = %0
	br label %call

call:		; preds = %F, %0
	%Fn = phi void ()* [ %foo2, %F ], [ %foo1, %0 ]		; <void ()*> [#uses=1]
	call void %Fn()
	ret void
}
