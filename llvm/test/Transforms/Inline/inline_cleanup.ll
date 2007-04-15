; Test that the inliner doesn't leave around dead allocas, and that it folds
; uncond branches away after it is done specializing.

; RUN: llvm-upgrade < %s | llvm-as | opt -inline | llvm-dis | \
; RUN:    not grep {alloca.*uses=0}
; RUN: llvm-upgrade < %s | llvm-as | opt -inline | llvm-dis | \
; RUN:    not grep {br label}

%A = weak global int 0		; <int*> [#uses=1]
%B = weak global int 0		; <int*> [#uses=1]
%C = weak global int 0		; <int*> [#uses=1]

implementation   ; Functions:

internal fastcc void %foo(int %X) {
entry:
	%ALL = alloca int, align 4		; <int*> [#uses=1]
	%tmp1 = and int %X, 1		; <int> [#uses=1]
	%tmp1 = seteq int %tmp1, 0		; <bool> [#uses=1]
	br bool %tmp1, label %cond_next, label %cond_true

cond_true:		; preds = %entry
	store int 1, int* %A
	br label %cond_next

cond_next:		; preds = %entry, %cond_true
	%tmp4 = and int %X, 2		; <int> [#uses=1]
	%tmp4 = seteq int %tmp4, 0		; <bool> [#uses=1]
	br bool %tmp4, label %cond_next7, label %cond_true5

cond_true5:		; preds = %cond_next
	store int 1, int* %B
	br label %cond_next7

cond_next7:		; preds = %cond_next, %cond_true5
	%tmp10 = and int %X, 4		; <int> [#uses=1]
	%tmp10 = seteq int %tmp10, 0		; <bool> [#uses=1]
	br bool %tmp10, label %cond_next13, label %cond_true11

cond_true11:		; preds = %cond_next7
	store int 1, int* %C
	br label %cond_next13

cond_next13:		; preds = %cond_next7, %cond_true11
	%tmp16 = and int %X, 8		; <int> [#uses=1]
	%tmp16 = seteq int %tmp16, 0		; <bool> [#uses=1]
	br bool %tmp16, label %UnifiedReturnBlock, label %cond_true17

cond_true17:		; preds = %cond_next13
	call void %ext( int* %ALL )
	ret void

UnifiedReturnBlock:		; preds = %cond_next13
	ret void
}

declare void %ext(int*)

void %test() {
entry:
	tail call fastcc void %foo( int 1 )
	tail call fastcc void %foo( int 2 )
	tail call fastcc void %foo( int 3 )
	tail call fastcc void %foo( int 8 )
	ret void
}
