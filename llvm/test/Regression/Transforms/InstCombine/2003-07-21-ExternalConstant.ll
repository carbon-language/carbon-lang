;
; Test: ExternalConstant
;
; Description:
;	This regression test helps check whether the instruction combining
;	optimization pass correctly handles global variables which are marked
;	as external and constant.
;
;	If a problem occurs, we should die on an assert().  Otherwise, we
;	should pass through the optimizer without failure.
;
; Extra code:
; RUN: llvm-as < %s | opt -instcombine
;

target endian = little
target pointersize = 32
%silly = external constant int		; <int*> [#uses=1]

implementation   ; Functions:

declare void %bzero(sbyte*, uint)

declare void %bcopy(sbyte*, sbyte*, uint)

declare int %bcmp(sbyte*, sbyte*, uint)

declare int %fputs(sbyte*, sbyte*)

declare int %fputs_unlocked(sbyte*, sbyte*)

int %function(int %a.1) {
entry:		; No predecessors!
	%a.0 = alloca int		; <int*> [#uses=2]
	%result = alloca int		; <int*> [#uses=2]
	store int %a.1, int* %a.0
	%tmp.0 = load int* %a.0		; <int> [#uses=1]
	%tmp.1 = load int* %silly		; <int> [#uses=1]
	%tmp.2 = add int %tmp.0, %tmp.1		; <int> [#uses=1]
	store int %tmp.2, int* %result
	br label %return

return:		; preds = %entry
	%tmp.3 = load int* %result		; <int> [#uses=1]
	ret int %tmp.3
}
