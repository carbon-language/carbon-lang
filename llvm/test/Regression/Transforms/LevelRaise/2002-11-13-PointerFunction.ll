; This testcase should be able to eliminate at least one of the casts.
;
; RUN: if as < %s | opt -raise | dis | grep 'REMOVE'
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

int %foo(sbyte * %PF) {
	%UPF = cast sbyte* %PF to uint()*
	%Ret = call uint %UPF()
	%REMOVE = cast uint %Ret to int
	ret int %REMOVE
}
