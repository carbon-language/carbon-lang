; This testcase should be able to eliminate at least one of the casts.
;
; RUN: llvm-as < %s | opt -raise | llvm-dis | not grep 'REMOVE'

int %foo(sbyte * %PF) {
	%UPF = cast sbyte* %PF to uint()*
	%Ret = call uint %UPF()
	%REMOVE = cast uint %Ret to int
	ret int %REMOVE
}
