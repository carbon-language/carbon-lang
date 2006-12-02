; Do not remove the invoke!
;
; RUN: llvm-upgrade < %s | llvm-as | opt -simplifycfg -disable-output

int %test() {
	%A = invoke int %test() to label %Ret except label %Ret
Ret:
	ret int %A
}
