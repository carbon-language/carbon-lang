; Do not remove the invoke!
;
; RUN: as < %s | opt -simplifycfg -disable-output

int %test() {
	%A = invoke int %test() to label %Ret except label %Ret
Ret:
	ret int %A
}
