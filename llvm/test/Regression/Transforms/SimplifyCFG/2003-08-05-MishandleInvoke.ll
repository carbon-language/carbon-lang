; Do not remove the invoke!
;
; RUN: as < %s | opt -simplifycfg | dis | grep invoke

int %test() {
	invoke int %test() to label %Ret except label %Ret
Ret:
	%A = add int 0, 1
	ret int %A
}
