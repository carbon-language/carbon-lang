; This testcase checks to make sure we can sink values which are only live on
; some exits out of the loop, and that we can do so without breaking dominator
; info.
;
; RUN: llvm-as < %s | opt -licm | llvm-dis | grep -C1 add | grep exit2:

implementation   ; Functions:

int %test(bool %C1, bool %C2, int *%P, int* %Q) {
Entry:
	br label %Loop

Loop:
	br bool %C1, label %Cont, label %exit1
Cont:
	%X = load int* %P
	store int %X, int* %Q
 	%V = add int %X, 1
	br bool %C2, label %Loop, label %exit2

exit1:
	ret int 0
exit2:
	ret int %V
}
