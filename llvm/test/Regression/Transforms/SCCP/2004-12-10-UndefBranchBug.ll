; RUN: llvm-as < %s | opt -sccp | llvm-dis | grep 'ret int 1'

; This function definitely returns 1, even if we don't know the direction
; of the branch.

int %foo() {
	br bool undef, label %T, label %T
T:
	%X = add int 0, 1
	ret int %X
}
