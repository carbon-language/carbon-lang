; RUN: llvm-as < %s | opt -sccp | llvm-dis | grep 'ret int 0'

; Test that SCCP has basic knowledge of when and/or nuke overdefined values.

int %test(int %X) {
	%Y = and int %X, 0
	ret int %Y
}
