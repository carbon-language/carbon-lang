; RUN: llvm-upgrade < %s | llvm-as | opt -sccp | llvm-dis | grep {ret i32 0}

; Test that SCCP has basic knowledge of when and/or nuke overdefined values.

int %test(int %X) {
	%Y = and int %X, 0
	ret int %Y
}
