; RUN: llvm-as < %s | opt -adce | llvm-dis | not grep call

; The call is not live just because the PHI uses the call retval!

int %test(int %X) {
	br label %Done

DeadBlock:
	%Y = call int %test(int 0)
	br label %Done

Done:
	%Z = phi int [%X, %0], [%Y, %DeadBlock]
	ret int %Z
}
