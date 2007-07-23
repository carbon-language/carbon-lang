; RUN: llvm-upgrade < %s | llvm-as | opt -fdse | llvm-dis | \
; RUN:    grep {store i32 1234567}

; Do not delete stores that are only partially killed.

int %test() {
	%V = alloca int
	store int 1234567, int* %V
	%V2 = cast int* %V to sbyte*
	store sbyte 0, sbyte* %V2
	%X = load int* %V
	ret int %X
}
