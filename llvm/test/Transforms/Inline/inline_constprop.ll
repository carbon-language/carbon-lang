; RUN: llvm-upgrade < %s | llvm-as | opt -inline | llvm-dis | not grep callee
; RUN: llvm-upgrade < %s | llvm-as | opt -inline | llvm-dis | not grep div

implementation

internal int %callee(int %A, int %B) {
	%C = div int %A, %B
	ret int %C
}

int %test() {
	%X = call int %callee(int 10, int 3)
	ret int %X
}
