; RUN: llvm-as < %s | opt -inline -disable-output &&
; RUN: llvm-as < %s | opt -inline | llvm-dis | not grep callee &&
; RUN: llvm-as < %s | opt -inline | llvm-dis | not grep div

implementation

internal int %callee(int %A, int %B) {
	%C = div int %A, %B
	ret int %C
}

int %test() {
	%X = call int %callee(int 10, int 3)
	ret int %X
}
