
; RUN: llvm-as < %s | opt -load-vn -gcse -instcombine | dis | grep sub

int %test(int* %P) {
	%X = load volatile int* %P
	%Y = load volatile int* %P
	%Z = sub int %X, %Y
	ret int %Z
}
