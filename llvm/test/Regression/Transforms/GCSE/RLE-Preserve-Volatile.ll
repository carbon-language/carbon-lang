
; RUN: llvm-as < %s | opt -load-vn -gcse -instcombine | llvm-dis | grep sub

int %test(int* %P) {
	%X = volatile load int* %P
	%Y = volatile load int* %P
	%Z = sub int %X, %Y
	ret int %Z
}
