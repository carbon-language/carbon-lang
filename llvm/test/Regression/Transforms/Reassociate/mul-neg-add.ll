; RUN: llvm-as < %s | opt -reassociate -instcombine | llvm-dis | not grep 'sub int 0'

int %test(int %X, int %Y, int %Z) {
	%A = sub int 0, %X
	%B = mul int %A, %Y
	%C = add int %B, %Z   ; (-X)*Y + Z -> Z-X*Y
	ret int %C
}
