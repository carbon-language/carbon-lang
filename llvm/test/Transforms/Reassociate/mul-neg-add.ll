; RUN: llvm-upgrade < %s | llvm-as | opt -reassociate -instcombine | llvm-dis |\
; RUN:   not grep {sub i32 0}

int %test(int %X, int %Y, int %Z) {
	%A = sub int 0, %X
	%B = mul int %A, %Y
	%C = add int %B, %Z   ; (-X)*Y + Z -> Z-X*Y
	ret int %C
}
