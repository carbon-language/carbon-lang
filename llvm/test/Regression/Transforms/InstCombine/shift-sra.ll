; RUN: llvm-as < %s | opt -instcombine -disable-output &&
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep 'shr int'

int %test0(int %X, ubyte %A) {
	%Y = shr int %X, ubyte %A  ; can be logical shift.
	%Z = and int %Y, 1
	ret int %Z
}
