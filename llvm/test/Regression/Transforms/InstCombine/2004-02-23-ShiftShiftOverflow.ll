; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep 34

int %test(int %X) {
	; Do not fold into shr X, 34, as this uses undefined behavior!
	%Y = shr int %X, ubyte 17
	%Z = shr int %Y, ubyte 17
	ret int %Z
}

int %test2(int %X) {
	; Do not fold into shl X, 34, as this uses undefined behavior!
	%Y = shl int %X, ubyte 17
	%Z = shl int %Y, ubyte 17
	ret int %Z
}
