; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | egrep 'shl|lshr|ashr' | wc -l | grep 3

int %test0(int %A, int %B, ubyte %C) {
	%X = shl int %A, ubyte %C
	%Y = shl int %B, ubyte %C
	%Z = and int %X, %Y
	ret int %Z
}

int %test1(int %A, int %B, ubyte %C) {
	%X = lshr int %A, ubyte %C
	%Y = lshr int %B, ubyte %C
	%Z = or int %X, %Y
	ret int %Z
}

int %test2(int %A, int %B, ubyte %C) {
	%X = ashr int %A, ubyte %C
	%Y = ashr int %B, ubyte %C
	%Z = xor int %X, %Y
	ret int %Z
}
