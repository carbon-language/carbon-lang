; RUN: llvm-as < %s | llc -march=ppc32 | not grep 'extsh\|rlwinm r3, r3'

int %test1(short %X) {
	%Y = cast short %X to int  ;; dead
	ret int %Y
}

int %test2(ushort %X) {
	%Y = cast ushort %X to int
	%Z = and int %Y, 65535      ;; dead
	ret int %Z
}
