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

void %test3() {
	%tmp.0 = call short %foo()            ;; no extsh!
	%tmp.1 = setlt short %tmp.0, 1234
	br bool %tmp.1, label %then, label %UnifiedReturnBlock

then:	
	call int %test1(short 0)
	ret void
UnifiedReturnBlock:
	ret void
}

declare short %foo()
