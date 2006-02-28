
; RUN: llvm-as < %s | llc -march=ppc32 &&
; RUN: llvm-as < %s | llc -march=ppc32 | not grep 'extsh\|rlwinm'

declare short %foo()

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

uint %test4(ushort* %P) {
        %tmp.1 = load ushort* %P
        %tmp.2 = cast ushort %tmp.1 to uint
        %tmp.3 = and uint %tmp.2, 255
        ret uint %tmp.3
}

uint %test5(short* %P) {
        %tmp.1 = load short* %P
        %tmp.2 = cast short %tmp.1 to ushort
        %tmp.3 = cast ushort %tmp.2 to uint
        %tmp.4 = and uint %tmp.3, 255
        ret uint %tmp.4
}

uint %test6(uint* %P) {
        %tmp.1 = load uint* %P
        %tmp.2 = and uint %tmp.1, 255
        ret uint %tmp.2
}

ushort %test7(float %a) {
        %tmp.1 = cast float %a to ushort
        ret ushort %tmp.1
}

