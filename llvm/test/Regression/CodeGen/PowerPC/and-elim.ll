; RUN: llvm-as < %s | llc -march=ppc32 &&
; RUN: llvm-as < %s | llc -march=ppc32 | not grep rlwin

void %test(ubyte* %P) {
	%W = load ubyte* %P
	%X = shl ubyte %W, ubyte 1
	%Y = add ubyte %X, 2
	%Z = and ubyte %Y, 254        ; dead and
	store ubyte %Z, ubyte* %P
	ret void
}

ushort %test2(ushort %crc) { ; No and's should be needed for the ushorts here.
        %tmp.1 = shr ushort %crc, ubyte 1
        %tmp.7 = xor ushort %tmp.1, 40961
        ret ushort %tmp.7
}

