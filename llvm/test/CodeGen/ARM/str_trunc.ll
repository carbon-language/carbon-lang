; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | \
; RUN:   grep strb | count 1
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | \
; RUN:   grep strh | count 1

void %test1(int %v, short* %ptr) {
        %tmp = cast int %v to short
	store short %tmp, short* %ptr
	ret void
}

void %test2(int %v, ubyte* %ptr) {
        %tmp = cast int %v to ubyte
	store ubyte %tmp, ubyte* %ptr
	ret void
}
