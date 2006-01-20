; RUN: llvm-as < %s | llc -march=x86 | grep 'shld' | wc -l | grep 1
;
; Check that the isel does not fold the shld, which already folds a load
; and has two uses, into a store.
%A = external global uint

uint %test5(uint %B, ubyte %C) {
	%tmp.1 = load uint *%A;
	%tmp.2 = shl uint %tmp.1, ubyte %C
	%tmp.3 = sub ubyte 32, %C
	%tmp.4 = shr uint %B, ubyte %tmp.3
	%tmp.5 = or uint %tmp.4, %tmp.2
	store uint %tmp.5, uint* %A
	ret uint %tmp.5
}
