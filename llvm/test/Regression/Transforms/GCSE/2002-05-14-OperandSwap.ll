; This entire chain of computation should be optimized away, but
; wasn't because the two multiplies were not detected as being identical.
;
; RUN: llvm-as < %s  | opt -gcse -instcombine -dce | llvm-dis | not grep sub

implementation   ; Functions:

uint "vnum_test4"(uint* %data) {
	%idx1 = getelementptr uint* %data, uint 1
	%idx2 = getelementptr uint* %data, uint 3
	%reg1101 = load uint* %idx1
	%reg1111 = load uint* %idx2
	%reg109 = mul uint %reg1101, %reg1111
	%reg108 = mul uint %reg1111, %reg1101
	%reg121 = sub uint %reg108, %reg109
	ret uint %reg121
}
