; This entire chain of computation should be optimized away, but
; wasn't because the two multiplies were not detected as being identical.
;
; RUN: if as < %s  | opt -gcse -instcombine -dce | dis | grep sub
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

implementation   ; Functions:

uint "vnum_test4"(uint* %data) {
	%reg1101 = load uint* %data, uint 1
	%reg1111 = load uint* %data, uint 3
	%reg109 = mul uint %reg1101, %reg1111
	%reg108 = mul uint %reg1111, %reg1101
	%reg121 = sub uint %reg108, %reg109
	ret uint %reg121
}
