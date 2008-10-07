;test for ADDC and ADDE expansion
;
; RUN: llvm-as < %s | llc -march=alpha -o %t.s -f
; XFAIL: *

define i128 @add128(i128 %x, i128 %y) {
entry:
	%tmp = add i128 %y, %x
	ret i128 %tmp
}
