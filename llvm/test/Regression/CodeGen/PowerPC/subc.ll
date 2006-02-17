; All of these ands and shifts should be folded into rlwimi's
; RUN: llvm-as < %s | llc -march=ppc32 | grep subfc | wc -l | grep 1 &&
; RUN: llvm-as < %s | llc -march=ppc32 | grep subfe | wc -l | grep 1 &&
; RUN: llvm-as < %s | llc -march=ppc32 | grep subfze | wc -l | grep 1 &&
; RUN: llvm-as < %s | llc -march=ppc32 | grep subfme | wc -l | grep 1 &&
; RUN: llvm-as < %s | llc -march=ppc32 | grep subfic | wc -l | grep 2
implementation   ; Functions:

long %sub_ll(long %a, long %b) {
entry:
	%tmp.2 = sub long %a, %b		; <long> [#uses=1]
	ret long %tmp.2
}

long %sub_l_5(long %a) {
entry:
	%tmp.1 = sub long 5, %a		; <long> [#uses=1]
	ret long %tmp.1
}

long %sub_l_m5(long %a) {
entry:
	%tmp.1 = sub long -5, %a		; <long> [#uses=1]
	ret long %tmp.1
}
