; All of these ands and shifts should be folded into rlwimi's
; RUN: llvm-as < %s | llc -march=ppc32 | grep addc | wc -l | grep 1 &&
; RUN: llvm-as < %s | llc -march=ppc32 | grep adde | wc -l | grep 1 &&
; RUN: llvm-as < %s | llc -march=ppc32 | grep addze | wc -l | grep 1 &&
; RUN: llvm-as < %s | llc -march=ppc32 | grep addme | wc -l | grep 1 &&
; RUN: llvm-as < %s | llc -march=ppc32 | grep addic | wc -l | grep 2

implementation   ; Functions:

long %add_ll(long %a, long %b) {
entry:
	%tmp.2 = add long %b, %a		; <long> [#uses=1]
	ret long %tmp.2
}

long %add_l_5(long %a) {
entry:
	%tmp.1 = add long %a, 5		; <long> [#uses=1]
	ret long %tmp.1
}

long %add_l_m5(long %a) {
entry:
	%tmp.1 = add long %a, -5		; <long> [#uses=1]
	ret long %tmp.1
}
