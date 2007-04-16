; All of these should be codegen'd without loading immediates
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -o %t -f
; RUN: grep addc %t | wc -l | grep 1
; RUN: grep adde %t | wc -l | grep 1
; RUN: grep addze %t | wc -l | grep 1
; RUN: grep addme %t | wc -l | grep 1
; RUN: grep addic %t | wc -l | grep 2

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
