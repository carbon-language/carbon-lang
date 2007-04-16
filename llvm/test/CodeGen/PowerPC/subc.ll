; All of these should be codegen'd without loading immediates
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -o %t -f
; RUN: grep subfc %t | wc -l | grep 1
; RUN: grep subfe %t | wc -l | grep 1
; RUN: grep subfze %t | wc -l | grep 1
; RUN: grep subfme %t | wc -l | grep 1
; RUN: grep subfic %t | wc -l | grep 2
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
