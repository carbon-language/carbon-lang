; Make sure this testcase codegens to the ornot instruction
; RUN: llvm-as < %s | llc -march=alpha | grep 'ornot'

implementation   ; Functions:

long %bar(long %x, long %y) {
entry:
	%tmp.1 = xor long %x, -1  		; <long> [#uses=1]
        %tmp.2 = or long %y, long %tmp.1
	ret long %tmp.2
}
