; Make sure this testcase codegens to the ctpop instruction
; XFAIL: *
; RUN: llvm-as < %s | llc -march=alpha -enable-alpha-CT | grep 'ctpop'

declare long %llvm.ctpop(long)

implementation   ; Functions:

long %bar(long %x) {
entry:
	%tmp.1 = call long %llvm.ctpop( long %x ) 
	ret long %tmp.1
}
