; Make sure this testcase codegens to the ctpop instruction
; RUN: llvm-as < %s | llc -march=alpha -mcpu=ev67 | grep -i 'ctpop'
; RUN: llvm-as < %s | llc -march=alpha -mattr=+CIX | grep -i 'ctpop'
; RUN: llvm-as < %s | llc -march=alpha -mcpu=ev6 | not grep -i 'ctpop'
; RUN: llvm-as < %s | llc -march=alpha -mcpu=ev56 | not grep -i 'ctpop'
; RUN: llvm-as < %s | llc -march=alpha -mattr=-CIX | not grep -i 'ctpop'


declare long %llvm.ctpop(long)

implementation   ; Functions:

long %bar(long %x) {
entry:
	%tmp.1 = call long %llvm.ctpop( long %x ) 
	ret long %tmp.1
}
