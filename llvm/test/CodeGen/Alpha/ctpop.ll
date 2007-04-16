; Make sure this testcase codegens to the ctpop instruction
; RUN: llvm-upgrade < %s | llvm-as | llc -march=alpha -mcpu=ev67 | grep -i ctpop
; RUN: llvm-upgrade < %s | llvm-as | llc -march=alpha -mattr=+CIX | \
; RUN:   grep -i ctpop
; RUN: llvm-upgrade < %s | llvm-as | llc -march=alpha -mcpu=ev6 | \
; RUN:   not grep -i ctpop
; RUN: llvm-upgrade < %s | llvm-as | llc -march=alpha -mcpu=ev56 | \
; RUN:   not grep -i ctpop
; RUN: llvm-upgrade < %s | llvm-as | llc -march=alpha -mattr=-CIX | \
; RUN:   not grep -i ctpop

declare long %llvm.ctpop.i64(long)

implementation   ; Functions:

long %bar(long %x) {
entry:
	%tmp.1 = call long %llvm.ctpop.i64( long %x ) 
	ret long %tmp.1
}
