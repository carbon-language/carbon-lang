; Make sure this testcase codegens to the ctlz instruction
; RUN: llvm-as < %s | llc -march=alpha -mcpu=ev67 | grep -i 'ctlz'
; RUN: llvm-as < %s | llc -march=alpha -mattr=+CIX | grep -i 'ctlz'
; RUN: llvm-as < %s | llc -march=alpha -mcpu=ev6 | not grep -i 'ctlz'
; RUN: llvm-as < %s | llc -march=alpha -mcpu=ev56 | not grep -i 'ctlz'
; RUN: llvm-as < %s | llc -march=alpha -mattr=-CIX | not grep -i 'ctlz'

declare ubyte %llvm.ctlz(ubyte)

implementation   ; Functions:

ubyte %bar(ubyte %x) {
entry:
	%tmp.1 = call ubyte %llvm.ctlz( ubyte %x ) 
	ret ubyte %tmp.1
}
