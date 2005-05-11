; Make sure this testcase does not use ctpop
; RUN: llvm-as < %s | llc -march=ppc32 | grep -i 'cntlzw'

declare int %llvm.cttz(int)

implementation   ; Functions:

int %bar(int %x) {
entry:
	%tmp.1 = call int %llvm.cttz( int %x ) 
	ret int %tmp.1
}
