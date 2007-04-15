; Make sure this testcase does not use ctpop
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 | grep -i cntlzw

declare uint %llvm.cttz.i32(uint)

implementation   ; Functions:

uint %bar(uint %x) {
entry:
	%tmp.1 = call uint %llvm.cttz.i32( uint %x ) 
	ret uint %tmp.1
}
