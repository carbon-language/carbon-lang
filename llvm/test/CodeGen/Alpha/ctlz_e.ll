; Make sure this testcase does not use ctpop
; RUN: llvm-upgrade < %s | llvm-as | llc -march=alpha | not grep -i ctpop 

declare ulong %llvm.ctlz.i64(ulong)

implementation   ; Functions:

ulong %bar(ulong %x) {
entry:
	%tmp.1 = call ulong %llvm.ctlz.i64( ulong %x ) 
	ret ulong %tmp.1
}
