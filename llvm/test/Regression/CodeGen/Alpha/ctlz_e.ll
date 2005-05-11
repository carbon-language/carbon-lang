; Make sure this testcase does not use ctpop
; RUN: llvm-as < %s | llc -march=alpha | grep -i 'ctpop' |wc -l |grep 0

declare ulong %llvm.ctlz(ulong)

implementation   ; Functions:

ulong %bar(ulong %x) {
entry:
	%tmp.1 = call ulong %llvm.ctlz( ulong %x ) 
	ret ulong %tmp.1
}
