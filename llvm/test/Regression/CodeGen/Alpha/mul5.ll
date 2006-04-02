; Make sure this testcase does not use mulq
; RUN: llvm-as < %s | llc -march=alpha | grep -i 'mul' |wc -l |grep 0

implementation   ; Functions:

ulong %foo(ulong %x) {
entry:
	%tmp.1 = mul ulong %x, 5		; <ulong> [#uses=1]
	ret ulong %tmp.1
}

long %bar(long %x) {
entry:
	%tmp.1 = mul long %x, 5		; <long> [#uses=1]
	ret long %tmp.1
}
