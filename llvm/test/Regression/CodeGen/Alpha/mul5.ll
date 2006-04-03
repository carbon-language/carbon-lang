; Make sure this testcase does not use mulq
; RUN: llvm-as < %s | llc -march=alpha | grep -i 'mul' |wc -l |grep 0

implementation   ; Functions:

ulong %foo1(ulong %x) {
entry:
	%tmp.1 = mul ulong %x, 9		; <ulong> [#uses=1]
	ret ulong %tmp.1
}
ulong %foo3(ulong %x) {
entry:
        %tmp.1 = mul ulong %x, 259
	ret ulong %tmp.1
}

ulong %foo4l(ulong %x) {
entry:
        %tmp.1 = mul ulong %x, 260
	ret ulong %tmp.1
}

ulong %foo4ln(ulong %x) {
entry:
        %tmp.1 = mul ulong %x, 252
	ret ulong %tmp.1
}

ulong %foo1n(ulong %x) {
entry:
        %tmp.1 = mul ulong %x, 511
	ret ulong %tmp.1
}

ulong %foo8l(ulong %x) {
entry:
        %tmp.1 = mul ulong %x, 768
        ret ulong %tmp.1
}

long %bar(long %x) {
entry:
	%tmp.1 = mul long %x, 5		; <long> [#uses=1]
	ret long %tmp.1
}
