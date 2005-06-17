; RUN: llvm-as < %s | opt -instcombine -disable-output
; PR585
bool %test() {
	%tmp.26 = div int 0, -2147483648		; <int> [#uses=1]
	%tmp.27 = seteq int %tmp.26, 0		; <bool> [#uses=1]
	ret bool %tmp.27
}
