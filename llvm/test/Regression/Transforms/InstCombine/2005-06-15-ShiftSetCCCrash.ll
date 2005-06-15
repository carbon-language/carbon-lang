; RUN: llvm-as < %s | opt -instcombine -disable-output
; PR577

bool %test() {
	%tmp.3 = shl int 0, ubyte 41		; <int> [#uses=1]
	%tmp.4 = setne int %tmp.3, 0		; <bool> [#uses=1]
	ret bool %tmp.4
}
