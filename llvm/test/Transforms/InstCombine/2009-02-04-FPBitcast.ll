; RUN: llvm-as < %s | opt -instcombine
; PR3468

define x86_fp80 @cast() {
	%tmp = bitcast i80 0 to x86_fp80		; <x86_fp80> [#uses=1]
	ret x86_fp80 %tmp
}
