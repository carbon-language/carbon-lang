; RUN: llvm-as < %s | llc
float %t(long %u) {
	%u = cast long %u to ulong		; <ulong> [#uses=1]
	%tmp5 = add ulong %u, 9007199254740991		; <ulong> [#uses=1]
	%tmp = setgt ulong %tmp5, 18014398509481982		; <bool> [#uses=1]
	br bool %tmp, label %T, label %F
T:
	ret float 1.0
F:
	call float %t(long 0)
	ret float 0.0
}
