; RUN: llvm-as < %s | opt -reassociate -disable-output

void %test(int %a, int %b, int %c, int %d) {
	%tmp.2 = xor int %a, %b		; <int> [#uses=1]
	%tmp.5 = xor int %c, %d		; <int> [#uses=1]
	%tmp.6 = xor int %tmp.2, %tmp.5		; <int> [#uses=1]
	%tmp.9 = xor int %c, %a		; <int> [#uses=1]
	%tmp.12 = xor int %b, %d		; <int> [#uses=1]
	%tmp.13 = xor int %tmp.9, %tmp.12		; <int> [#uses=1]
	%tmp.16 = xor int %tmp.6, %tmp.13		; <int> [#uses=0]
	ret void
}
