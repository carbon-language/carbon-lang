; RUN: llvm-as < %s | opt -reassociate -instcombine | llvm-dis | grep 'ret int 0'

int %f(int %a0, int %a1, int %a2, int %a3, int %a4) {
	%tmp.2 = add int %a4, %a3		; <int> [#uses=1]
	%tmp.4 = add int %tmp.2, %a2		; <int> [#uses=1]
	%tmp.6 = add int %tmp.4, %a1		; <int> [#uses=1]
	%tmp.8 = add int %tmp.6, %a0		; <int> [#uses=1]
	%tmp.11 = add int %a3, %a2		; <int> [#uses=1]
	%tmp.13 = add int %tmp.11, %a1		; <int> [#uses=1]
	%tmp.15 = add int %tmp.13, %a0		; <int> [#uses=1]
	%tmp.18 = add int %a2, %a1		; <int> [#uses=1]
	%tmp.20 = add int %tmp.18, %a0		; <int> [#uses=1]
	%tmp.23 = add int %a1, %a0		; <int> [#uses=1]
	%tmp.26 = sub int %tmp.8, %tmp.15		; <int> [#uses=1]
	%tmp.28 = add int %tmp.26, %tmp.20		; <int> [#uses=1]
	%tmp.30 = sub int %tmp.28, %tmp.23		; <int> [#uses=1]
	%tmp.32 = sub int %tmp.30, %a4		; <int> [#uses=1]
	%tmp.34 = sub int %tmp.32, %a2		; <int> [#uses=2]
	%T = mul int %tmp.34, %tmp.34
	ret int %T
}
