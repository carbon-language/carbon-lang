; RUN: llvm-as < %s | llc -march=x86 | grep lea

declare int %foo()
int %test() {
	%tmp.0 = tail call int %foo( )		; <int> [#uses=1]
	%tmp.1 = mul int %tmp.0, 9		; <int> [#uses=1]
	ret int %tmp.1
}
