; RUN: llvm-as < %s | llc

int %test(int %tmp93) {
	%tmp98 = shl int %tmp93, ubyte 31		; <int> [#uses=1]
	%tmp99 = shr int %tmp98, ubyte 31		; <int> [#uses=1]
	%tmp99 = cast int %tmp99 to sbyte		; <sbyte> [#uses=1]
	%tmp99100 = cast sbyte %tmp99 to int		; <int> [#uses=1]
	ret int %tmp99100
}

