; llvm-as < %s | opt -instcombine | llvm-dis | grep 'to sbyte'

%.Base64_1 = external constant [4 x sbyte] 

ubyte %test(sbyte %X) {
	%tmp.12 = sub sbyte %X, cast ([4 x sbyte]* %.Base64_1 to sbyte)		; <sbyte> [#uses=1]
	%tmp.13 = cast sbyte %tmp.12 to ubyte		; <ubyte> [#uses=1]
	ret ubyte %tmp.13
}
