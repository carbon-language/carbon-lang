; RUN: llvm-as < %s | opt -basicaa -load-vn -gcse -instcombine | llvm-dis | not grep sub

%S = type { int, sbyte }

sbyte %test(sbyte** %P) {
	%A = load sbyte** %P
	%B = load sbyte* %A

	%X = load sbyte** %P
	%Y = load sbyte* %X

	%R = sub sbyte %B, %Y
	ret sbyte %R
}

sbyte %test(%S ** %P) {
	%A = load %S** %P
	%B = getelementptr %S* %A, int 0, ubyte 1
	%C = load sbyte* %B

	%X = load %S** %P
	%Y = getelementptr %S* %X, int 0, ubyte 1
	%Z = load sbyte* %Y

	%R = sub sbyte %C, %Z
	ret sbyte %R
}
