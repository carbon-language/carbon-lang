; RUN: llvm-as < %s | opt -basicaa -load-vn -gcse -instcombine | llvm-dis | not grep sub

sbyte %test(sbyte** %P) {
	%A = load sbyte** %P
	%B = load sbyte* %A

	%X = load sbyte** %P
	%Y = load sbyte* %X

	%R = sub sbyte %B, %Y
	ret sbyte %R
}
