; RUN: llvm-as < %s | opt -basicaa -load-vn -gcse -instcombine | llvm-dis | not grep sub
declare int %strlen(sbyte*)

int %test(sbyte* %P) {
	%X = call int %strlen(sbyte* %P)
	%A = add int %X, 14
	%Y = call int %strlen(sbyte* %P)
	%Z = sub int %X, %Y
	%B = add int %A, %Z
	ret int %B
}
