; Test that pure functions are cse'd away

; RUN: llvm-as < %s | opt -globalsmodref-aa -load-vn -gcse -instcombine | llvm-dis | not grep sub

int %pure(int %X) {
	%Y = add int %X, 1
	ret int %Y
}

int %test(int %X) {
	%A = call int %pure(int %X)
	%B = call int %pure(int %X)
	%C = sub int %A, %B
	ret int %C
}
