; RUN: llvm-as < %s | opt -reassociate -instcombine -constprop -dce | llvm-dis | not grep add

int %test(int %A) {
	%X = add int %A, 1
	%Y = add int %A, 1
	%r = sub int %X, %Y
	ret int %r               ; Should be equal to 0!
}
