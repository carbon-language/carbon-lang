; With sub reassociation, constant folding can eliminate all of the constants.
;
; RUN: llvm-as < %s | opt -reassociate -constprop -instcombine -dce | llvm-dis | not grep add

int %test(int %A, int %B) {
	%W = add int 5, %B
	%X = add int -7, %A
	%Y = sub int %X, %W
	%Z = add int %Y, 12
	ret int %Z
}
