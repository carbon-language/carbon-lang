; With sub reassociation, constant folding can eliminate the 12 and -12 constants.
;
; RUN: llvm-as < %s | opt -reassociate -constprop -instcombine -die | llvm-dis | not grep 12

int "test"(int %A, int %B) {
	%X = add int -12, %A
	%Y = sub int %X, %B
	%Z = add int %Y, 12
	ret int %Z
}
