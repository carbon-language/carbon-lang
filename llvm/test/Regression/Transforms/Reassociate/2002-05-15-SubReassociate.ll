; With sub reassociation, constant folding can eliminate all of the constants.
;
; RUN: if as < %s | opt -reassociate -constprop -instcombine -dce | dis | grep add
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

int "test"(int %A, int %B) {
	%W = add int -5, %B
	%X = add int -7, %A
	%Y = sub int %X, %W
	%Z = add int %Y, 12
	ret int %Z
}
