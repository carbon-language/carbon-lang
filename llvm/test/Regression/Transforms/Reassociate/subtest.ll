; With sub reassociation, constant folding can eliminate the 12 and -12 constants.
;
; RUN: if as < %s | opt -reassociate -constprop -instcombine -die | dis | grep add
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

int "test"(int %A, int %B) {
	%X = add int -12, %A
	%Y = sub int %X, %B
	%Z = add int %Y, 12
	ret int %Z
}
