; RUN: if as < %s | opt -reassociate -instcombine -constprop -dce | dis | grep add
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

int "test"(int %A) {
	%X = add int %A, 1
	%Y = add int %A, 1
	%r = sub int %X, %Y
	ret int %r               ; Should be equal to 0!
}
