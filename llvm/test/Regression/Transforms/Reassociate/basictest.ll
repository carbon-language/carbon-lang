; With reassociation, constant folding can eliminate the 12 and -12 constants.
;
; RUN: if as < %s | opt -reassociate -constprop -instcombine -die | dis | grep add
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

int "test"(int %arg) {
	%tmp1 = sub int -12, %arg
	%tmp2 = add int %tmp1, 12
	ret int %tmp2
}
