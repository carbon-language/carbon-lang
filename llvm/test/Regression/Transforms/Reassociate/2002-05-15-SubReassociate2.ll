; With sub reassociation, constant folding can eliminate the two 12 constants.
;
; RUN: if as < %s | opt -reassociate -constprop -dce | dis | grep 12
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

int "test"(int %A, int %B, int %C, int %D) {
	%M = add int %A, 12
	%N = add int %M, %B
	%O = add int %N, %C
	%P = sub int %D, %O
	%Q = add int %P, 12
	ret int %Q
}
