; This testcase shows a bug where an common subexpression exists, but there
; is no shared dominator block that the expression can be hoisted out to.
;
; RUN: if as < %s | opt -load-vn -gcse | dis | grep load
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

int %test(int* %P) {
	store int 5, int* %P
	%Z = load int* %P
        ret int %Z
}

