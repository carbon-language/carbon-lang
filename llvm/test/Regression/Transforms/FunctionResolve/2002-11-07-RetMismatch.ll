; RUN: if as < %s | opt -funcresolve -globaldce | dis | grep declare
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

declare void %qsortg(sbyte*, int, int)

void %test() {
	call void %qsortg(sbyte* null, int 0, int 0)
	ret void
}

int %qsortg(sbyte* %base, int %n, int %size) {
	ret int %n
}
