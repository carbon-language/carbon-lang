; RUN: if as < %s | opt -pinodes -instcombine -die | dis | grep add
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

int "test"(int %i, int %j) {
	%c = seteq int %i, 0
	br bool %c, label %iIsZero, label %iIsNotZero

iIsZero:
	%j2 = add int %j, %i      ; This is always equal to j
	ret int %j2

iIsNotZero:
	ret int 1
}
