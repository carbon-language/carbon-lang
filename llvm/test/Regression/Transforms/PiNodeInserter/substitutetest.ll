; RUN: as < %s | opt -pinodes -instcombine -die | dis | not grep add

int "test"(int %i, int %j) {
	%c = seteq int %i, 0
	br bool %c, label %iIsZero, label %iIsNotZero

iIsZero:
	%j2 = add int %j, %i      ; This is always equal to j
	ret int %j2

iIsNotZero:
	ret int 1
}
