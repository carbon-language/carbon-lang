; RUN: llvm-as < %s | llc -march=ppc32 | not grep cmp

int %seli32_1(int %a) {
entry:
	%tmp.1 = setlt int %a, 0
	%retval = select bool %tmp.1, int 5, int 0
	ret int %retval
}

int %seli32_2(int %a, int %b) {
entry:
	%tmp.1 = setlt int %a, 0
	%retval = select bool %tmp.1, int %b, int 0
	ret int %retval
}

int %seli32_3(int %a, short %b) {
entry:
	%tmp.2 = cast short %b to int
	%tmp.1 = setlt int %a, 0
	%retval = select bool %tmp.1, int %tmp.2, int 0
	ret int %retval
}

int %seli32_4(int %a, ushort %b) {
entry:
	%tmp.2 = cast ushort %b to int
	%tmp.1 = setlt int %a, 0
	%retval = select bool %tmp.1, int %tmp.2, int 0
	ret int %retval
}

short %seli16_1(short %a) {
entry:
	%tmp.1 = setlt short %a, 0
	%retval = select bool %tmp.1, short 7, short 0
	ret short %retval
}

short %seli16_2(int %a, short %b) {
entry:
	%tmp.1 = setlt int %a, 0
	%retval = select bool %tmp.1, short %b, short 0
	ret short %retval
}
