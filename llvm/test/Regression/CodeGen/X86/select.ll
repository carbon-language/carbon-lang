; RUN: llvm-as < %s | llc -march=x86

bool %boolSel(bool %A, bool %B, bool %C) {
	%X = select bool %A, bool %B, bool %C
	ret bool %X
}

sbyte %byteSel(bool %A, sbyte %B, sbyte %C) {
	%X = select bool %A, sbyte %B, sbyte %C
	ret sbyte %X
}

short %shortSel(bool %A, short %B, short %C) {
	%X = select bool %A, short %B, short %C
	ret short %X
}

int %intSel(bool %A, int %B, int %C) {
	%X = select bool %A, int %B, int %C
	ret int %X
}

long %longSel(bool %A, long %B, long %C) {
	%X = select bool %A, long %B, long %C
	ret long %X
}

double %doubleSel(bool %A, double %B, double %C) {
	%X = select bool %A, double %B, double %C
	ret double %X
}

