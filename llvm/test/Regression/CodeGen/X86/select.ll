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

sbyte %foldSel(bool %A, sbyte %B, sbyte %C) {
	%Cond = setlt sbyte %B, %C
	%X = select bool %Cond, sbyte %B, sbyte %C
	ret sbyte %X
}

int %foldSel2(bool %A, int %B, int %C) {
	%Cond = seteq int %B, %C
	%X = select bool %Cond, int %B, int %C
	ret int %X
}

int %foldSel2(bool %A, int %B, int %C, double %X, double %Y) {
	%Cond = setlt double %X, %Y
	%X = select bool %Cond, int %B, int %C
	ret int %X
}

float %foldSel3(bool %A, float %B, float %C, uint %X, uint %Y) {
	%Cond = setlt uint %X, %Y
	%X = select bool %Cond, float %B, float %C
	ret float %X
}

float %nofoldSel4(bool %A, float %B, float %C, int %X, int %Y) {
	; X86 doesn't have a cmov that reads the right flags for this!
	%Cond = setlt int %X, %Y
	%X = select bool %Cond, float %B, float %C
	ret float %X
}

