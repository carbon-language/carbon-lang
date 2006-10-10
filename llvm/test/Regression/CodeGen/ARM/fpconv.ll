; RUN: llvm-as < %s | llc -march=arm &&
; RUN: llvm-as < %s | llc -march=arm | grep fcvtds &&
; RUN: llvm-as < %s | llc -march=arm | grep fcvtsd

float %f1(double %x) {
entry:
	%tmp1 = cast double %x to float
	ret float %tmp1
}

double %f2(float %x) {
entry:
	%tmp1 = cast float %x to double
	ret double %tmp1
}

int %f3(float %x) {
entry:
        %tmp = cast float %x to int
        ret int %tmp
}

int %f4(double %x) {
entry:
        %tmp = cast double %x to int
        ret int %tmp
}

uint %f5(float %x) {
entry:
        %tmp = cast float %x to uint
        ret uint %tmp
}

uint %f6(double %x) {
entry:
        %tmp = cast double %x to uint
        ret uint %tmp
}
