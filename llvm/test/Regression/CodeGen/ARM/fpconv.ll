; RUN: llvm-as < %s | llc -march=arm &&
; RUN: llvm-as < %s | llc -march=arm | grep fcvtsd &&
; RUN: llvm-as < %s | llc -march=arm | grep fcvtds &&
; RUN: llvm-as < %s | llc -march=arm | grep ftosis &&
; RUN: llvm-as < %s | llc -march=arm | grep ftouis &&
; RUN: llvm-as < %s | llc -march=arm | grep ftosid &&
; RUN: llvm-as < %s | llc -march=arm | grep ftouid &&
; RUN: llvm-as < %s | llc -march=arm | grep fsitos &&
; RUN: llvm-as < %s | llc -march=arm | grep fsitod &&
; RUN: llvm-as < %s | llc -march=arm | grep fuitos &&
; RUN: llvm-as < %s | llc -march=arm | grep fuitod

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

uint %f4(float %x) {
entry:
        %tmp = cast float %x to uint
        ret uint %tmp
}

int %f5(double %x) {
entry:
        %tmp = cast double %x to int
        ret int %tmp
}

uint %f6(double %x) {
entry:
        %tmp = cast double %x to uint
        ret uint %tmp
}

float %f7(int %a) {
entry:
	%tmp = cast int %a to float
	ret float %tmp
}

double %f8(int %a) {
entry:
        %tmp = cast int %a to double
        ret double %tmp
}

float %f9(uint %a) {
entry:
	%tmp = cast uint %a to float
	ret float %tmp
}

double %f10(uint %a) {
entry:
	%tmp = cast uint %a to double
	ret double %tmp
}
