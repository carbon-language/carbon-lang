; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 > %t
; RUN: grep fcvtsd %t
; RUN: grep fcvtds %t
; RUN: grep ftosizs %t
; RUN: grep ftouizs %t
; RUN: grep ftosizd %t
; RUN: grep ftouizd %t
; RUN: grep fsitos %t
; RUN: grep fsitod %t
; RUN: grep fuitos %t
; RUN: grep fuitod %t

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
