; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 | grep fcvtsd &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 | grep fcvtds &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 | grep ftosizs &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 | grep ftouizs &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 | grep ftosizd &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 | grep ftouizd &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 | grep fsitos &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 | grep fsitod &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 | grep fuitos &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+vfp2 | grep fuitod

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
