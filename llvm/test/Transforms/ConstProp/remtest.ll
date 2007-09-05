; Ensure constant propagation of remainder instructions is working correctly.

; RUN: llvm-upgrade < %s | llvm-as | opt -constprop -die | llvm-dis | not grep rem

int %test1() {
	%R = rem int 4, 3
	ret int %R
}

int %test2() {
	%R = rem int 123, -23
	ret int %R
}

float %test3() {
	%R = rem float 0x4028E66660000000, 0x405ECDA1C0000000
	ret float %R
}

double %test4() {
	%R = rem double 0x4073833BEE07AFF8, 0x4028AAABB2A0D19C
	ret double %R
}
