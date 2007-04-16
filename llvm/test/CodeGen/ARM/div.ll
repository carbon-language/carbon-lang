; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm > %t
; RUN: grep __divsi3  %t
; RUN: grep __udivsi3 %t
; RUN: grep __modsi3  %t
; RUN: grep __umodsi3 %t

int %f1(int %a, int %b) {
entry:
	%tmp1 = div int %a, %b
	ret int %tmp1
}

uint %f2(uint %a, uint %b) {
entry:
	%tmp1 = div uint %a, %b
	ret uint %tmp1
}

int %f3(int %a, int %b) {
entry:
	%tmp1 = rem int %a, %b
	ret int %tmp1
}

uint %f4(uint %a, uint %b) {
entry:
	%tmp1 = rem uint %a, %b
	ret uint %tmp1
}
