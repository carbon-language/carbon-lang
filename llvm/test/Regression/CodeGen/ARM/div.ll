; RUN: llvm-as < %s | llc -march=arm &&
; RUN: llvm-as < %s | llc -march=arm | grep __divsi3  &&
; RUN: llvm-as < %s | llc -march=arm | grep __udivsi3 &&
; RUN: llvm-as < %s | llc -march=arm | grep __modsi3  &&
; RUN: llvm-as < %s | llc -march=arm | grep __umodsi3

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
