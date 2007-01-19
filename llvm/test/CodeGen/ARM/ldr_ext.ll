; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep "ldrb"  | wc -l | grep 1 &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep "ldrsb" | wc -l | grep 1 &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep "ldrh"  | wc -l | grep 1 &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep "ldrsh" | wc -l | grep 1

int %test1(ubyte* %v) {
	%tmp = load ubyte* %v
	%tmp1 = cast ubyte %tmp to int
	ret int %tmp1
}

int %test2(ushort* %v) {
	%tmp = load ushort* %v
	%tmp1 = cast ushort %tmp to int
	ret int %tmp1
}

int %test3(sbyte* %v) {
	%tmp = load sbyte* %v
	%tmp1 = cast sbyte %tmp to int
	ret int %tmp1
}

int %test4(short* %v) {
	%tmp = load short* %v
	%tmp1 = cast short %tmp to int
	ret int %tmp1
}
