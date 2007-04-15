; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+v5TE
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+v5TE | \
; RUN:   grep smulbt | wc -l | grep 1
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+v5TE | \
; RUN:   grep smultt | wc -l | grep 1
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+v5TE | \
; RUN:   grep smlabt | wc -l | grep 1

%x = weak global short 0
%y = weak global short 0

int %f1(int %y) {
	%tmp = load short* %x
	%tmp1 = add short %tmp, 2
	%tmp2 = cast short %tmp1 to int
	%tmp3 = shr int %y, ubyte 16
	%tmp4 = mul int %tmp2, %tmp3
	ret int %tmp4
}

int %f2(int %x, int %y) {
	%tmp1 = shr int %x, ubyte 16
	%tmp3 = shr int %y, ubyte 16
	%tmp4 = mul int %tmp3, %tmp1
	ret int %tmp4
}

int %f3(int %a, short %x, int %y) {
	%tmp = cast short %x to int
	%tmp2 = shr int %y, ubyte 16
	%tmp3 = mul int %tmp2, %tmp
	%tmp5 = add int %tmp3, %a
	ret int %tmp5
}
