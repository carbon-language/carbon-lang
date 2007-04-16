; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | \
; RUN:   grep -- {-2147483648} | wc -l | grep 3
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep mvn | wc -l | grep 3
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep adds | wc -l | grep 1
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep adc | wc -l | grep 1
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep {subs } | wc -l | grep 1
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep sbc | wc -l | grep 1
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | \
; RUN:   grep smull | wc -l | grep 1
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | \
; RUN:   grep umull | wc -l | grep 1 
; RUN: llvm-upgrade < %s | llvm-as | llc -march=thumb | \
; RUN:   grep mvn | wc -l | grep 1 
; RUN: llvm-upgrade < %s | llvm-as | llc -march=thumb | \
; RUN:   grep adc | wc -l | grep 1 
; RUN: llvm-upgrade < %s | llvm-as | llc -march=thumb | \
; RUN:   grep sbc | wc -l | grep 1 
; RUN: llvm-upgrade < %s | llvm-as | llc -march=thumb | grep __muldi3
; END.

long %f1() {
entry:
	ret long 0
}

long %f2() {
entry:
	ret long 1
}

long %f3() {
entry:
	ret long 2147483647
}

long %f4() {
entry:
	ret long 2147483648
}

long %f5() {
entry:
	ret long 9223372036854775807
}

ulong %f6(ulong %x, ulong %y) {
entry:
	%tmp1 = add ulong %y, 1
	ret ulong %tmp1
}

void %f7() {
entry:
	%tmp = call long %f8()
	ret void
}
declare long %f8()

long %f9(long %a, long %b) {
entry:
	%tmp = sub long %a, %b
	ret long %tmp
}

long %f(int %a, int %b) {
entry:
	%tmp = cast int %a to long
	%tmp1 = cast int %b to long
	%tmp2 = mul long %tmp1, %tmp
	ret long %tmp2
}

ulong %g(uint %a, uint %b) {
entry:
	%tmp = cast uint %a to ulong
	%tmp1 = cast uint %b to ulong
	%tmp2 = mul ulong %tmp1, %tmp
	ret ulong %tmp2
}

ulong %f10() {
entry:
	%a = alloca ulong, align 8
	%retval = load ulong* %a
	ret ulong %retval
}
