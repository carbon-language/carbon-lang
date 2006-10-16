; RUN: llvm-as < %s | llc -march=arm &&
; RUN: llvm-as < %s | llc -march=arm | grep __ashldi3 &&
; RUN: llvm-as < %s | llc -march=arm | grep __ashrdi3 &&
; RUN: llvm-as < %s | llc -march=arm | grep __lshrdi3
uint %f1(ulong %x, ubyte %y) {
entry:
	%a = shl ulong %x, ubyte %y
	%b = cast ulong %a to uint
	ret uint %b
}

uint %f2(long %x, ubyte %y) {
entry:
	%a = shr long %x, ubyte %y
	%b = cast long %a to uint
	ret uint %b
}

uint %f3(ulong %x, ubyte %y) {
entry:
	%a = shr ulong %x, ubyte %y
	%b = cast ulong %a to uint
	ret uint %b
}
