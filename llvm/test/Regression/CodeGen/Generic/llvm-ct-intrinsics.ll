; Make sure this testcase is supported by all code generators
; RUN: llvm-upgrade < %s | llvm-as | llc

declare ulong %llvm.ctpop.i64(ulong)
declare uint %llvm.ctpop.i32(uint)
declare ushort %llvm.ctpop.i16(ushort)
declare ubyte %llvm.ctpop.i8(ubyte)

void %ctpoptest(ubyte %A, ushort %B, uint %C, ulong %D, 
                ubyte *%AP, ushort* %BP, uint* %CP, ulong* %DP) {
	%a = call ubyte %llvm.ctpop.i8(ubyte %A)
	%b = call ushort %llvm.ctpop.i16(ushort %B)
	%c = call uint %llvm.ctpop.i32(uint %C)
	%d = call ulong %llvm.ctpop.i64(ulong %D)

	store ubyte %a, ubyte* %AP
	store ushort %b, ushort* %BP
	store uint   %c, uint* %CP
	store ulong  %d, ulong* %DP
	ret void
}

declare ulong %llvm.ctlz.i64(ulong)
declare uint %llvm.ctlz.i32(uint)
declare ushort %llvm.ctlz.i16(ushort)
declare ubyte %llvm.ctlz.i8(ubyte)

void %ctlztest(ubyte %A, ushort %B, uint %C, ulong %D, 
               ubyte *%AP, ushort* %BP, uint* %CP, ulong* %DP) {
	%a = call ubyte %llvm.ctlz.i8(ubyte %A)
	%b = call ushort %llvm.ctlz.i16(ushort %B)
	%c = call uint %llvm.ctlz.i32(uint %C)
	%d = call ulong %llvm.ctlz.i64(ulong %D)

	store ubyte %a, ubyte* %AP
	store ushort %b, ushort* %BP
	store uint   %c, uint* %CP
	store ulong  %d, ulong* %DP
	ret void
}

declare ulong %llvm.cttz.i64(ulong)
declare uint %llvm.cttz.i32(uint)
declare ushort %llvm.cttz.i16(ushort)
declare ubyte %llvm.cttz.i8(ubyte)

void %cttztest(ubyte %A, ushort %B, uint %C, ulong %D, 
               ubyte *%AP, ushort* %BP, uint* %CP, ulong* %DP) {
	%a = call ubyte %llvm.cttz.i8(ubyte %A)
	%b = call ushort %llvm.cttz.i16(ushort %B)
	%c = call uint %llvm.cttz.i32(uint %C)
	%d = call ulong %llvm.cttz.i64(ulong %D)

	store ubyte %a, ubyte* %AP
	store ushort %b, ushort* %BP
	store uint   %c, uint* %CP
	store ulong  %d, ulong* %DP
	ret void
}
