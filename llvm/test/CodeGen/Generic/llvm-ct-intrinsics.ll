; Make sure this testcase is supported by all code generators
; RUN: llvm-upgrade < %s | llvm-as | llc

declare uint %llvm.ctpop.i64(ulong)
declare uint %llvm.ctpop.i32(uint)
declare uint %llvm.ctpop.i16(ushort)
declare uint %llvm.ctpop.i8(ubyte)

void %ctpoptest(ubyte %A, ushort %B, uint %C, ulong %D, 
                uint *%AP, uint* %BP, uint* %CP, uint* %DP) {
	%a = call uint %llvm.ctpop.i8(ubyte %A)
	%b = call uint %llvm.ctpop.i16(ushort %B)
	%c = call uint %llvm.ctpop.i32(uint %C)
	%d = call uint %llvm.ctpop.i64(ulong %D)

	store uint %a, uint* %AP
	store uint %b, uint* %BP
	store uint %c, uint* %CP
	store uint %d, uint* %DP
	ret void
}

declare uint %llvm.ctlz.i64(ulong)
declare uint %llvm.ctlz.i32(uint)
declare uint %llvm.ctlz.i16(ushort)
declare uint %llvm.ctlz.i8(ubyte)

void %ctlztest(ubyte %A, ushort %B, uint %C, ulong %D, 
               uint *%AP, uint* %BP, uint* %CP, uint* %DP) {
	%a = call uint %llvm.ctlz.i8(ubyte %A)
	%b = call uint %llvm.ctlz.i16(ushort %B)
	%c = call uint %llvm.ctlz.i32(uint %C)
	%d = call uint %llvm.ctlz.i64(ulong %D)

	store uint %a, uint* %AP
	store uint %b, uint* %BP
	store uint %c, uint* %CP
	store uint %d, uint* %DP
	ret void
}

declare uint %llvm.cttz.i64(ulong)
declare uint %llvm.cttz.i32(uint)
declare uint %llvm.cttz.i16(ushort)
declare uint %llvm.cttz.i8(ubyte)

void %cttztest(ubyte %A, ushort %B, uint %C, ulong %D, 
               uint *%AP, uint* %BP, uint* %CP, uint* %DP) {
	%a = call uint %llvm.cttz.i8(ubyte %A)
	%b = call uint %llvm.cttz.i16(ushort %B)
	%c = call uint %llvm.cttz.i32(uint %C)
	%d = call uint %llvm.cttz.i64(ulong %D)

	store uint %a, uint* %AP
	store uint %b, uint* %BP
	store uint %c, uint* %CP
	store uint %d, uint* %DP
	ret void
}
