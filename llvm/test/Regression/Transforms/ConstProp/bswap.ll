; bswap should be constant folded when it is passed a constant argument

; RUN: llvm-as < %s | opt -constprop | llvm-dis | not grep call

declare ushort %llvm.bswap.i16(ushort)
declare uint %llvm.bswap.i32(uint)
declare ulong %llvm.bswap.i64(ulong)

ushort %W() {
	%Z = call ushort %llvm.bswap.i16(ushort 1)
	ret ushort %Z
}

uint %X() {
	%Z = call uint %llvm.bswap.i32(uint 1)
	ret uint %Z
}

ulong %Y() {
	%Z = call ulong %llvm.bswap.i64(ulong 1)
	ret ulong %Z
}
