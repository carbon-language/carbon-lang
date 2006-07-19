; RUN: llvm-as < %s | llc -march=ppc32

void %img2buf(int %symbol_size_in_bytes, ushort* %ui16) {
	%tmp93 = load ushort* null		; <ushort> [#uses=1]
	%tmp99 = call ushort %llvm.bswap.i16( ushort %tmp93 )		; <ushort> [#uses=1]
	store ushort %tmp99, ushort* %ui16
	ret void
}

declare ushort %llvm.bswap.i16(ushort)
