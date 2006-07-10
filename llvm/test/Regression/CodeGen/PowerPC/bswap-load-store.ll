; RUN: llvm-as < %s | llc -march=ppc32 | grep 'stwbrx\|lwbrx\|sthbrx\|lhbrx' | wc -l | grep 4 &&
; RUN: llvm-as < %s | llc -march=ppc32 | not grep rlwinm &&
; RUN: llvm-as < %s | llc -march=ppc32 | not grep rlwimi &&
; RUN: llvm-as < %s | llc -march=ppc64 | grep 'stwbrx\|lwbrx\|sthbrx\|lhbrx' | wc -l | grep 4 &&
; RUN: llvm-as < %s | llc -march=ppc64 | not grep rlwinm &&
; RUN: llvm-as < %s | llc -march=ppc64 | not grep rlwimi

void %STWBRX(uint %i, sbyte* %ptr, int %off) {
	%tmp1 = getelementptr sbyte* %ptr, int %off
	%tmp1 = cast sbyte* %tmp1 to uint*
	%tmp13 = tail call uint %llvm.bswap.i32(uint %i)
	store uint %tmp13, uint* %tmp1
	ret void
}

uint %LWBRX(sbyte* %ptr, int %off) {
	%tmp1 = getelementptr sbyte* %ptr, int %off
	%tmp1 = cast sbyte* %tmp1 to uint*		
	%tmp = load uint* %tmp1		
	%tmp14 = tail call uint %llvm.bswap.i32( uint %tmp )
	ret uint %tmp14
}

void %STHBRX(ushort %s, sbyte* %ptr, int %off) {
	%tmp1 = getelementptr sbyte* %ptr, int %off
	%tmp1 = cast sbyte* %tmp1 to ushort*
	%tmp5 = call ushort %llvm.bswap.i16( ushort %s )
	store ushort %tmp5, ushort* %tmp1
	ret void
}

ushort %LHBRX(sbyte* %ptr, int %off) {
	%tmp1 = getelementptr sbyte* %ptr, int %off
	%tmp1 = cast sbyte* %tmp1 to ushort*
	%tmp = load ushort* %tmp1
	%tmp6 = call ushort %llvm.bswap.i16(ushort %tmp)
	ret ushort %tmp6
}

declare uint %llvm.bswap.i32(uint)

declare ushort %llvm.bswap.i16(ushort)
