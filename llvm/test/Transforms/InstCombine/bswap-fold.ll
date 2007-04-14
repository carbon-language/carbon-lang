; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | \
; RUN:   grep ret | wc -l | grep 3
; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | \
; RUN:   not grep call.*bswap

bool %test1(ushort %tmp2) {
	%tmp10 = call ushort %llvm.bswap.i16( ushort %tmp2 )		
	%tmp = seteq ushort %tmp10, 1		
	ret bool %tmp
}

bool %test2(uint %tmp) {
	%tmp34 = tail call uint %llvm.bswap.i32( uint %tmp )		
	%tmp = seteq uint %tmp34, 1		
	ret bool %tmp
}

declare uint %llvm.bswap.i32(uint)

bool %test3(ulong %tmp) {
	%tmp34 = tail call ulong %llvm.bswap.i64( ulong %tmp )		
	%tmp = seteq ulong %tmp34, 1		
	ret bool %tmp
}

declare ulong %llvm.bswap.i64(ulong)

declare ushort %llvm.bswap.i16(ushort)
