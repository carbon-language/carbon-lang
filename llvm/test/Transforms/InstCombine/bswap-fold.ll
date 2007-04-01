; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | grep ret | wc -l | grep 3 &&
; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | not grep 'call.*bswap'

bool %test1(ushort %tmp2) {
	%tmp10 = call ushort %llvm.bswap.i16.i16( ushort %tmp2 )		
	%tmp = seteq ushort %tmp10, 1		
	ret bool %tmp
}

bool %test2(uint %tmp) {
	%tmp34 = tail call uint %llvm.bswap.i32.i32( uint %tmp )		
	%tmp = seteq uint %tmp34, 1		
	ret bool %tmp
}

bool %test3(ulong %tmp) {
	%tmp34 = tail call ulong %llvm.bswap.i64.i64( ulong %tmp )		
	%tmp = seteq ulong %tmp34, 1		
	ret bool %tmp
}

declare ulong %llvm.bswap.i64.i64(ulong)

declare ushort %llvm.bswap.i16.i16(ushort)

declare uint %llvm.bswap.i32.i32(uint)
