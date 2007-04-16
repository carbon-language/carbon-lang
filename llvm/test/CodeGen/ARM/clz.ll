; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+v5t | grep clz

declare uint %llvm.ctlz.i32(uint)

uint %test(uint %x) {
	%tmp.1 = call uint %llvm.ctlz.i32( uint %x ) 
	ret uint %tmp.1
}
