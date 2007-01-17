; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm

void %f() {
entry:
	call void %llvm.memmove.i32( sbyte* null, sbyte* null, uint 64, uint 0 )
	call void %llvm.memcpy.i32( sbyte* null, sbyte* null, uint 64, uint 0 )
	call void %llvm.memset.i32( sbyte* null, ubyte 64, uint 0, uint 0 )
	unreachable
}

declare void %llvm.memmove.i32(sbyte*, sbyte*, uint, uint)
declare void %llvm.memcpy.i32(sbyte*, sbyte*, uint, uint)
declare void %llvm.memset.i32(sbyte*, ubyte, uint, uint)
