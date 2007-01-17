; RUN: llvm-upgrade < %s | llvm-as -o /dev/null -f

void %test(int* %P, int* %Q) {
entry:
	%tmp.1 = cast int* %P to sbyte*		; <sbyte*> [#uses=2]
	%tmp.3 = cast int* %Q to sbyte*		; <sbyte*> [#uses=3]
	tail call void %llvm.memcpy.i32( sbyte* %tmp.1, sbyte* %tmp.3, uint 100000, uint 1 )
	tail call void %llvm.memcpy.i64( sbyte* %tmp.1, sbyte* %tmp.3, ulong 100000, uint 1 )
	tail call void %llvm.memset.i32( sbyte* %tmp.3, ubyte 14, uint 10000, uint 0 )
	tail call void %llvm.memmove.i32( sbyte* %tmp.1, sbyte* %tmp.3, uint 123124, uint 1 )
	ret void
}

declare void %llvm.memcpy.i32(sbyte*, sbyte*, uint, uint)
declare void %llvm.memcpy.i64(sbyte*, sbyte*, ulong, uint)
declare void %llvm.memset.i32(sbyte*, ubyte, uint, uint)
declare void %llvm.memmove.i32(sbyte*, sbyte*, uint, uint)
