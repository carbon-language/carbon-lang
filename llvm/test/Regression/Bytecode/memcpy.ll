; RUN: llvm-dis %s.bc-16 -o /dev/null -f &&
; RUN: llc %s.bc-16 -o /dev/null -f -march=c &&
; RUN: llvm-as < %s

void %test(int* %P, int* %Q) {
entry:
	%tmp.1 = cast int* %P to sbyte*		; <sbyte*> [#uses=2]
	%tmp.3 = cast int* %Q to sbyte*		; <sbyte*> [#uses=3]
	tail call void %llvm.memcpy( sbyte* %tmp.1, sbyte* %tmp.3, uint 100000, uint 1 )
	tail call void %llvm.memcpy( sbyte* %tmp.1, sbyte* %tmp.3, ulong 100000, uint 1 )
	tail call void %llvm.memset( sbyte* %tmp.3, ubyte 14, uint 10000, uint 0 )
	tail call void %llvm.memmove( sbyte* %tmp.1, sbyte* %tmp.3, uint 123124, uint 1 )
	ret void
}

declare void %llvm.memcpy(sbyte*, sbyte*, uint, uint)
declare void %llvm.memcpy(sbyte*, sbyte*, ulong, uint)

declare void %llvm.memset(sbyte*, ubyte, uint, uint)

declare void %llvm.memmove(sbyte*, sbyte*, uint, uint)
