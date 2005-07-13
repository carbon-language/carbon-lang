; There should be exactly two calls here (memset and malloc), no more.
; RUN: llvm-as < %s | llc -march=alpha | grep jsr | wc -l | grep 2

implementation   ; Functions:

declare void %llvm.memset(sbyte*, ubyte, ulong, uint)

bool %l12_l94_bc_divide_endif_2E_3_2E_ce(int* %tmp.71.reload, uint %scale2.1.3, uint %extra.0, %typedef.bc_struct* %n1, %typedef.bc_struct* %n2, int* %tmp.92.reload, uint %tmp.94.reload, int* %tmp.98.reload, uint %tmp.100.reload, sbyte** %tmp.112.out, uint* %tmp.157.out, sbyte** %tmp.158.out) {
newFuncRoot:
	%tmp.120 = add uint %extra.0, 2		; <uint> [#uses=1]
	%tmp.122 = add uint %tmp.120, %tmp.94.reload		; <uint> [#uses=1]
	%tmp.123 = add uint %tmp.122, %tmp.100.reload		; <uint> [#uses=2]
	%tmp.112 = malloc sbyte, uint %tmp.123		; <sbyte*> [#uses=3]
	%tmp.137 = cast uint %tmp.123 to ulong		; <ulong> [#uses=1]
	tail call void %llvm.memset( sbyte* %tmp.112, ubyte 0, ulong %tmp.137, uint 0 )
	ret bool true
}
