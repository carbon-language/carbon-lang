; RUN: llvm-as < %s | opt -instcombine -disable-output	

	%struct.rtx_const = type { uint, { %union.real_extract } }
	%struct.rtx_def = type { int, [1 x %union.rtunion_def] }
	%union.real_extract = type { double }
	%union.rtunion_def = type { uint }

implementation   ; Functions:

fastcc void %decode_rtx_const(%struct.rtx_def* %x, %struct.rtx_const* %value) {
	%tmp.54 = getelementptr %struct.rtx_const* %value, int 0, uint 0		; <uint*> [#uses=1]
	%tmp.56 = getelementptr %struct.rtx_def* %x, int 0, uint 0		; <int*> [#uses=1]
	%tmp.57 = load int* %tmp.56		; <int> [#uses=1]
	%tmp.58 = shl int %tmp.57, ubyte 8		; <int> [#uses=1]
	%tmp.59 = shr int %tmp.58, ubyte 24		; <int> [#uses=1]
	%tmp.60 = cast int %tmp.59 to ushort		; <ushort> [#uses=1]
	%tmp.61 = cast ushort %tmp.60 to uint		; <uint> [#uses=1]
	%tmp.62 = shl uint %tmp.61, ubyte 16		; <uint> [#uses=1]
	%tmp.65 = or uint 0, %tmp.62		; <uint> [#uses=1]
	store uint %tmp.65, uint* %tmp.54
	ret void
}
