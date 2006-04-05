; RUN: llvm-as < %s | llc -march=x86 -mcpu=yonah 

target endian = little
target pointersize = 32
target triple = "i686-apple-darwin8.6.1"
	%struct.GLTColor4 = type { float, float, float, float }
	%struct.GLTCoord3 = type { float, float, float }
	%struct.__GLIContextRec = type { { %struct.anon, { [24 x [16 x float]], [24 x [16 x float]] }, %struct.GLTColor4, { float, float, float, float, %struct.GLTCoord3, float } }, { float, float, float, float, float, float, float, float, [4 x uint], [4 x uint], [4 x uint] } }
	%struct.__GLvertex = type { %struct.GLTColor4, %struct.GLTColor4, %struct.GLTColor4, %struct.GLTColor4, %struct.GLTColor4, %struct.GLTCoord3, float, %struct.GLTColor4, float, float, float, ubyte, ubyte, ubyte, ubyte, [4 x float], [2 x sbyte*], uint, uint, [16 x %struct.GLTColor4] }
	%struct.anon = type { float, float, float, float, float, float, float, float }

implementation   ; Functions:

declare <4 x float> %llvm.x86.sse.cmp.ps(<4 x float>, <4 x float>, sbyte)

declare <4 x int> %llvm.x86.sse2.packssdw.128(<4 x int>, <4 x int>)

declare int %llvm.x86.sse2.pmovmskb.128(<16 x sbyte>)

void %gleLLVMVecInterpolateClip() {
entry:
	br bool false, label %cond_false, label %cond_false183

cond_false:		; preds = %entry
	br bool false, label %cond_false183, label %cond_true69

cond_true69:		; preds = %cond_false
	ret void

cond_false183:		; preds = %cond_false, %entry
	%vuizmsk.0.1 = phi <4 x int> [ < int -1, int -1, int -1, int 0 >, %entry ], [ < int -1, int 0, int 0, int 0 >, %cond_false ]		; <<4 x int>> [#uses=2]
	%tmp192 = extractelement <4 x int> %vuizmsk.0.1, uint 2		; <int> [#uses=1]
	%tmp193 = extractelement <4 x int> %vuizmsk.0.1, uint 3		; <int> [#uses=2]
	%tmp195 = insertelement <4 x int> zeroinitializer, int %tmp192, uint 1		; <<4 x int>> [#uses=1]
	%tmp196 = insertelement <4 x int> %tmp195, int %tmp193, uint 2		; <<4 x int>> [#uses=1]
	%tmp197 = insertelement <4 x int> %tmp196, int %tmp193, uint 3		; <<4 x int>> [#uses=1]
	%tmp336 = and <4 x int> zeroinitializer, %tmp197		; <<4 x int>> [#uses=1]
	%tmp337 = cast <4 x int> %tmp336 to <4 x float>		; <<4 x float>> [#uses=1]
	%tmp378 = tail call <4 x float> %llvm.x86.sse.cmp.ps( <4 x float> %tmp337, <4 x float> zeroinitializer, sbyte 1 )		; <<4 x float>> [#uses=1]
	%tmp379 = cast <4 x float> %tmp378 to <4 x int>		; <<4 x int>> [#uses=1]
	%tmp388 = tail call <4 x int> %llvm.x86.sse2.packssdw.128( <4 x int> zeroinitializer, <4 x int> %tmp379 )		; <<4 x int>> [#uses=1]
	%tmp392 = cast <4 x int> %tmp388 to <8 x short>		; <<8 x short>> [#uses=1]
	%tmp399 = extractelement <8 x short> %tmp392, uint 7		; <short> [#uses=1]
	%tmp423 = insertelement <8 x short> zeroinitializer, short %tmp399, uint 7		; <<8 x short>> [#uses=1]
	%tmp427 = cast <8 x short> %tmp423 to <16 x sbyte>		; <<16 x sbyte>> [#uses=1]
	%tmp428 = tail call int %llvm.x86.sse2.pmovmskb.128( <16 x sbyte> %tmp427 )		; <int> [#uses=1]
	%tmp432 = cast int %tmp428 to sbyte		; <sbyte> [#uses=1]
	%tmp = and sbyte %tmp432, 42		; <sbyte> [#uses=1]
	%tmp436 = cast sbyte %tmp to ubyte		; <ubyte> [#uses=1]
	%tmp446 = cast ubyte %tmp436 to uint		; <uint> [#uses=1]
	%tmp447 = shl uint %tmp446, ubyte 24		; <uint> [#uses=1]
	%tmp449 = or uint 0, %tmp447		; <uint> [#uses=1]
	store uint %tmp449, uint* null
	ret void
}
