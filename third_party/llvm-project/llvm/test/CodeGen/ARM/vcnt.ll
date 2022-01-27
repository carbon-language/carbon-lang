; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - | FileCheck %s
; NB: this tests vcnt, vclz, and vcls

define <8 x i8> @vcnt8(<8 x i8>* %A) nounwind {
;CHECK-LABEL: vcnt8:
;CHECK: vcnt.8 {{d[0-9]+}}, {{d[0-9]+}}
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = call <8 x i8> @llvm.ctpop.v8i8(<8 x i8> %tmp1)
	ret <8 x i8> %tmp2
}

define <16 x i8> @vcntQ8(<16 x i8>* %A) nounwind {
;CHECK-LABEL: vcntQ8:
;CHECK: vcnt.8 {{q[0-9]+}}, {{q[0-9]+}}
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = call <16 x i8> @llvm.ctpop.v16i8(<16 x i8> %tmp1)
	ret <16 x i8> %tmp2
}

declare <8 x i8>  @llvm.ctpop.v8i8(<8 x i8>) nounwind readnone
declare <16 x i8> @llvm.ctpop.v16i8(<16 x i8>) nounwind readnone

define <8 x i8> @vclz8(<8 x i8>* %A) nounwind {
;CHECK-LABEL: vclz8:
;CHECK: vclz.i8 {{d[0-9]+}}, {{d[0-9]+}}
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = call <8 x i8> @llvm.ctlz.v8i8(<8 x i8> %tmp1, i1 0)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vclz16(<4 x i16>* %A) nounwind {
;CHECK-LABEL: vclz16:
;CHECK: vclz.i16 {{d[0-9]+}}, {{d[0-9]+}}
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = call <4 x i16> @llvm.ctlz.v4i16(<4 x i16> %tmp1, i1 0)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vclz32(<2 x i32>* %A) nounwind {
;CHECK-LABEL: vclz32:
;CHECK: vclz.i32 {{d[0-9]+}}, {{d[0-9]+}}
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = call <2 x i32> @llvm.ctlz.v2i32(<2 x i32> %tmp1, i1 0)
	ret <2 x i32> %tmp2
}

define <1 x i64> @vclz64(<1 x i64>* %A) nounwind {
;CHECK-LABEL: vclz64:
	%tmp1 = load <1 x i64>, <1 x i64>* %A
	%tmp2 = call <1 x i64> @llvm.ctlz.v1i64(<1 x i64> %tmp1, i1 0)
	ret <1 x i64> %tmp2
}

define <16 x i8> @vclzQ8(<16 x i8>* %A) nounwind {
;CHECK-LABEL: vclzQ8:
;CHECK: vclz.i8 {{q[0-9]+}}, {{q[0-9]+}}
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = call <16 x i8> @llvm.ctlz.v16i8(<16 x i8> %tmp1, i1 0)
	ret <16 x i8> %tmp2
}

define <8 x i16> @vclzQ16(<8 x i16>* %A) nounwind {
;CHECK-LABEL: vclzQ16:
;CHECK: vclz.i16 {{q[0-9]+}}, {{q[0-9]+}}
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = call <8 x i16> @llvm.ctlz.v8i16(<8 x i16> %tmp1, i1 0)
	ret <8 x i16> %tmp2
}

define <4 x i32> @vclzQ32(<4 x i32>* %A) nounwind {
;CHECK-LABEL: vclzQ32:
;CHECK: vclz.i32 {{q[0-9]+}}, {{q[0-9]+}}
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = call <4 x i32> @llvm.ctlz.v4i32(<4 x i32> %tmp1, i1 0)
	ret <4 x i32> %tmp2
}

define <2 x i64> @vclzQ64(<2 x i64>* %A) nounwind {
;CHECK-LABEL: vclzQ64:
	%tmp1 = load <2 x i64>, <2 x i64>* %A
	%tmp2 = call <2 x i64> @llvm.ctlz.v2i64(<2 x i64> %tmp1, i1 0)
	ret <2 x i64> %tmp2
}

define <8 x i8> @vclz8b(<8 x i8>* %A) nounwind {
;CHECK-LABEL: vclz8b:
;CHECK: vclz.i8 {{d[0-9]+}}, {{d[0-9]+}}
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = call <8 x i8> @llvm.ctlz.v8i8(<8 x i8> %tmp1, i1 1)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vclz16b(<4 x i16>* %A) nounwind {
;CHECK-LABEL: vclz16b:
;CHECK: vclz.i16 {{d[0-9]+}}, {{d[0-9]+}}
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = call <4 x i16> @llvm.ctlz.v4i16(<4 x i16> %tmp1, i1 1)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vclz32b(<2 x i32>* %A) nounwind {
;CHECK-LABEL: vclz32b:
;CHECK: vclz.i32 {{d[0-9]+}}, {{d[0-9]+}}
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = call <2 x i32> @llvm.ctlz.v2i32(<2 x i32> %tmp1, i1 1)
	ret <2 x i32> %tmp2
}

define <1 x i64> @vclz64b(<1 x i64>* %A) nounwind {
;CHECK-LABEL: vclz64b:
	%tmp1 = load <1 x i64>, <1 x i64>* %A
	%tmp2 = call <1 x i64> @llvm.ctlz.v1i64(<1 x i64> %tmp1, i1 1)
	ret <1 x i64> %tmp2
}

define <16 x i8> @vclzQ8b(<16 x i8>* %A) nounwind {
;CHECK-LABEL: vclzQ8b:
;CHECK: vclz.i8 {{q[0-9]+}}, {{q[0-9]+}}
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = call <16 x i8> @llvm.ctlz.v16i8(<16 x i8> %tmp1, i1 1)
	ret <16 x i8> %tmp2
}

define <8 x i16> @vclzQ16b(<8 x i16>* %A) nounwind {
;CHECK-LABEL: vclzQ16b:
;CHECK: vclz.i16 {{q[0-9]+}}, {{q[0-9]+}}
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = call <8 x i16> @llvm.ctlz.v8i16(<8 x i16> %tmp1, i1 1)
	ret <8 x i16> %tmp2
}

define <4 x i32> @vclzQ32b(<4 x i32>* %A) nounwind {
;CHECK-LABEL: vclzQ32b:
;CHECK: vclz.i32 {{q[0-9]+}}, {{q[0-9]+}}
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = call <4 x i32> @llvm.ctlz.v4i32(<4 x i32> %tmp1, i1 1)
	ret <4 x i32> %tmp2
}

define <2 x i64> @vclzQ64b(<2 x i64>* %A) nounwind {
;CHECK-LABEL: vclzQ64b:
	%tmp1 = load <2 x i64>, <2 x i64>* %A
	%tmp2 = call <2 x i64> @llvm.ctlz.v2i64(<2 x i64> %tmp1, i1 1)
	ret <2 x i64> %tmp2
}

declare <8 x i8>  @llvm.ctlz.v8i8(<8 x i8>, i1) nounwind readnone
declare <4 x i16> @llvm.ctlz.v4i16(<4 x i16>, i1) nounwind readnone
declare <2 x i32> @llvm.ctlz.v2i32(<2 x i32>, i1) nounwind readnone
declare <1 x i64> @llvm.ctlz.v1i64(<1 x i64>, i1) nounwind readnone

declare <16 x i8> @llvm.ctlz.v16i8(<16 x i8>, i1) nounwind readnone
declare <8 x i16> @llvm.ctlz.v8i16(<8 x i16>, i1) nounwind readnone
declare <4 x i32> @llvm.ctlz.v4i32(<4 x i32>, i1) nounwind readnone
declare <2 x i64> @llvm.ctlz.v2i64(<2 x i64>, i1) nounwind readnone

define <8 x i8> @vclss8(<8 x i8>* %A) nounwind {
;CHECK-LABEL: vclss8:
;CHECK: vcls.s8
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = call <8 x i8> @llvm.arm.neon.vcls.v8i8(<8 x i8> %tmp1)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vclss16(<4 x i16>* %A) nounwind {
;CHECK-LABEL: vclss16:
;CHECK: vcls.s16
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = call <4 x i16> @llvm.arm.neon.vcls.v4i16(<4 x i16> %tmp1)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vclss32(<2 x i32>* %A) nounwind {
;CHECK-LABEL: vclss32:
;CHECK: vcls.s32
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = call <2 x i32> @llvm.arm.neon.vcls.v2i32(<2 x i32> %tmp1)
	ret <2 x i32> %tmp2
}

define <16 x i8> @vclsQs8(<16 x i8>* %A) nounwind {
;CHECK-LABEL: vclsQs8:
;CHECK: vcls.s8
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = call <16 x i8> @llvm.arm.neon.vcls.v16i8(<16 x i8> %tmp1)
	ret <16 x i8> %tmp2
}

define <8 x i16> @vclsQs16(<8 x i16>* %A) nounwind {
;CHECK-LABEL: vclsQs16:
;CHECK: vcls.s16
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = call <8 x i16> @llvm.arm.neon.vcls.v8i16(<8 x i16> %tmp1)
	ret <8 x i16> %tmp2
}

define <4 x i32> @vclsQs32(<4 x i32>* %A) nounwind {
;CHECK-LABEL: vclsQs32:
;CHECK: vcls.s32
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = call <4 x i32> @llvm.arm.neon.vcls.v4i32(<4 x i32> %tmp1)
	ret <4 x i32> %tmp2
}

declare <8 x i8>  @llvm.arm.neon.vcls.v8i8(<8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vcls.v4i16(<4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vcls.v2i32(<2 x i32>) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vcls.v16i8(<16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vcls.v8i16(<8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vcls.v4i32(<4 x i32>) nounwind readnone
