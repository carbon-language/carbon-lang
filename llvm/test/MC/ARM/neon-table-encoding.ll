; RUN: llc -show-mc-encoding -march=arm -mcpu=cortex-a8 -mattr=+neon < %s | FileCheck %s

; XFAIL: *

%struct.__neon_int8x8x2_t = type { <8 x i8>, <8 x i8> }
%struct.__neon_int8x8x3_t = type { <8 x i8>,  <8 x i8>, <8 x i8> }
%struct.__neon_int8x8x4_t = type { <8 x i8>,  <8 x i8>,  <8 x i8>, <8 x i8> }

define <8 x i8> @vtbl1(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vtbl.8	d16, {d17}, d16         @ encoding: [0xa0,0x08,0xf1,0xf3]
	%tmp3 = call <8 x i8> @llvm.arm.neon.vtbl1(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <8 x i8> @vtbl2(<8 x i8>* %A, %struct.__neon_int8x8x2_t* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load %struct.__neon_int8x8x2_t* %B
        %tmp3 = extractvalue %struct.__neon_int8x8x2_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int8x8x2_t %tmp2, 1
; CHECK: vtbl.8	d16, {d16, d17}, d18    @ encoding: [0xa2,0x09,0xf0,0xf3]
	%tmp5 = call <8 x i8> @llvm.arm.neon.vtbl2(<8 x i8> %tmp1, <8 x i8> %tmp3, <8 x i8> %tmp4)
	ret <8 x i8> %tmp5
}

define <8 x i8> @vtbl3(<8 x i8>* %A, %struct.__neon_int8x8x3_t* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load %struct.__neon_int8x8x3_t* %B
        %tmp3 = extractvalue %struct.__neon_int8x8x3_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int8x8x3_t %tmp2, 1
        %tmp5 = extractvalue %struct.__neon_int8x8x3_t %tmp2, 2
; CHECK: vtbl.8	d16, {d16, d17, d18}, d20 @ encoding: [0xa4,0x0a,0xf0,0xf3]
	%tmp6 = call <8 x i8> @llvm.arm.neon.vtbl3(<8 x i8> %tmp1, <8 x i8> %tmp3, <8 x i8> %tmp4, <8 x i8> %tmp5)
	ret <8 x i8> %tmp6
}

define <8 x i8> @vtbl4(<8 x i8>* %A, %struct.__neon_int8x8x4_t* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load %struct.__neon_int8x8x4_t* %B
        %tmp3 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 1
        %tmp5 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 2
        %tmp6 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 3
; CHECK: vtbl.8	d16, {d16, d17, d18, d19}, d20 @ encoding: [0xa4,0x0b,0xf0,0xf3]
	%tmp7 = call <8 x i8> @llvm.arm.neon.vtbl4(<8 x i8> %tmp1, <8 x i8> %tmp3, <8 x i8> %tmp4, <8 x i8> %tmp5, <8 x i8> %tmp6)
	ret <8 x i8> %tmp7
}

define <8 x i8> @vtbx1(<8 x i8>* %A, <8 x i8>* %B, <8 x i8>* %C) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = load <8 x i8>* %C
; CHECK: vtbx.8	d18, {d16}, d17         @ encoding: [0xe1,0x28,0xf0,0xf3]
	%tmp4 = call <8 x i8> @llvm.arm.neon.vtbx1(<8 x i8> %tmp1, <8 x i8> %tmp2, <8 x i8> %tmp3)
	ret <8 x i8> %tmp4
}

define <8 x i8> @vtbx2(<8 x i8>* %A, %struct.__neon_int8x8x2_t* %B, <8 x i8>* %C) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load %struct.__neon_int8x8x2_t* %B
        %tmp3 = extractvalue %struct.__neon_int8x8x2_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int8x8x2_t %tmp2, 1
	%tmp5 = load <8 x i8>* %C
; CHECK: vtbx.8	d19, {d16, d17}, d18    @ encoding: [0xe2,0x39,0xf0,0xf3]
	%tmp6 = call <8 x i8> @llvm.arm.neon.vtbx2(<8 x i8> %tmp1, <8 x i8> %tmp3, <8 x i8> %tmp4, <8 x i8> %tmp5)
	ret <8 x i8> %tmp6
}

define <8 x i8> @vtbx3(<8 x i8>* %A, %struct.__neon_int8x8x3_t* %B, <8 x i8>* %C) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load %struct.__neon_int8x8x3_t* %B
        %tmp3 = extractvalue %struct.__neon_int8x8x3_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int8x8x3_t %tmp2, 1
        %tmp5 = extractvalue %struct.__neon_int8x8x3_t %tmp2, 2
	%tmp6 = load <8 x i8>* %C
; CHECK: vtbx.8	d20, {d16, d17, d18}, d21 @ encoding: [0xe5,0x4a,0xf0,0xf3]
	%tmp7 = call <8 x i8> @llvm.arm.neon.vtbx3(<8 x i8> %tmp1, <8 x i8> %tmp3, <8 x i8> %tmp4, <8 x i8> %tmp5, <8 x i8> %tmp6)
	ret <8 x i8> %tmp7
}

define <8 x i8> @vtbx4(<8 x i8>* %A, %struct.__neon_int8x8x4_t* %B, <8 x i8>* %C) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load %struct.__neon_int8x8x4_t* %B
        %tmp3 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 0
        %tmp4 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 1
        %tmp5 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 2
        %tmp6 = extractvalue %struct.__neon_int8x8x4_t %tmp2, 3
	%tmp7 = load <8 x i8>* %C
; CHECK: vtbx.8	d20, {d16, d17, d18, d19}, d21 @ encoding: [0xe5,0x4b,0xf0,0xf3]
	%tmp8 = call <8 x i8> @llvm.arm.neon.vtbx4(<8 x i8> %tmp1, <8 x i8> %tmp3, <8 x i8> %tmp4, <8 x i8> %tmp5, <8 x i8> %tmp6, <8 x i8> %tmp7)
	ret <8 x i8> %tmp8
}

declare <8 x i8>  @llvm.arm.neon.vtbl1(<8 x i8>, <8 x i8>) nounwind readnone
declare <8 x i8>  @llvm.arm.neon.vtbl2(<8 x i8>, <8 x i8>, <8 x i8>) nounwind readnone
declare <8 x i8>  @llvm.arm.neon.vtbl3(<8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>) nounwind readnone
declare <8 x i8>  @llvm.arm.neon.vtbl4(<8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>) nounwind readnone

declare <8 x i8>  @llvm.arm.neon.vtbx1(<8 x i8>, <8 x i8>, <8 x i8>) nounwind readnone
declare <8 x i8>  @llvm.arm.neon.vtbx2(<8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>) nounwind readnone
declare <8 x i8>  @llvm.arm.neon.vtbx3(<8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>) nounwind readnone
declare <8 x i8>  @llvm.arm.neon.vtbx4(<8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>) nounwind readnone
