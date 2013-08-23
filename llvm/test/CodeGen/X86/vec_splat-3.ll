; RUN: llc <%s -march=x86 -mcpu=penryn -mattr=sse4.1 | FileCheck %s

; Splat test for v8i16
define <8 x i16> @shuf_8i16_0(<8 x i16> %T0, <8 x i16> %T1) nounwind readnone {
	%tmp6 = shufflevector <8 x i16> %T0, <8 x i16> %T1, <8 x i32> <i32 0, i32 undef, i32 undef, i32 0, i32 undef, i32 undef, i32 undef, i32 undef>
	ret <8 x i16> %tmp6

; CHECK-LABEL: shuf_8i16_0:
; CHECK: pshuflw $0
}

define <8 x i16> @shuf_8i16_1(<8 x i16> %T0, <8 x i16> %T1) nounwind readnone {
	%tmp6 = shufflevector <8 x i16> %T0, <8 x i16> %T1, <8 x i32> <i32 1, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
	ret <8 x i16> %tmp6

; CHECK-LABEL: shuf_8i16_1:
; CHECK: pshuflw $5
}

define <8 x i16> @shuf_8i16_2(<8 x i16> %T0, <8 x i16> %T1) nounwind readnone {
	%tmp6 = shufflevector <8 x i16> %T0, <8 x i16> %T1, <8 x i32> <i32 2, i32 undef, i32 undef, i32 2, i32 undef, i32 2, i32 undef, i32 undef>
	ret <8 x i16> %tmp6

; CHECK-LABEL: shuf_8i16_2:
; CHECK: punpcklwd
; CHECK-NEXT: pshufd $-86
}

define <8 x i16> @shuf_8i16_3(<8 x i16> %T0, <8 x i16> %T1) nounwind readnone {
	%tmp6 = shufflevector <8 x i16> %T0, <8 x i16> %T1, <8 x i32> <i32 3, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
	ret <8 x i16> %tmp6

; CHECK-LABEL: shuf_8i16_3:
; CHECK: pshuflw $15
}

define <8 x i16> @shuf_8i16_4(<8 x i16> %T0, <8 x i16> %T1) nounwind readnone {
	%tmp6 = shufflevector <8 x i16> %T0, <8 x i16> %T1, <8 x i32> <i32 4, i32 undef, i32 undef, i32 undef, i32 4, i32 undef, i32 undef, i32 undef>
	ret <8 x i16> %tmp6

; CHECK-LABEL: shuf_8i16_4:
; CHECK: movhlps
}

define <8 x i16> @shuf_8i16_5(<8 x i16> %T0, <8 x i16> %T1) nounwind readnone {
	%tmp6 = shufflevector <8 x i16> %T0, <8 x i16> %T1, <8 x i32> <i32 5, i32 undef, i32 undef, i32 5, i32 undef, i32 undef, i32 undef, i32 undef>
	ret <8 x i16> %tmp6

; CHECK-LABEL: shuf_8i16_5:
; CHECK: punpckhwd
; CHECK-NEXT: pshufd $85
}

define <8 x i16> @shuf_8i16_6(<8 x i16> %T0, <8 x i16> %T1) nounwind readnone {
	%tmp6 = shufflevector <8 x i16> %T0, <8 x i16> %T1, <8 x i32> <i32 6, i32 6, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
	ret <8 x i16> %tmp6

; CHECK-LABEL: shuf_8i16_6:
; CHECK: punpckhwd
; CHECK-NEXT: pshufd $-86
}

define <8 x i16> @shuf_8i16_7(<8 x i16> %T0, <8 x i16> %T1) nounwind readnone {
	%tmp6 = shufflevector <8 x i16> %T0, <8 x i16> %T1, <8 x i32> <i32 7, i32 undef, i32 undef, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
	ret <8 x i16> %tmp6

; CHECK-LABEL: shuf_8i16_7:
; CHECK: punpckhwd
; CHECK-NEXT: pshufd $-1
}

; Splat test for v16i8
define <16 x i8> @shuf_16i8_8(<16 x i8> %T0, <16 x i8> %T1) nounwind readnone {
	%tmp6 = shufflevector <16 x i8> %T0, <16 x i8> %T1, <16 x i32> <i32 0, i32 undef, i32 undef, i32 0, i32 undef, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
	ret <16 x i8> %tmp6

; CHECK-LABEL: shuf_16i8_8:
; CHECK: punpcklbw
; CHECK-NEXT: punpcklbw
; CHECK-NEXT: pshufd $0
}

define <16 x i8> @shuf_16i8_9(<16 x i8> %T0, <16 x i8> %T1) nounwind readnone {
	%tmp6 = shufflevector <16 x i8> %T0, <16 x i8> %T1, <16 x i32> <i32 1, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef >
	ret <16 x i8> %tmp6

; CHECK-LABEL: shuf_16i8_9:
; CHECK: punpcklbw
; CHECK-NEXT: punpcklbw
; CHECK-NEXT: pshufd $85
}

define <16 x i8> @shuf_16i8_10(<16 x i8> %T0, <16 x i8> %T1) nounwind readnone {
	%tmp6 = shufflevector <16 x i8> %T0, <16 x i8> %T1, <16 x i32> <i32 2, i32 undef, i32 undef, i32 2, i32 undef, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
	ret <16 x i8> %tmp6

; CHECK-LABEL: shuf_16i8_10:
; CHECK: punpcklbw
; CHECK-NEXT: punpcklbw
; CHECK-NEXT: pshufd $-86
}

define <16 x i8> @shuf_16i8_11(<16 x i8> %T0, <16 x i8> %T1) nounwind readnone {
	%tmp6 = shufflevector <16 x i8> %T0, <16 x i8> %T1, <16 x i32> <i32 3, i32 undef, i32 undef, i32 3, i32 undef, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
	ret <16 x i8> %tmp6

; CHECK-LABEL: shuf_16i8_11:
; CHECK: punpcklbw
; CHECK-NEXT: punpcklbw
; CHECK-NEXT: pshufd $-1
}


define <16 x i8> @shuf_16i8_12(<16 x i8> %T0, <16 x i8> %T1) nounwind readnone {
	%tmp6 = shufflevector <16 x i8> %T0, <16 x i8> %T1, <16 x i32> <i32 4, i32 undef, i32 undef, i32 undef, i32 4, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef >
	ret <16 x i8> %tmp6

; CHECK-LABEL: shuf_16i8_12:
; CHECK: pshufd $5
}

define <16 x i8> @shuf_16i8_13(<16 x i8> %T0, <16 x i8> %T1) nounwind readnone {
	%tmp6 = shufflevector <16 x i8> %T0, <16 x i8> %T1, <16 x i32> <i32 5, i32 undef, i32 undef, i32 5, i32 undef, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5>
	ret <16 x i8> %tmp6

; CHECK-LABEL: shuf_16i8_13:
; CHECK: punpcklbw
; CHECK-NEXT: punpckhbw
; CHECK-NEXT: pshufd $85
}

define <16 x i8> @shuf_16i8_14(<16 x i8> %T0, <16 x i8> %T1) nounwind readnone {
	%tmp6 = shufflevector <16 x i8> %T0, <16 x i8> %T1, <16 x i32> <i32 6, i32 undef, i32 undef, i32 6, i32 undef, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6>
	ret <16 x i8> %tmp6

; CHECK-LABEL: shuf_16i8_14:
; CHECK: punpcklbw
; CHECK-NEXT: punpckhbw
; CHECK-NEXT: pshufd $-86
}

define <16 x i8> @shuf_16i8_15(<16 x i8> %T0, <16 x i8> %T1) nounwind readnone {
	%tmp6 = shufflevector <16 x i8> %T0, <16 x i8> %T1, <16 x i32> <i32 7, i32 undef, i32 undef, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef >
	ret <16 x i8> %tmp6

; CHECK-LABEL: shuf_16i8_15:
; CHECK: punpcklbw
; CHECK-NEXT: punpckhbw
; CHECK-NEXT: pshufd $-1
}

define <16 x i8> @shuf_16i8_16(<16 x i8> %T0, <16 x i8> %T1) nounwind readnone {
	%tmp6 = shufflevector <16 x i8> %T0, <16 x i8> %T1, <16 x i32> <i32 8, i32 undef, i32 undef, i32 8, i32 undef, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8>
	ret <16 x i8> %tmp6

; CHECK-LABEL: shuf_16i8_16:
; CHECK: punpckhbw
; CHECK-NEXT: punpcklbw
; CHECK-NEXT: pshufd $0
}

define <16 x i8> @shuf_16i8_17(<16 x i8> %T0, <16 x i8> %T1) nounwind readnone {
	%tmp6 = shufflevector <16 x i8> %T0, <16 x i8> %T1, <16 x i32> <i32 9, i32 undef, i32 undef, i32 9, i32 undef, i32 9, i32 9, i32 9, i32 9, i32 9, i32 9, i32 9, i32 9, i32 9, i32 9, i32 9>
	ret <16 x i8> %tmp6

; CHECK-LABEL: shuf_16i8_17:
; CHECK: punpckhbw
; CHECK-NEXT: punpcklbw
; CHECK-NEXT: pshufd $85
}

define <16 x i8> @shuf_16i8_18(<16 x i8> %T0, <16 x i8> %T1) nounwind readnone {
	%tmp6 = shufflevector <16 x i8> %T0, <16 x i8> %T1, <16 x i32> <i32 10, i32 undef, i32 undef, i32 10, i32 undef, i32 10, i32 10, i32 10, i32 10, i32 10, i32 10, i32 10, i32 10, i32 10, i32 10, i32 10>
	ret <16 x i8> %tmp6

; CHECK-LABEL: shuf_16i8_18:
; CHECK: punpckhbw
; CHECK-NEXT: punpcklbw
; CHECK-NEXT: pshufd $-86
}

define <16 x i8> @shuf_16i8_19(<16 x i8> %T0, <16 x i8> %T1) nounwind readnone {
	%tmp6 = shufflevector <16 x i8> %T0, <16 x i8> %T1, <16 x i32> <i32 11, i32 undef, i32 undef, i32 11, i32 undef, i32 11, i32 11, i32 11, i32 11, i32 11, i32 11, i32 11, i32 11, i32 11, i32 11, i32 11>
	ret <16 x i8> %tmp6

; CHECK-LABEL: shuf_16i8_19:
; CHECK: punpckhbw
; CHECK-NEXT: punpcklbw
; CHECK-NEXT: pshufd $-1
}

define <16 x i8> @shuf_16i8_20(<16 x i8> %T0, <16 x i8> %T1) nounwind readnone {
	%tmp6 = shufflevector <16 x i8> %T0, <16 x i8> %T1, <16 x i32> <i32 12, i32 undef, i32 undef, i32 12, i32 undef, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12>
	ret <16 x i8> %tmp6

; CHECK-LABEL: shuf_16i8_20:
; CHECK: punpckhbw
; CHECK-NEXT: punpckhbw
; CHECK-NEXT: pshufd $0
}

define <16 x i8> @shuf_16i8_21(<16 x i8> %T0, <16 x i8> %T1) nounwind readnone {
	%tmp6 = shufflevector <16 x i8> %T0, <16 x i8> %T1, <16 x i32> <i32 13, i32 undef, i32 undef, i32 13, i32 undef, i32 13, i32 13, i32 13, i32 13, i32 13, i32 13, i32 13, i32 13, i32 13, i32 13, i32 13>
	ret <16 x i8> %tmp6

; CHECK-LABEL: shuf_16i8_21:
; CHECK: punpckhbw
; CHECK-NEXT: punpckhbw
; CHECK-NEXT: pshufd $85
}

define <16 x i8> @shuf_16i8_22(<16 x i8> %T0, <16 x i8> %T1) nounwind readnone {
	%tmp6 = shufflevector <16 x i8> %T0, <16 x i8> %T1, <16 x i32> <i32 14, i32 undef, i32 undef, i32 14, i32 undef, i32 14, i32 14, i32 14, i32 14, i32 14, i32 14, i32 14, i32 14, i32 14, i32 14, i32 14>
	ret <16 x i8> %tmp6

; CHECK-LABEL: shuf_16i8_22:
; CHECK: punpckhbw
; CHECK-NEXT: punpckhbw
; CHECK-NEXT: pshufd $-86
}

define <16 x i8> @shuf_16i8_23(<16 x i8> %T0, <16 x i8> %T1) nounwind readnone {
	%tmp6 = shufflevector <16 x i8> %T0, <16 x i8> %T1, <16 x i32> <i32 15, i32 undef, i32 undef, i32 15, i32 undef, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15>
	ret <16 x i8> %tmp6

; CHECK-LABEL: shuf_16i8_23:
; CHECK: punpckhbw
; CHECK-NEXT: punpckhbw
; CHECK-NEXT: pshufd $-1
}
