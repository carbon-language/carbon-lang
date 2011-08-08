; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

;;; Shift left
; CHECK: vpslld
; CHECK: vpslld
define <8 x i32> @vshift00(<8 x i32> %a) nounwind readnone {
  %s = shl <8 x i32> %a, <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32
2>
  ret <8 x i32> %s
}

; CHECK: vpsllw
; CHECK: vpsllw
define <16 x i16> @vshift01(<16 x i16> %a) nounwind readnone {
  %s = shl <16 x i16> %a, <i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2>
  ret <16 x i16> %s
}

; CHECK: vpsllq
; CHECK: vpsllq
define <4 x i64> @vshift02(<4 x i64> %a) nounwind readnone {
  %s = shl <4 x i64> %a, <i64 2, i64 2, i64 2, i64 2>
  ret <4 x i64> %s
}

;;; Logical Shift right
; CHECK: vpsrld
; CHECK: vpsrld
define <8 x i32> @vshift03(<8 x i32> %a) nounwind readnone {
  %s = lshr <8 x i32> %a, <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32
2>
  ret <8 x i32> %s
}

; CHECK: vpsrlw
; CHECK: vpsrlw
define <16 x i16> @vshift04(<16 x i16> %a) nounwind readnone {
  %s = lshr <16 x i16> %a, <i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2>
  ret <16 x i16> %s
}

; CHECK: vpsrlq
; CHECK: vpsrlq
define <4 x i64> @vshift05(<4 x i64> %a) nounwind readnone {
  %s = lshr <4 x i64> %a, <i64 2, i64 2, i64 2, i64 2>
  ret <4 x i64> %s
}

;;; Arithmetic Shift right
; CHECK: vpsrad
; CHECK: vpsrad
define <8 x i32> @vshift06(<8 x i32> %a) nounwind readnone {
  %s = ashr <8 x i32> %a, <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32
2>
  ret <8 x i32> %s
}

; CHECK: vpsraw
; CHECK: vpsraw
define <16 x i16> @vshift07(<16 x i16> %a) nounwind readnone {
  %s = ashr <16 x i16> %a, <i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2>
  ret <16 x i16> %s
}

