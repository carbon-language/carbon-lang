; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=core-avx2 -mattr=+avx2 | FileCheck %s

; CHECK: vpandn
; CHECK: vpandn  %ymm
; CHECK: ret
define <4 x i64> @vpandn(<4 x i64> %a, <4 x i64> %b) nounwind uwtable readnone ssp {
entry:
  ; Force the execution domain with an add.
  %a2 = add <4 x i64> %a, <i64 1, i64 1, i64 1, i64 1>
  %y = xor <4 x i64> %a2, <i64 -1, i64 -1, i64 -1, i64 -1>
  %x = and <4 x i64> %a, %y
  ret <4 x i64> %x
}

; CHECK: vpand
; CHECK: vpand %ymm
; CHECK: ret
define <4 x i64> @vpand(<4 x i64> %a, <4 x i64> %b) nounwind uwtable readnone ssp {
entry:
  ; Force the execution domain with an add.
  %a2 = add <4 x i64> %a, <i64 1, i64 1, i64 1, i64 1>
  %x = and <4 x i64> %a2, %b
  ret <4 x i64> %x
}

; CHECK: vpor
; CHECK: vpor %ymm
; CHECK: ret
define <4 x i64> @vpor(<4 x i64> %a, <4 x i64> %b) nounwind uwtable readnone ssp {
entry:
  ; Force the execution domain with an add.
  %a2 = add <4 x i64> %a, <i64 1, i64 1, i64 1, i64 1>
  %x = or <4 x i64> %a2, %b
  ret <4 x i64> %x
}

; CHECK: vpxor
; CHECK: vpxor %ymm
; CHECK: ret
define <4 x i64> @vpxor(<4 x i64> %a, <4 x i64> %b) nounwind uwtable readnone ssp {
entry:
  ; Force the execution domain with an add.
  %a2 = add <4 x i64> %a, <i64 1, i64 1, i64 1, i64 1>
  %x = xor <4 x i64> %a2, %b
  ret <4 x i64> %x
}

; CHECK: vpblendvb
; CHECK: vpblendvb %ymm
; CHECK: ret
define <32 x i8> @vpblendvb(<32 x i8> %x, <32 x i8> %y) {
  %min_is_x = icmp ult <32 x i8> %x, %y
  %min = select <32 x i1> %min_is_x, <32 x i8> %x, <32 x i8> %y
  ret <32 x i8> %min
}


; CHECK: variable_shl0
; CHECK: psllvd
; CHECK: ret
define <4 x i32> @variable_shl0(<4 x i32> %x, <4 x i32> %y) {
  %k = shl <4 x i32> %x, %y
  ret <4 x i32> %k
}
; CHECK: variable_shl1
; CHECK: psllvd
; CHECK: ret
define <8 x i32> @variable_shl1(<8 x i32> %x, <8 x i32> %y) {
  %k = shl <8 x i32> %x, %y
  ret <8 x i32> %k
}
; CHECK: variable_shl2
; CHECK: psllvq
; CHECK: ret
define <2 x i64> @variable_shl2(<2 x i64> %x, <2 x i64> %y) {
  %k = shl <2 x i64> %x, %y
  ret <2 x i64> %k
}
; CHECK: variable_shl3
; CHECK: psllvq
; CHECK: ret
define <4 x i64> @variable_shl3(<4 x i64> %x, <4 x i64> %y) {
  %k = shl <4 x i64> %x, %y
  ret <4 x i64> %k
}
; CHECK: variable_srl0
; CHECK: psrlvd
; CHECK: ret
define <4 x i32> @variable_srl0(<4 x i32> %x, <4 x i32> %y) {
  %k = lshr <4 x i32> %x, %y
  ret <4 x i32> %k
}
; CHECK: variable_srl1
; CHECK: psrlvd
; CHECK: ret
define <8 x i32> @variable_srl1(<8 x i32> %x, <8 x i32> %y) {
  %k = lshr <8 x i32> %x, %y
  ret <8 x i32> %k
}
; CHECK: variable_srl2
; CHECK: psrlvq
; CHECK: ret
define <2 x i64> @variable_srl2(<2 x i64> %x, <2 x i64> %y) {
  %k = lshr <2 x i64> %x, %y
  ret <2 x i64> %k
}
; CHECK: variable_srl3
; CHECK: psrlvq
; CHECK: ret
define <4 x i64> @variable_srl3(<4 x i64> %x, <4 x i64> %y) {
  %k = lshr <4 x i64> %x, %y
  ret <4 x i64> %k
}

; CHECK: variable_sra0
; CHECK: psravd
; CHECK: ret
define <4 x i32> @variable_sra0(<4 x i32> %x, <4 x i32> %y) {
  %k = ashr <4 x i32> %x, %y
  ret <4 x i32> %k
}
; CHECK: variable_sra1
; CHECK: psravd
; CHECK: ret
define <8 x i32> @variable_sra1(<8 x i32> %x, <8 x i32> %y) {
  %k = ashr <8 x i32> %x, %y
  ret <8 x i32> %k
}
