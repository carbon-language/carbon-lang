; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl | FileCheck %s

; CHECK-LABEL: trunc_16x32_to_16x8
; CHECK: vpmovdb
; CHECK: ret
define <16 x i8> @trunc_16x32_to_16x8(<16 x i32> %i) nounwind readnone {
  %x = trunc <16 x i32> %i to <16 x i8>
  ret <16 x i8> %x
}

; CHECK-LABEL: trunc_8x64_to_8x16
; CHECK: vpmovqw
; CHECK: ret
define <8 x i16> @trunc_8x64_to_8x16(<8 x i64> %i) nounwind readnone {
  %x = trunc <8 x i64> %i to <8 x i16>
  ret <8 x i16> %x
}


; CHECK-LABEL: zext_16x8_to_16x32
; CHECK: vpmovzxbd {{.*}}%zmm
; CHECK: ret
define <16 x i32> @zext_16x8_to_16x32(<16 x i8> %i) nounwind readnone {
  %x = zext <16 x i8> %i to <16 x i32>
  ret <16 x i32> %x
}

; CHECK-LABEL: sext_16x8_to_16x32
; CHECK: vpmovsxbd {{.*}}%zmm
; CHECK: ret
define <16 x i32> @sext_16x8_to_16x32(<16 x i8> %i) nounwind readnone {
  %x = sext <16 x i8> %i to <16 x i32>
  ret <16 x i32> %x
}


; CHECK-LABEL: zext_16x16_to_16x32
; CHECK: vpmovzxwd {{.*}}%zmm
; CHECK: ret
define <16 x i32> @zext_16x16_to_16x32(<16 x i16> %i) nounwind readnone {
  %x = zext <16 x i16> %i to <16 x i32>
  ret <16 x i32> %x
}

; CHECK-LABEL: zext_8x16_to_8x64
; CHECK: vpmovzxwq
; CHECK: ret
define <8 x i64> @zext_8x16_to_8x64(<8 x i16> %i) nounwind readnone {
  %x = zext <8 x i16> %i to <8 x i64>
  ret <8 x i64> %x
}

;CHECK-LABEL: fptrunc_test
;CHECK: vcvtpd2ps {{.*}}%zmm
;CHECK: ret
define <8 x float> @fptrunc_test(<8 x double> %a) nounwind readnone {
  %b = fptrunc <8 x double> %a to <8 x float>
  ret <8 x float> %b
}

;CHECK-LABEL: fpext_test
;CHECK: vcvtps2pd {{.*}}%zmm
;CHECK: ret
define <8 x double> @fpext_test(<8 x float> %a) nounwind readnone {
  %b = fpext <8 x float> %a to <8 x double>
  ret <8 x double> %b
}

; CHECK-LABEL: zext_16i1_to_16xi32
; CHECK: vpbroadcastd LCP{{.*}}(%rip), %zmm0 {%k1} {z}
; CHECK: ret
define   <16 x i32> @zext_16i1_to_16xi32(i16 %b) {
  %a = bitcast i16 %b to <16 x i1>
  %c = zext <16 x i1> %a to <16 x i32>
  ret <16 x i32> %c
}

; CHECK-LABEL: zext_8i1_to_8xi64
; CHECK: vpbroadcastq LCP{{.*}}(%rip), %zmm0 {%k1} {z}
; CHECK: ret
define   <8 x i64> @zext_8i1_to_8xi64(i8 %b) {
  %a = bitcast i8 %b to <8 x i1>
  %c = zext <8 x i1> %a to <8 x i64>
  ret <8 x i64> %c
}

; CHECK-LABEL: trunc_16i8_to_16i1
; CHECK: vpmovsxbd
; CHECK: vpandd
; CHECK: vptestmd
; CHECK: ret
define i16 @trunc_16i8_to_16i1(<16 x i8> %a) {
  %mask_b = trunc <16 x i8>%a to <16 x i1>
  %mask = bitcast <16 x i1> %mask_b to i16
  ret i16 %mask
}

; CHECK-LABEL: trunc_16i32_to_16i1
; CHECK: vpandd
; CHECK: vptestmd
; CHECK: ret
define i16 @trunc_16i32_to_16i1(<16 x i32> %a) {
  %mask_b = trunc <16 x i32>%a to <16 x i1>
  %mask = bitcast <16 x i1> %mask_b to i16
  ret i16 %mask
}

; CHECK-LABEL: trunc_8i16_to_8i1
; CHECK: vpmovsxwq
; CHECK: vpandq LCP{{.*}}(%rip){1to8}
; CHECK: vptestmq
; CHECK: ret
define i8 @trunc_8i16_to_8i1(<8 x i16> %a) {
  %mask_b = trunc <8 x i16>%a to <8 x i1>
  %mask = bitcast <8 x i1> %mask_b to i8
  ret i8 %mask
}

; CHECK-LABEL: sext_8i1_8i32
; CHECK: vpbroadcastq  LCP{{.*}}(%rip), %zmm0 {%k1} {z}
; CHECK: ret
define <8 x i32> @sext_8i1_8i32(<8 x i32> %a1, <8 x i32> %a2) nounwind {
  %x = icmp slt <8 x i32> %a1, %a2
  %x1 = xor <8 x i1>%x, <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>
  %y = sext <8 x i1> %x1 to <8 x i32>
  ret <8 x i32> %y
}

; CHECK-LABEL: trunc_v16i32_to_v16i16
; CHECK: vpmovdw
; CHECK: ret
define <16 x i16> @trunc_v16i32_to_v16i16(<16 x i32> %x) {
  %1 = trunc <16 x i32> %x to <16 x i16>
  ret <16 x i16> %1
}

; CHECK-LABEL: trunc_i32_to_i1
; CHECK: andl
; CHECK: kmov
; CHECK: kortest
; CKECK: orl
; CHECK: ret
define i16 @trunc_i32_to_i1(i32 %a) {
  %a_i = trunc i32 %a to i1
  %maskv = insertelement <16 x i1> <i1 true, i1 false, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, i1 %a_i, i32 0
  %res = bitcast <16 x i1> %maskv to i16
  ret i16 %res
}
