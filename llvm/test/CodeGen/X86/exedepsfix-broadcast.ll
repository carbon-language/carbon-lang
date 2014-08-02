; RUN: llc -O3 -mtriple=x86_64-apple-macosx -o - < %s -mattr=+avx2 -enable-unsafe-fp-math -mcpu=core2 | FileCheck %s
; Check that the ExeDepsFix pass correctly fixes the domain for broadcast instructions.
; <rdar://problem/16354675>

; CHECK-LABEL: ExeDepsFix_broadcastss
; CHECK: broadcastss
; CHECK: vandps
; CHECK: vmaxps
; CHECK: ret
define <4 x float> @ExeDepsFix_broadcastss(<4 x float> %arg, <4 x float> %arg2) {
  %bitcast = bitcast <4 x float> %arg to <4 x i32>
  %and = and <4 x i32> %bitcast, <i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647>
  %floatcast = bitcast <4 x i32> %and to <4 x float>
  %max_is_x = fcmp oge <4 x float> %floatcast, %arg2
  %max = select <4 x i1> %max_is_x, <4 x float> %floatcast, <4 x float> %arg2
  ret <4 x float> %max
}

; CHECK-LABEL: ExeDepsFix_broadcastss256
; CHECK: broadcastss
; CHECK: vandps
; CHECK: vmaxps
; CHECK: ret
define <8 x float> @ExeDepsFix_broadcastss256(<8 x float> %arg, <8 x float> %arg2) {
  %bitcast = bitcast <8 x float> %arg to <8 x i32>
  %and = and <8 x i32> %bitcast, <i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647>
  %floatcast = bitcast <8 x i32> %and to <8 x float>
  %max_is_x = fcmp oge <8 x float> %floatcast, %arg2
  %max = select <8 x i1> %max_is_x, <8 x float> %floatcast, <8 x float> %arg2
  ret <8 x float> %max
}


; CHECK-LABEL: ExeDepsFix_broadcastss_inreg
; CHECK: broadcastss
; CHECK: vandps
; CHECK: vmaxps
; CHECK: ret
define <4 x float> @ExeDepsFix_broadcastss_inreg(<4 x float> %arg, <4 x float> %arg2, i32 %broadcastvalue) {
  %bitcast = bitcast <4 x float> %arg to <4 x i32>
  %in = insertelement <4 x i32> undef, i32 %broadcastvalue, i32 0
  %mask = shufflevector <4 x i32> %in, <4 x i32> undef, <4 x i32> zeroinitializer
  %and = and <4 x i32> %bitcast, %mask
  %floatcast = bitcast <4 x i32> %and to <4 x float>
  %max_is_x = fcmp oge <4 x float> %floatcast, %arg2
  %max = select <4 x i1> %max_is_x, <4 x float> %floatcast, <4 x float> %arg2
  ret <4 x float> %max
}

; CHECK-LABEL: ExeDepsFix_broadcastss256_inreg
; CHECK: broadcastss
; CHECK: vandps
; CHECK: vmaxps
; CHECK: ret
define <8 x float> @ExeDepsFix_broadcastss256_inreg(<8 x float> %arg, <8 x float> %arg2, i32 %broadcastvalue) {
  %bitcast = bitcast <8 x float> %arg to <8 x i32>
  %in = insertelement <8 x i32> undef, i32 %broadcastvalue, i32 0
  %mask = shufflevector <8 x i32> %in, <8 x i32> undef, <8 x i32> zeroinitializer
  %and = and <8 x i32> %bitcast, %mask
  %floatcast = bitcast <8 x i32> %and to <8 x float>
  %max_is_x = fcmp oge <8 x float> %floatcast, %arg2
  %max = select <8 x i1> %max_is_x, <8 x float> %floatcast, <8 x float> %arg2
  ret <8 x float> %max
}

; CHECK-LABEL: ExeDepsFix_broadcastsd
; In that case the broadcast is directly folded into vandpd.
; CHECK: vandpd
; CHECK: vmaxpd
; CHECK:ret
define <2 x double> @ExeDepsFix_broadcastsd(<2 x double> %arg, <2 x double> %arg2) {
  %bitcast = bitcast <2 x double> %arg to <2 x i64>
  %and = and <2 x i64> %bitcast, <i64 2147483647, i64 2147483647>
  %floatcast = bitcast <2 x i64> %and to <2 x double>
  %max_is_x = fcmp oge <2 x double> %floatcast, %arg2
  %max = select <2 x i1> %max_is_x, <2 x double> %floatcast, <2 x double> %arg2
  ret <2 x double> %max
}

; CHECK-LABEL: ExeDepsFix_broadcastsd256
; CHECK: broadcastsd
; CHECK: vandpd
; CHECK: vmaxpd
; CHECK: ret
define <4 x double> @ExeDepsFix_broadcastsd256(<4 x double> %arg, <4 x double> %arg2) {
  %bitcast = bitcast <4 x double> %arg to <4 x i64>
  %and = and <4 x i64> %bitcast, <i64 2147483647, i64 2147483647, i64 2147483647, i64 2147483647>
  %floatcast = bitcast <4 x i64> %and to <4 x double>
  %max_is_x = fcmp oge <4 x double> %floatcast, %arg2
  %max = select <4 x i1> %max_is_x, <4 x double> %floatcast, <4 x double> %arg2
  ret <4 x double> %max
}


; CHECK-LABEL: ExeDepsFix_broadcastsd_inreg
; ExeDepsFix works top down, thus it coalesces vpunpcklqdq domain with
; vpand and there is nothing more you can do to match vmaxpd.
; CHECK: vpunpcklqdq
; CHECK: vpand
; CHECK: vmaxpd
; CHECK: ret
define <2 x double> @ExeDepsFix_broadcastsd_inreg(<2 x double> %arg, <2 x double> %arg2, i64 %broadcastvalue) {
  %bitcast = bitcast <2 x double> %arg to <2 x i64>
  %in = insertelement <2 x i64> undef, i64 %broadcastvalue, i32 0
  %mask = shufflevector <2 x i64> %in, <2 x i64> undef, <2 x i32> zeroinitializer
  %and = and <2 x i64> %bitcast, %mask
  %floatcast = bitcast <2 x i64> %and to <2 x double>
  %max_is_x = fcmp oge <2 x double> %floatcast, %arg2
  %max = select <2 x i1> %max_is_x, <2 x double> %floatcast, <2 x double> %arg2
  ret <2 x double> %max
}

; CHECK-LABEL: ExeDepsFix_broadcastsd256_inreg
; CHECK: broadcastsd
; CHECK: vandpd
; CHECK: vmaxpd
; CHECK: ret
define <4 x double> @ExeDepsFix_broadcastsd256_inreg(<4 x double> %arg, <4 x double> %arg2, i64 %broadcastvalue) {
  %bitcast = bitcast <4 x double> %arg to <4 x i64>
  %in = insertelement <4 x i64> undef, i64 %broadcastvalue, i32 0
  %mask = shufflevector <4 x i64> %in, <4 x i64> undef, <4 x i32> zeroinitializer
  %and = and <4 x i64> %bitcast, %mask
  %floatcast = bitcast <4 x i64> %and to <4 x double>
  %max_is_x = fcmp oge <4 x double> %floatcast, %arg2
  %max = select <4 x i1> %max_is_x, <4 x double> %floatcast, <4 x double> %arg2
  ret <4 x double> %max
}

