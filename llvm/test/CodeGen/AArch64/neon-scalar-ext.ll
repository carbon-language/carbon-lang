; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon < %s | FileCheck %s

define <1 x i64> @test_zext_v1i32_v1i64(<2 x i32> %v) nounwind readnone {
; CHECK-LABEL: test_zext_v1i32_v1i64:
; CHECK: ushll	v0.2d, v0.2s, #0
  %1 = extractelement <2 x i32> %v, i32 0
  %2 = insertelement <1 x i32> undef, i32 %1, i32 0
  %3 = zext <1 x i32> %2 to <1 x i64>
  ret <1 x i64> %3
}

define <1 x i32> @test_zext_v1i16_v1i32(<4 x i16> %v) nounwind readnone {
; CHECK-LABEL: test_zext_v1i16_v1i32:
; CHECK: ushll	v0.4s, v0.4h, #0
  %1 = extractelement <4 x i16> %v, i32 0
  %2 = insertelement <1 x i16> undef, i16 %1, i32 0
  %3 = zext <1 x i16> %2 to <1 x i32>
  ret <1 x i32> %3
}

define <1 x i16> @test_zext_v1i8_v1i16(<8 x i8> %v) nounwind readnone {
; CHECK-LABEL: test_zext_v1i8_v1i16:
; CHECK: ushll	v0.8h, v0.8b, #0
  %1 = extractelement <8 x i8> %v, i32 0
  %2 = insertelement <1 x i8> undef, i8 %1, i32 0
  %3 = zext <1 x i8> %2 to <1 x i16>
  ret <1 x i16> %3
}

define <1 x i32> @test_zext_v1i8_v1i32(<8 x i8> %v) nounwind readnone {
; CHECK-LABEL: test_zext_v1i8_v1i32:
; CHECK: ushll	v0.8h, v0.8b, #0
; CHECK: ushll	v0.4s, v0.4h, #0
  %1 = extractelement <8 x i8> %v, i32 0
  %2 = insertelement <1 x i8> undef, i8 %1, i32 0
  %3 = zext <1 x i8> %2 to <1 x i32>
  ret <1 x i32> %3
}

define <1 x i64> @test_zext_v1i16_v1i64(<4 x i16> %v) nounwind readnone {
; CHECK-LABEL: test_zext_v1i16_v1i64:
; CHECK: dup    h0, v0.h[0]
  %1 = extractelement <4 x i16> %v, i32 0
  %2 = insertelement <1 x i16> undef, i16 %1, i32 0
  %3 = zext <1 x i16> %2 to <1 x i64>
  ret <1 x i64> %3
}

define <1 x i64> @test_zext_v1i8_v1i64(<8 x i8> %v) nounwind readnone {
; CHECK-LABEL: test_zext_v1i8_v1i64:
; CHECK: dup	b0, v0.b[0]
  %1 = extractelement <8 x i8> %v, i32 0
  %2 = insertelement <1 x i8> undef, i8 %1, i32 0
  %3 = zext <1 x i8> %2 to <1 x i64>
  ret <1 x i64> %3
}

define <1 x i64> @test_sext_v1i32_v1i64(<2 x i32> %v) nounwind readnone {
; CHECK-LABEL: test_sext_v1i32_v1i64:
; CHECK: sshll	v0.2d, v0.2s, #0
  %1 = extractelement <2 x i32> %v, i32 0
  %2 = insertelement <1 x i32> undef, i32 %1, i32 0
  %3 = sext <1 x i32> %2 to <1 x i64>
  ret <1 x i64> %3
}

define <1 x i32> @test_sext_v1i16_v1i32(<4 x i16> %v) nounwind readnone {
; CHECK-LABEL: test_sext_v1i16_v1i32:
; CHECK: sshll	v0.4s, v0.4h, #0
  %1 = extractelement <4 x i16> %v, i32 0
  %2 = insertelement <1 x i16> undef, i16 %1, i32 0
  %3 = sext <1 x i16> %2 to <1 x i32>
  ret <1 x i32> %3
}

define <1 x i16> @test_sext_v1i8_v1i16(<8 x i8> %v) nounwind readnone {
; CHECK-LABEL: test_sext_v1i8_v1i16:
; CHECK: sshll	v0.8h, v0.8b, #0
  %1 = extractelement <8 x i8> %v, i32 0
  %2 = insertelement <1 x i8> undef, i8 %1, i32 0
  %3 = sext <1 x i8> %2 to <1 x i16>
  ret <1 x i16> %3
}

define <1 x i32> @test_sext_v1i8_v1i32(<8 x i8> %v) nounwind readnone {
; CHECK-LABEL: test_sext_v1i8_v1i32:
; CHECK: sshll	v0.8h, v0.8b, #0
; CHECK: sshll	v0.4s, v0.4h, #0
  %1 = extractelement <8 x i8> %v, i32 0
  %2 = insertelement <1 x i8> undef, i8 %1, i32 0
  %3 = sext <1 x i8> %2 to <1 x i32>
  ret <1 x i32> %3
}

define <1 x i64> @test_sext_v1i16_v1i64(<4 x i16> %v) nounwind readnone {
; CHECK-LABEL: test_sext_v1i16_v1i64:
; CHECK: sshll	v0.4s, v0.4h, #0
; CHECK: sshll	v0.2d, v0.2s, #0
  %1 = extractelement <4 x i16> %v, i32 0
  %2 = insertelement <1 x i16> undef, i16 %1, i32 0
  %3 = sext <1 x i16> %2 to <1 x i64>
  ret <1 x i64> %3
}

define <1 x i64> @test_sext_v1i8_v1i64(<8 x i8> %v) nounwind readnone {
; CHECK-LABEL: test_sext_v1i8_v1i64:
; CHECK: sshll	v0.8h, v0.8b, #0
; CHECK: sshll	v0.4s, v0.4h, #0
; CHECK: sshll	v0.2d, v0.2s, #0
  %1 = extractelement <8 x i8> %v, i32 0
  %2 = insertelement <1 x i8> undef, i8 %1, i32 0
  %3 = sext <1 x i8> %2 to <1 x i64>
  ret <1 x i64> %3
}
