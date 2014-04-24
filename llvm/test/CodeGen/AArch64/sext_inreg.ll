; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon | FileCheck %s

; arm64: This test contains much that is unique and valuable. Unfortunately the
; bits that are unique aren't valuable and the bits that are valuable aren't
; unique. (weird ABI types vs bog-standard shifting & extensions).

; For formal arguments, we have the following vector type promotion,
; v2i8 is promoted to v2i32(f64)
; v2i16 is promoted to v2i32(f64)
; v4i8 is promoted to v4i16(f64)
; v8i1 is promoted to v8i16(f128)

define <2 x i8> @test_sext_inreg_v2i8i16(<2 x i8> %v1, <2 x i8> %v2) nounwind readnone {
; CHECK-LABEL: test_sext_inreg_v2i8i16
; CHECK: sshll   v0.8h, v0.8b, #0
; CHECK-NEXT: uzp1    v0.8h, v0.8h, v0.8h
; CHECK-NEXT: sshll   v1.8h, v1.8b, #0
; CHECK-NEXT: uzp1    v1.8h, v1.8h, v1.8h
  %1 = sext <2 x i8> %v1 to <2 x i16>
  %2 = sext <2 x i8> %v2 to <2 x i16>
  %3 = shufflevector <2 x i16> %1, <2 x i16> %2, <2 x i32> <i32 0, i32 2>
  %4 = trunc <2 x i16> %3 to <2 x i8>
  ret <2 x i8> %4
}

define <2 x i8> @test_sext_inreg_v2i8i16_2(<2 x i32> %v1, <2 x i32> %v2) nounwind readnone {
; CHECK-LABEL: test_sext_inreg_v2i8i16_2
; CHECK: sshll   v0.8h, v0.8b, #0
; CHECK-NEXT: uzp1    v0.8h, v0.8h, v0.8h
; CHECK-NEXT: sshll   v1.8h, v1.8b, #0
; CHECK-NEXT: uzp1    v1.8h, v1.8h, v1.8h
  %a1 = shl <2 x i32> %v1, <i32 24, i32 24>
  %a2 = ashr <2 x i32> %a1, <i32 24, i32 24>
  %b1 = shl <2 x i32> %v2, <i32 24, i32 24>
  %b2 = ashr <2 x i32> %b1, <i32 24, i32 24>
  %c = shufflevector <2 x i32> %a2, <2 x i32> %b2, <2 x i32> <i32 0, i32 2>
  %d = trunc <2 x i32> %c to <2 x i8>
  ret <2 x i8> %d
}

define <2 x i8> @test_sext_inreg_v2i8i32(<2 x i8> %v1, <2 x i8> %v2) nounwind readnone {
; CHECK-LABEL: test_sext_inreg_v2i8i32
; CHECK: sshll	 v0.8h, v0.8b, #0
; CHECK-NEXT: uzp1    v0.8h, v0.8h, v0.8h
; CHECK-NEXT: sshll	 v1.8h, v1.8b, #0
; CHECK-NEXT: uzp1    v1.8h, v1.8h, v1.8h
  %1 = sext <2 x i8> %v1 to <2 x i32>
  %2 = sext <2 x i8> %v2 to <2 x i32>
  %3 = shufflevector <2 x i32> %1, <2 x i32> %2, <2 x i32> <i32 0, i32 2>
  %4 = trunc <2 x i32> %3 to <2 x i8>
  ret <2 x i8> %4
}

define <2 x i8> @test_sext_inreg_v2i8i64(<2 x i8> %v1, <2 x i8> %v2) nounwind readnone {
; CHECK-LABEL: test_sext_inreg_v2i8i64
; CHECK: ushll   v1.2d, v1.2s, #0
; CHECK: ushll   v0.2d, v0.2s, #0
; CHECK: shl     v0.2d, v0.2d, #56
; CHECK: sshr    v0.2d, v0.2d, #56
; CHECK: shl     v1.2d, v1.2d, #56
; CHECK: sshr    v1.2d, v1.2d, #56
  %1 = sext <2 x i8> %v1 to <2 x i64>
  %2 = sext <2 x i8> %v2 to <2 x i64>
  %3 = shufflevector <2 x i64> %1, <2 x i64> %2, <2 x i32> <i32 0, i32 2>
  %4 = trunc <2 x i64> %3 to <2 x i8>
  ret <2 x i8> %4
}

define <4 x i8> @test_sext_inreg_v4i8i16(<4 x i8> %v1, <4 x i8> %v2) nounwind readnone {
; CHECK-LABEL: test_sext_inreg_v4i8i16
; CHECK: sshll   v0.8h, v0.8b, #0
; CHECK-NEXT: uzp1    v0.8h, v0.8h, v0.8h
; CHECK-NEXT: sshll   v1.8h, v1.8b, #0
; CHECK-NEXT: uzp1    v1.8h, v1.8h, v1.8h
  %1 = sext <4 x i8> %v1 to <4 x i16>
  %2 = sext <4 x i8> %v2 to <4 x i16>
  %3 = shufflevector <4 x i16> %1, <4 x i16> %2, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %4 = trunc <4 x i16> %3 to <4 x i8>
  ret <4 x i8> %4
}

define <4 x i8> @test_sext_inreg_v4i8i16_2(<4 x i16> %v1, <4 x i16> %v2) nounwind readnone {
; CHECK-LABEL: test_sext_inreg_v4i8i16_2
; CHECK: sshll   v0.8h, v0.8b, #0
; CHECK-NEXT: uzp1    v0.8h, v0.8h, v0.8h
; CHECK-NEXT: sshll   v1.8h, v1.8b, #0
; CHECK-NEXT: uzp1    v1.8h, v1.8h, v1.8h
  %a1 = shl <4 x i16> %v1, <i16 8, i16 8, i16 8, i16 8>
  %a2 = ashr <4 x i16> %a1, <i16 8, i16 8, i16 8, i16 8>
  %b1 = shl <4 x i16> %v2, <i16 8, i16 8, i16 8, i16 8>
  %b2 = ashr <4 x i16> %b1, <i16 8, i16 8, i16 8, i16 8>
  %c = shufflevector <4 x i16> %a2, <4 x i16> %b2, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %d = trunc <4 x i16> %c to <4 x i8>
  ret <4 x i8> %d
}

define <4 x i8> @test_sext_inreg_v4i8i32(<4 x i8> %v1, <4 x i8> %v2) nounwind readnone {
; CHECK-LABEL: test_sext_inreg_v4i8i32
; CHECK: ushll   v1.4s, v1.4h, #0
; CHECK: ushll   v0.4s, v0.4h, #0
; CHECK: shl     v0.4s, v0.4s, #24
; CHECK: sshr    v0.4s, v0.4s, #24
; CHECK: shl     v1.4s, v1.4s, #24
; CHECK: sshr    v1.4s, v1.4s, #24
  %1 = sext <4 x i8> %v1 to <4 x i32>
  %2 = sext <4 x i8> %v2 to <4 x i32>
  %3 = shufflevector <4 x i32> %1, <4 x i32> %2, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %4 = trunc <4 x i32> %3 to <4 x i8>
  ret <4 x i8> %4
}

define <8 x i8> @test_sext_inreg_v8i8i16(<8 x i8> %v1, <8 x i8> %v2) nounwind readnone {
; CHECK-LABEL: test_sext_inreg_v8i8i16
; CHECK: sshll   v0.8h, v0.8b, #0
; CHECK: sshll   v1.8h, v1.8b, #0
  %1 = sext <8 x i8> %v1 to <8 x i16>
  %2 = sext <8 x i8> %v2 to <8 x i16>
  %3 = shufflevector <8 x i16> %1, <8 x i16> %2, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  %4 = trunc <8 x i16> %3 to <8 x i8>
  ret <8 x i8> %4
}

define <8 x i1> @test_sext_inreg_v8i1i16(<8 x i1> %v1, <8 x i1> %v2) nounwind readnone {
; CHECK-LABEL: test_sext_inreg_v8i1i16
; CHECK: ushll   v1.8h, v1.8b, #0
; CHECK: ushll   v0.8h, v0.8b, #0
; CHECK: shl     v0.8h, v0.8h, #15
; CHECK: sshr    v0.8h, v0.8h, #15
; CHECK: shl     v1.8h, v1.8h, #15
; CHECK: sshr    v1.8h, v1.8h, #15
  %1 = sext <8 x i1> %v1 to <8 x i16>
  %2 = sext <8 x i1> %v2 to <8 x i16>
  %3 = shufflevector <8 x i16> %1, <8 x i16> %2, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  %4 = trunc <8 x i16> %3 to <8 x i1>
  ret <8 x i1> %4
}

define <2 x i16> @test_sext_inreg_v2i16i32(<2 x i16> %v1, <2 x i16> %v2) nounwind readnone {
; CHECK-LABEL: test_sext_inreg_v2i16i32
; CHECK: sshll   v0.4s, v0.4h, #0
; CHECK-NEXT: uzp1    v0.4s, v0.4s, v0.4s
; CHECK-NEXT: sshll   v1.4s, v1.4h, #0
; CHECK-NEXT: uzp1    v1.4s, v1.4s, v1.4s
  %1 = sext <2 x i16> %v1 to <2 x i32>
  %2 = sext <2 x i16> %v2 to <2 x i32>
  %3 = shufflevector <2 x i32> %1, <2 x i32> %2, <2 x i32> <i32 0, i32 2>
  %4 = trunc <2 x i32> %3 to <2 x i16>
  ret <2 x i16> %4
}

define <2 x i16> @test_sext_inreg_v2i16i32_2(<2 x i32> %v1, <2 x i32> %v2) nounwind readnone {
; CHECK-LABEL: test_sext_inreg_v2i16i32_2
; CHECK: sshll   v0.4s, v0.4h, #0
; CHECK-NEXT: uzp1    v0.4s, v0.4s, v0.4s
; CHECK-NEXT: sshll   v1.4s, v1.4h, #0
; CHECK-NEXT: uzp1    v1.4s, v1.4s, v1.4s
  %a1 = shl <2 x i32> %v1, <i32 16, i32 16>
  %a2 = ashr <2 x i32> %a1, <i32 16, i32 16>
  %b1 = shl <2 x i32> %v2, <i32 16, i32 16>
  %b2 = ashr <2 x i32> %b1, <i32 16, i32 16>
  %c = shufflevector <2 x i32> %a2, <2 x i32> %b2, <2 x i32> <i32 0, i32 2>
  %d = trunc <2 x i32> %c to <2 x i16>
  ret <2 x i16> %d
}

define <2 x i16> @test_sext_inreg_v2i16i64(<2 x i16> %v1, <2 x i16> %v2) nounwind readnone {
; CHECK-LABEL: test_sext_inreg_v2i16i64
; CHECK: ushll   v1.2d, v1.2s, #0
; CHECK: ushll   v0.2d, v0.2s, #0
; CHECK: shl     v0.2d, v0.2d, #48
; CHECK: sshr    v0.2d, v0.2d, #48
; CHECK: shl     v1.2d, v1.2d, #48
; CHECK: sshr    v1.2d, v1.2d, #48
  %1 = sext <2 x i16> %v1 to <2 x i64>
  %2 = sext <2 x i16> %v2 to <2 x i64>
  %3 = shufflevector <2 x i64> %1, <2 x i64> %2, <2 x i32> <i32 0, i32 2>
  %4 = trunc <2 x i64> %3 to <2 x i16>
  ret <2 x i16> %4
}

define <4 x i16> @test_sext_inreg_v4i16i32(<4 x i16> %v1, <4 x i16> %v2) nounwind readnone {
; CHECK-LABEL: test_sext_inreg_v4i16i32
; CHECK: sshll v0.4s, v0.4h, #0
; CHECK: sshll v1.4s, v1.4h, #0
  %1 = sext <4 x i16> %v1 to <4 x i32>
  %2 = sext <4 x i16> %v2 to <4 x i32>
  %3 = shufflevector <4 x i32> %1, <4 x i32> %2, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %4 = trunc <4 x i32> %3 to <4 x i16>
  ret <4 x i16> %4
}

define <2 x i32> @test_sext_inreg_v2i32i64(<2 x i32> %v1, <2 x i32> %v2) nounwind readnone {
; CHECK-LABEL: test_sext_inreg_v2i32i64
; CHECK: sshll v0.2d, v0.2s, #0
; CHECK: sshll v1.2d, v1.2s, #0
  %1 = sext <2 x i32> %v1 to <2 x i64>
  %2 = sext <2 x i32> %v2 to <2 x i64>
  %3 = shufflevector <2 x i64> %1, <2 x i64> %2, <2 x i32> <i32 0, i32 2>
  %4 = trunc <2 x i64> %3 to <2 x i32>
  ret <2 x i32> %4
}

