; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon -mattr=+crypto | FileCheck %s
; RUN: not llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon 2>&1 | FileCheck --check-prefix=CHECK-NO-CRYPTO %s

declare <4 x i32> @llvm.arm.neon.sha256su1.v4i32(<4 x i32>, <4 x i32>, <4 x i32>) #1

declare <4 x i32> @llvm.arm.neon.sha256h2.v4i32(<4 x i32>, <4 x i32>, <4 x i32>) #1

declare <4 x i32> @llvm.arm.neon.sha256h.v4i32(<4 x i32>, <4 x i32>, <4 x i32>) #1

declare <4 x i32> @llvm.arm.neon.sha1su0.v4i32(<4 x i32>, <4 x i32>, <4 x i32>) #1

declare <4 x i32> @llvm.aarch64.neon.sha1m(<4 x i32>, <1 x i32>, <4 x i32>) #1

declare <4 x i32> @llvm.aarch64.neon.sha1p(<4 x i32>, <1 x i32>, <4 x i32>) #1

declare <4 x i32> @llvm.aarch64.neon.sha1c(<4 x i32>, <1 x i32>, <4 x i32>) #1

declare <4 x i32> @llvm.arm.neon.sha256su0.v4i32(<4 x i32>, <4 x i32>) #1

declare <4 x i32> @llvm.arm.neon.sha1su1.v4i32(<4 x i32>, <4 x i32>) #1

declare <1 x i32> @llvm.arm.neon.sha1h.v1i32(<1 x i32>) #1

declare <16 x i8> @llvm.arm.neon.aesimc.v16i8(<16 x i8>) #1

declare <16 x i8> @llvm.arm.neon.aesmc.v16i8(<16 x i8>) #1

declare <16 x i8> @llvm.arm.neon.aesd.v16i8(<16 x i8>, <16 x i8>) #1

declare <16 x i8> @llvm.arm.neon.aese.v16i8(<16 x i8>, <16 x i8>) #1

define <16 x i8> @test_vaeseq_u8(<16 x i8> %data, <16 x i8> %key) {
; CHECK: test_vaeseq_u8:
; CHECK: aese {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
; CHECK-NO-CRYPTO: Cannot select: intrinsic %llvm.arm.neon.aese
entry:
  %aese.i = tail call <16 x i8> @llvm.arm.neon.aese.v16i8(<16 x i8> %data, <16 x i8> %key)
  ret <16 x i8> %aese.i
}

define <16 x i8> @test_vaesdq_u8(<16 x i8> %data, <16 x i8> %key) {
; CHECK: test_vaesdq_u8:
; CHECK: aesd {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %aesd.i = tail call <16 x i8> @llvm.arm.neon.aesd.v16i8(<16 x i8> %data, <16 x i8> %key)
  ret <16 x i8> %aesd.i
}

define <16 x i8> @test_vaesmcq_u8(<16 x i8> %data) {
; CHECK: test_vaesmcq_u8:
; CHECK: aesmc {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %aesmc.i = tail call <16 x i8> @llvm.arm.neon.aesmc.v16i8(<16 x i8> %data)
  ret <16 x i8> %aesmc.i
}

define <16 x i8> @test_vaesimcq_u8(<16 x i8> %data) {
; CHECK: test_vaesimcq_u8:
; CHECK: aesimc {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %aesimc.i = tail call <16 x i8> @llvm.arm.neon.aesimc.v16i8(<16 x i8> %data)
  ret <16 x i8> %aesimc.i
}

define i32 @test_vsha1h_u32(i32 %hash_e) {
; CHECK: test_vsha1h_u32:
; CHECK: sha1h {{s[0-9]+}}, {{s[0-9]+}}
entry:
  %sha1h.i = insertelement <1 x i32> undef, i32 %hash_e, i32 0
  %sha1h1.i = tail call <1 x i32> @llvm.arm.neon.sha1h.v1i32(<1 x i32> %sha1h.i)
  %0 = extractelement <1 x i32> %sha1h1.i, i32 0
  ret i32 %0
}

define <4 x i32> @test_vsha1su1q_u32(<4 x i32> %tw0_3, <4 x i32> %w12_15) {
; CHECK: test_vsha1su1q_u32:
; CHECK: sha1su1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %sha1su12.i = tail call <4 x i32> @llvm.arm.neon.sha1su1.v4i32(<4 x i32> %tw0_3, <4 x i32> %w12_15)
  ret <4 x i32> %sha1su12.i
}

define <4 x i32> @test_vsha256su0q_u32(<4 x i32> %w0_3, <4 x i32> %w4_7) {
; CHECK: test_vsha256su0q_u32:
; CHECK: sha256su0 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %sha256su02.i = tail call <4 x i32> @llvm.arm.neon.sha256su0.v4i32(<4 x i32> %w0_3, <4 x i32> %w4_7)
  ret <4 x i32> %sha256su02.i
}

define <4 x i32> @test_vsha1cq_u32(<4 x i32> %hash_abcd, i32 %hash_e, <4 x i32> %wk) {
; CHECK: test_vsha1cq_u32:
; CHECK: sha1c {{q[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.4s
entry:
  %sha1c.i = insertelement <1 x i32> undef, i32 %hash_e, i32 0
  %sha1c1.i = tail call <4 x i32> @llvm.aarch64.neon.sha1c(<4 x i32> %hash_abcd, <1 x i32> %sha1c.i, <4 x i32> %wk)
  ret <4 x i32> %sha1c1.i
}

define <4 x i32> @test_vsha1pq_u32(<4 x i32> %hash_abcd, i32 %hash_e, <4 x i32> %wk) {
; CHECK: test_vsha1pq_u32:
; CHECK: sha1p {{q[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.4s
entry:
  %sha1p.i = insertelement <1 x i32> undef, i32 %hash_e, i32 0
  %sha1p1.i = tail call <4 x i32> @llvm.aarch64.neon.sha1p(<4 x i32> %hash_abcd, <1 x i32> %sha1p.i, <4 x i32> %wk)
  ret <4 x i32> %sha1p1.i
}

define <4 x i32> @test_vsha1mq_u32(<4 x i32> %hash_abcd, i32 %hash_e, <4 x i32> %wk) {
; CHECK: test_vsha1mq_u32:
; CHECK: sha1m {{q[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.4s
entry:
  %sha1m.i = insertelement <1 x i32> undef, i32 %hash_e, i32 0
  %sha1m1.i = tail call <4 x i32> @llvm.aarch64.neon.sha1m(<4 x i32> %hash_abcd, <1 x i32> %sha1m.i, <4 x i32> %wk)
  ret <4 x i32> %sha1m1.i
}

define <4 x i32> @test_vsha1su0q_u32(<4 x i32> %w0_3, <4 x i32> %w4_7, <4 x i32> %w8_11) {
; CHECK: test_vsha1su0q_u32:
; CHECK: sha1su0 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %sha1su03.i = tail call <4 x i32> @llvm.arm.neon.sha1su0.v4i32(<4 x i32> %w0_3, <4 x i32> %w4_7, <4 x i32> %w8_11)
  ret <4 x i32> %sha1su03.i
}

define <4 x i32> @test_vsha256hq_u32(<4 x i32> %hash_abcd, <4 x i32> %hash_efgh, <4 x i32> %wk) {
; CHECK: test_vsha256hq_u32:
; CHECK: sha256h {{q[0-9]+}}, {{q[0-9]+}}, {{v[0-9]+}}.4s
entry:
  %sha256h3.i = tail call <4 x i32> @llvm.arm.neon.sha256h.v4i32(<4 x i32> %hash_abcd, <4 x i32> %hash_efgh, <4 x i32> %wk)
  ret <4 x i32> %sha256h3.i
}

define <4 x i32> @test_vsha256h2q_u32(<4 x i32> %hash_efgh, <4 x i32> %hash_abcd, <4 x i32> %wk) {
; CHECK: test_vsha256h2q_u32:
; CHECK: sha256h2 {{q[0-9]+}}, {{q[0-9]+}}, {{v[0-9]+}}.4s
entry:
  %sha256h23.i = tail call <4 x i32> @llvm.arm.neon.sha256h2.v4i32(<4 x i32> %hash_efgh, <4 x i32> %hash_abcd, <4 x i32> %wk)
  ret <4 x i32> %sha256h23.i
}

define <4 x i32> @test_vsha256su1q_u32(<4 x i32> %tw0_3, <4 x i32> %w8_11, <4 x i32> %w12_15) {
; CHECK: test_vsha256su1q_u32:
; CHECK: sha256su1 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %sha256su13.i = tail call <4 x i32> @llvm.arm.neon.sha256su1.v4i32(<4 x i32> %tw0_3, <4 x i32> %w8_11, <4 x i32> %w12_15)
  ret <4 x i32> %sha256su13.i
}

