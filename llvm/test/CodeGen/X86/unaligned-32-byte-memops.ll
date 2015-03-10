; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx | FileCheck %s --check-prefix=SANDYB --check-prefix=CHECK
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=core-avx-i | FileCheck %s --check-prefix=SANDYB --check-prefix=CHECK
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=btver2 | FileCheck %s --check-prefix=BTVER2 --check-prefix=CHECK
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=core-avx2 | FileCheck %s --check-prefix=HASWELL --check-prefix=CHECK

; On Sandy Bridge or Ivy Bridge, we should not generate an unaligned 32-byte load
; because that is slower than two 16-byte loads. 
; Other AVX-capable chips don't have that problem.

define <8 x float> @load32bytes(<8 x float>* %Ap) {
  ; CHECK-LABEL: load32bytes

  ; SANDYB: vmovaps
  ; SANDYB: vinsertf128
  ; SANDYB: retq

  ; BTVER2: vmovups
  ; BTVER2: retq

  ; HASWELL: vmovups
  ; HASWELL: retq

  %A = load <8 x float>, <8 x float>* %Ap, align 16
  ret <8 x float> %A
}

; On Sandy Bridge or Ivy Bridge, we should not generate an unaligned 32-byte store
; because that is slowerthan two 16-byte stores. 
; Other AVX-capable chips don't have that problem.

define void @store32bytes(<8 x float> %A, <8 x float>* %P) {
  ; CHECK-LABEL: store32bytes

  ; SANDYB: vextractf128
  ; SANDYB: vmovaps
  ; SANDYB: retq

  ; BTVER2: vmovups
  ; BTVER2: retq

  ; HASWELL: vmovups
  ; HASWELL: retq

  store <8 x float> %A, <8 x float>* %P, align 16
  ret void
}

; Merge two consecutive 16-byte subvector loads into a single 32-byte load
; if it's faster.

define <8 x float> @combine_16_byte_loads_no_intrinsic(<4 x float>* %ptr) {
  ; CHECK-LABEL: combine_16_byte_loads_no_intrinsic

  ; SANDYB: vmovups
  ; SANDYB-NEXT: vinsertf128
  ; SANDYB-NEXT: retq

  ; BTVER2: vmovups
  ; BTVER2-NEXT: retq

  ; HASWELL: vmovups
  ; HASWELL-NEXT: retq

  %ptr1 = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i64 3
  %ptr2 = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i64 4
  %v1 = load <4 x float>, <4 x float>* %ptr1, align 1
  %v2 = load <4 x float>, <4 x float>* %ptr2, align 1
  %v3 = shufflevector <4 x float> %v1, <4 x float> %v2, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x float> %v3
}

; Swap the order of the shufflevector operands to ensure that the
; pattern still matches.
define <8 x float> @combine_16_byte_loads_no_intrinsic_swap(<4 x float>* %ptr) {
  ; CHECK-LABEL: combine_16_byte_loads_no_intrinsic_swap

  ; SANDYB: vmovups
  ; SANDYB-NEXT: vinsertf128
  ; SANDYB-NEXT: retq

  ; BTVER2: vmovups
  ; BTVER2-NEXT: retq

  ; HASWELL: vmovups
  ; HASWELL-NEXT: retq

  %ptr1 = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i64 4
  %ptr2 = getelementptr inbounds <4 x float>, <4 x float>* %ptr, i64 5
  %v1 = load <4 x float>, <4 x float>* %ptr1, align 1
  %v2 = load <4 x float>, <4 x float>* %ptr2, align 1
  %v3 = shufflevector <4 x float> %v2, <4 x float> %v1, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3>
  ret <8 x float> %v3
}

; Check each element type other than float to make sure it is handled correctly.
; Use the loaded values with an 'add' to make sure we're using the correct load type.
; Even though BtVer2 has fast 32-byte loads, we should not generate those for
; 256-bit integer vectors because BtVer2 doesn't have AVX2.

define <4 x i64> @combine_16_byte_loads_i64(<2 x i64>* %ptr, <4 x i64> %x) {
  ; CHECK-LABEL: combine_16_byte_loads_i64

  ; SANDYB: vextractf128
  ; SANDYB-NEXT: vpaddq
  ; SANDYB-NEXT: vpaddq
  ; SANDYB-NEXT: vinsertf128
  ; SANDYB-NEXT: retq

  ; BTVER2: vextractf128
  ; BTVER2-NEXT: vpaddq
  ; BTVER2-NEXT: vpaddq
  ; BTVER2-NEXT: vinsertf128
  ; BTVER2-NEXT: retq

  ; HASWELL-NOT: vextract
  ; HASWELL: vpaddq
  ; HASWELL-NEXT: retq

  %ptr1 = getelementptr inbounds <2 x i64>, <2 x i64>* %ptr, i64 5
  %ptr2 = getelementptr inbounds <2 x i64>, <2 x i64>* %ptr, i64 6
  %v1 = load <2 x i64>, <2 x i64>* %ptr1, align 1
  %v2 = load <2 x i64>, <2 x i64>* %ptr2, align 1
  %v3 = shufflevector <2 x i64> %v1, <2 x i64> %v2, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %v4 = add <4 x i64> %v3, %x
  ret <4 x i64> %v4
}

define <8 x i32> @combine_16_byte_loads_i32(<4 x i32>* %ptr, <8 x i32> %x) {
  ; CHECK-LABEL: combine_16_byte_loads_i32

  ; SANDYB: vextractf128
  ; SANDYB-NEXT: vpaddd
  ; SANDYB-NEXT: vpaddd
  ; SANDYB-NEXT: vinsertf128
  ; SANDYB-NEXT: retq

  ; BTVER2: vextractf128
  ; BTVER2-NEXT: vpaddd
  ; BTVER2-NEXT: vpaddd
  ; BTVER2-NEXT: vinsertf128
  ; BTVER2-NEXT: retq

  ; HASWELL-NOT: vextract
  ; HASWELL: vpaddd
  ; HASWELL-NEXT: retq

  %ptr1 = getelementptr inbounds <4 x i32>, <4 x i32>* %ptr, i64 6
  %ptr2 = getelementptr inbounds <4 x i32>, <4 x i32>* %ptr, i64 7
  %v1 = load <4 x i32>, <4 x i32>* %ptr1, align 1
  %v2 = load <4 x i32>, <4 x i32>* %ptr2, align 1
  %v3 = shufflevector <4 x i32> %v1, <4 x i32> %v2, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v4 = add <8 x i32> %v3, %x
  ret <8 x i32> %v4
}

define <16 x i16> @combine_16_byte_loads_i16(<8 x i16>* %ptr, <16 x i16> %x) {
  ; CHECK-LABEL: combine_16_byte_loads_i16

  ; SANDYB: vextractf128
  ; SANDYB-NEXT: vpaddw
  ; SANDYB-NEXT: vpaddw
  ; SANDYB-NEXT: vinsertf128
  ; SANDYB-NEXT: retq

  ; BTVER2: vextractf128
  ; BTVER2-NEXT: vpaddw
  ; BTVER2-NEXT: vpaddw
  ; BTVER2-NEXT: vinsertf128
  ; BTVER2-NEXT: retq

  ; HASWELL-NOT: vextract
  ; HASWELL: vpaddw
  ; HASWELL-NEXT: retq

  %ptr1 = getelementptr inbounds <8 x i16>, <8 x i16>* %ptr, i64 7
  %ptr2 = getelementptr inbounds <8 x i16>, <8 x i16>* %ptr, i64 8
  %v1 = load <8 x i16>, <8 x i16>* %ptr1, align 1
  %v2 = load <8 x i16>, <8 x i16>* %ptr2, align 1
  %v3 = shufflevector <8 x i16> %v1, <8 x i16> %v2, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %v4 = add <16 x i16> %v3, %x
  ret <16 x i16> %v4
}

define <32 x i8> @combine_16_byte_loads_i8(<16 x i8>* %ptr, <32 x i8> %x) {
  ; CHECK-LABEL: combine_16_byte_loads_i8

  ; SANDYB: vextractf128
  ; SANDYB-NEXT: vpaddb
  ; SANDYB-NEXT: vpaddb
  ; SANDYB-NEXT: vinsertf128
  ; SANDYB-NEXT: retq

  ; BTVER2: vextractf128
  ; BTVER2-NEXT: vpaddb
  ; BTVER2-NEXT: vpaddb
  ; BTVER2-NEXT: vinsertf128
  ; BTVER2-NEXT: retq

  ; HASWELL-NOT: vextract
  ; HASWELL: vpaddb
  ; HASWELL-NEXT: retq

  %ptr1 = getelementptr inbounds <16 x i8>, <16 x i8>* %ptr, i64 8
  %ptr2 = getelementptr inbounds <16 x i8>, <16 x i8>* %ptr, i64 9
  %v1 = load <16 x i8>, <16 x i8>* %ptr1, align 1
  %v2 = load <16 x i8>, <16 x i8>* %ptr2, align 1
  %v3 = shufflevector <16 x i8> %v1, <16 x i8> %v2, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %v4 = add <32 x i8> %v3, %x
  ret <32 x i8> %v4
}

define <4 x double> @combine_16_byte_loads_double(<2 x double>* %ptr, <4 x double> %x) {
  ; CHECK-LABEL: combine_16_byte_loads_double

  ; SANDYB: vmovupd
  ; SANDYB-NEXT: vinsertf128
  ; SANDYB-NEXT: vaddpd
  ; SANDYB-NEXT: retq

  ; BTVER2-NOT: vinsertf128
  ; BTVER2: vaddpd
  ; BTVER2-NEXT: retq

  ; HASWELL-NOT: vinsertf128
  ; HASWELL: vaddpd
  ; HASWELL-NEXT: retq

  %ptr1 = getelementptr inbounds <2 x double>, <2 x double>* %ptr, i64 9
  %ptr2 = getelementptr inbounds <2 x double>, <2 x double>* %ptr, i64 10
  %v1 = load <2 x double>, <2 x double>* %ptr1, align 1
  %v2 = load <2 x double>, <2 x double>* %ptr2, align 1
  %v3 = shufflevector <2 x double> %v1, <2 x double> %v2, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %v4 = fadd <4 x double> %v3, %x
  ret <4 x double> %v4
}

