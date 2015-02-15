; RUN: llc < %s -mcpu=x86-64 -mattr=+avx512f | FileCheck %s --check-prefix=ALL --check-prefix=AVX512 --check-prefix=AVX512F
; RUN: llc < %s -mcpu=x86-64 -mattr=+avx512bw | FileCheck %s --check-prefix=ALL --check-prefix=AVX512 --check-prefix=AVX512BW

target triple = "x86_64-unknown-unknown"

define <16 x float> @shuffle_v16f32_00_10_01_11_04_14_05_15_08_18_09_19_0c_1c_0d_1d(<16 x float> %a, <16 x float> %b) {
; ALL-LABEL: shuffle_v16f32_00_10_01_11_04_14_05_15_08_18_09_19_0c_1c_0d_1d:
; ALL:       # BB#0:
; ALL-NEXT:    vunpcklps {{.*#+}} zmm0 = zmm0[0],zmm1[0],zmm0[1],zmm1[1],zmm0[4],zmm1[4],zmm0[5],zmm1[5],zmm0[8],zmm1[8],zmm0[9],zmm1[9],zmm0[12],zmm1[12],zmm0[13],zmm1[13]
; ALL-NEXT:    retq
  %shuffle = shufflevector <16 x float> %a, <16 x float> %b, <16 x i32><i32 0, i32 16, i32 1, i32 17, i32 4, i32 20, i32 5, i32 21, i32 8, i32 24, i32 9, i32 25, i32 12, i32 28, i32 13, i32 29>
  ret <16 x float> %shuffle
}

define <16 x i32> @shuffle_v16i32_00_10_01_11_04_14_05_15_08_18_09_19_0c_1c_0d_1d(<16 x i32> %a, <16 x i32> %b) {
; ALL-LABEL: shuffle_v16i32_00_10_01_11_04_14_05_15_08_18_09_19_0c_1c_0d_1d:
; ALL:       # BB#0:
; ALL-NEXT:    vpunpckldq {{.*#+}} zmm0 = zmm0[0],zmm1[0],zmm0[1],zmm1[1],zmm0[4],zmm1[4],zmm0[5],zmm1[5],zmm0[8],zmm1[8],zmm0[9],zmm1[9],zmm0[12],zmm1[12],zmm0[13],zmm1[13]
; ALL-NEXT:    retq
  %shuffle = shufflevector <16 x i32> %a, <16 x i32> %b, <16 x i32><i32 0, i32 16, i32 1, i32 17, i32 4, i32 20, i32 5, i32 21, i32 8, i32 24, i32 9, i32 25, i32 12, i32 28, i32 13, i32 29>
  ret <16 x i32> %shuffle
}

define <16 x float> @shuffle_v16f32_02_12_03_13_06_16_07_17_0a_1a_0b_1b_0e_1e_0f_1f(<16 x float> %a, <16 x float> %b) {
; ALL-LABEL: shuffle_v16f32_02_12_03_13_06_16_07_17_0a_1a_0b_1b_0e_1e_0f_1f:
; ALL:       # BB#0:
; ALL-NEXT:    vunpckhps {{.*#+}} zmm0 = zmm0[2],zmm1[2],zmm0[3],zmm1[3],zmm0[6],zmm1[6],zmm0[7],zmm1[7],zmm0[10],zmm1[10],zmm0[11],zmm1[11],zmm0[14],zmm1[14],zmm0[15],zmm1[15]
; ALL-NEXT:    retq
  %shuffle = shufflevector <16 x float> %a, <16 x float> %b, <16 x i32><i32 2, i32 18, i32 3, i32 19, i32 6, i32 22, i32 7, i32 23, i32 10, i32 26, i32 11, i32 27, i32 14, i32 30, i32 15, i32 31>
  ret <16 x float> %shuffle
}

define <16 x i32> @shuffle_v16i32_02_12_03_13_06_16_07_17_0a_1a_0b_1b_0e_1e_0f_1f(<16 x i32> %a, <16 x i32> %b) {
; ALL-LABEL: shuffle_v16i32_02_12_03_13_06_16_07_17_0a_1a_0b_1b_0e_1e_0f_1f:
; ALL:       # BB#0:
; ALL-NEXT:    vpunpckhdq {{.*#+}} zmm0 = zmm0[2],zmm1[2],zmm0[3],zmm1[3],zmm0[6],zmm1[6],zmm0[7],zmm1[7],zmm0[10],zmm1[10],zmm0[11],zmm1[11],zmm0[14],zmm1[14],zmm0[15],zmm1[15]
; ALL-NEXT:    retq
  %shuffle = shufflevector <16 x i32> %a, <16 x i32> %b, <16 x i32><i32 2, i32 18, i32 3, i32 19, i32 6, i32 22, i32 7, i32 23, i32 10, i32 26, i32 11, i32 27, i32 14, i32 30, i32 15, i32 31>
  ret <16 x i32> %shuffle
}
