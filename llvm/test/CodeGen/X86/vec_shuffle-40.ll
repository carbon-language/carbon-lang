; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=core-avx2 | FileCheck %s

define void @shuffle_v16i16(<16 x i16>* %a) {
; CHECK-LABEL: shuffle_v16i16:
; CHECK: vpshufb {{.*}}%ymm
; CHECK-NOT: vpshufb {{.*}}%xmm
entry:
  %0 = load <16 x i16>* %a, align 32
  %shuffle = shufflevector <16 x i16> %0, <16 x i16> undef, <16 x i32> <i32 1, i32 1, i32 3, i32 3, i32 5, i32 5, i32 7, i32 7, i32 9, i32 9, i32 11, i32 11, i32 13, i32 13, i32 15, i32 15>
  store <16 x i16> %shuffle, <16 x i16>* %a, align 32
  ret void
}

define void @shuffle_v16i16_lanecrossing(<16 x i16>* %a) {
; CHECK-LABEL: shuffle_v16i16_lanecrossing:
; CHECK-NOT: vpshufb {{.*}}%ymm
entry:
  %0 = load <16 x i16>* %a, align 32
  %shuffle = shufflevector <16 x i16> %0, <16 x i16> undef, <16 x i32> <i32 1, i32 1, i32 3, i32 3, i32 5, i32 13, i32 7, i32 7, i32 9, i32 9, i32 11, i32 11, i32 13, i32 13, i32 15, i32 15>
  store <16 x i16> %shuffle, <16 x i16>* %a, align 32
  ret void
}
