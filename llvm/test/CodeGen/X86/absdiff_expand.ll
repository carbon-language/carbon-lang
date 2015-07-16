; RUN: llc -mtriple=x86_64-unknown-linux-gnu  < %s | FileCheck %s -check-prefix=CHECK

declare <4 x i8> @llvm.uabsdiff.v4i8(<4 x i8>, <4 x i8>)

define <4 x i8> @test_uabsdiff_v4i8_expand(<4 x i8> %a1, <4 x i8> %a2) {
; CHECK-LABEL: test_uabsdiff_v4i8_expand
; CHECK:             psubd  %xmm1, %xmm0
; CHECK-NEXT:        pxor   %xmm1, %xmm1
; CHECK-NEXT:        psubd  %xmm0, %xmm1
; CHECK-NEXT:        movdqa  .LCPI{{[0-9_]*}}
; CHECK-NEXT:        movdqa  %xmm1, %xmm3
; CHECK-NEXT:        pxor   %xmm2, %xmm3
; CHECK-NEXT:        pcmpgtd        %xmm3, %xmm2
; CHECK-NEXT:        pand    %xmm2, %xmm0
; CHECK-NEXT:        pandn   %xmm1, %xmm2
; CHECK-NEXT:        por     %xmm2, %xmm0
; CHECK-NEXT:        retq

  %1 = call <4 x i8> @llvm.uabsdiff.v4i8(<4 x i8> %a1, <4 x i8> %a2)
  ret <4 x i8> %1
}

declare <4 x i8> @llvm.sabsdiff.v4i8(<4 x i8>, <4 x i8>)

define <4 x i8> @test_sabsdiff_v4i8_expand(<4 x i8> %a1, <4 x i8> %a2) {
; CHECK-LABEL: test_sabsdiff_v4i8_expand
; CHECK:      psubd  %xmm1, %xmm0
; CHECK-NEXT: pxor   %xmm1, %xmm1
; CHECK-NEXT: pxor    %xmm2, %xmm2
; CHECK-NEXT: psubd  %xmm0, %xmm2
; CHECK-NEXT: pcmpgtd  %xmm2, %xmm1
; CHECK-NEXT: pand    %xmm1, %xmm0
; CHECK-NEXT: pandn   %xmm2, %xmm1
; CHECK-NEXT: por     %xmm1, %xmm0
; CHECK-NEXT: retq

  %1 = call <4 x i8> @llvm.sabsdiff.v4i8(<4 x i8> %a1, <4 x i8> %a2)
  ret <4 x i8> %1
}


declare <8 x i8> @llvm.sabsdiff.v8i8(<8 x i8>, <8 x i8>)

define <8 x i8> @test_sabsdiff_v8i8_expand(<8 x i8> %a1, <8 x i8> %a2) {
; CHECK-LABEL: test_sabsdiff_v8i8_expand
; CHECK:      psubw  %xmm1, %xmm0
; CHECK-NEXT: pxor   %xmm1, %xmm1
; CHECK-NEXT: pxor   %xmm2, %xmm2
; CHECK-NEXT: psubw  %xmm0, %xmm2
; CHECK-NEXT: pcmpgtw        %xmm2, %xmm1
; CHECK-NEXT: pand  %xmm1, %xmm0
; CHECK-NEXT: pandn %xmm2, %xmm1
; CHECK-NEXT: por  %xmm1, %xmm0
; CHECK-NEXT: retq
  %1 = call <8 x i8> @llvm.sabsdiff.v8i8(<8 x i8> %a1, <8 x i8> %a2)
  ret <8 x i8> %1
}

declare <16 x i8> @llvm.uabsdiff.v16i8(<16 x i8>, <16 x i8>)

define <16 x i8> @test_uabsdiff_v16i8_expand(<16 x i8> %a1, <16 x i8> %a2) {
; CHECK-LABEL: test_uabsdiff_v16i8_expand
; CHECK:             psubb  %xmm1, %xmm0
; CHECK-NEXT:        pxor   %xmm1, %xmm1
; CHECK-NEXT:        psubb  %xmm0, %xmm1
; CHECK-NEXT:        movdqa  .LCPI{{[0-9_]*}}
; CHECK-NEXT:        movdqa  %xmm1, %xmm3
; CHECK-NEXT:        pxor   %xmm2, %xmm3
; CHECK-NEXT:        pcmpgtb        %xmm3, %xmm2
; CHECK-NEXT:        pand    %xmm2, %xmm0
; CHECK-NEXT:        pandn   %xmm1, %xmm2
; CHECK-NEXT:        por     %xmm2, %xmm0
; CHECK-NEXT:        retq
  %1 = call <16 x i8> @llvm.uabsdiff.v16i8(<16 x i8> %a1, <16 x i8> %a2)
  ret <16 x i8> %1
}

declare <8 x i16> @llvm.uabsdiff.v8i16(<8 x i16>, <8 x i16>)

define <8 x i16> @test_uabsdiff_v8i16_expand(<8 x i16> %a1, <8 x i16> %a2) {
; CHECK-LABEL: test_uabsdiff_v8i16_expand
; CHECK:             psubw  %xmm1, %xmm0
; CHECK-NEXT:        pxor   %xmm1, %xmm1
; CHECK-NEXT:        psubw  %xmm0, %xmm1
; CHECK-NEXT:        movdqa  .LCPI{{[0-9_]*}}
; CHECK-NEXT:        movdqa  %xmm1, %xmm3
; CHECK-NEXT:        pxor   %xmm2, %xmm3
; CHECK-NEXT:        pcmpgtw        %xmm3, %xmm2
; CHECK-NEXT:        pand    %xmm2, %xmm0
; CHECK-NEXT:        pandn   %xmm1, %xmm2
; CHECK-NEXT:        por     %xmm2, %xmm0
; CHECK-NEXT:        retq
  %1 = call <8 x i16> @llvm.uabsdiff.v8i16(<8 x i16> %a1, <8 x i16> %a2)
  ret <8 x i16> %1
}

declare <8 x i16> @llvm.sabsdiff.v8i16(<8 x i16>, <8 x i16>)

define <8 x i16> @test_sabsdiff_v8i16_expand(<8 x i16> %a1, <8 x i16> %a2) {
; CHECK-LABEL: test_sabsdiff_v8i16_expand
; CHECK:      psubw  %xmm1, %xmm0
; CHECK-NEXT: pxor   %xmm1, %xmm1
; CHECK-NEXT: pxor   %xmm2, %xmm2
; CHECK-NEXT: psubw  %xmm0, %xmm2
; CHECK-NEXT: pcmpgtw        %xmm2, %xmm1
; CHECK-NEXT: pand  %xmm1, %xmm0
; CHECK-NEXT: pandn %xmm2, %xmm1
; CHECK-NEXT: por  %xmm1, %xmm0
; CHECK-NEXT: retq
  %1 = call <8 x i16> @llvm.sabsdiff.v8i16(<8 x i16> %a1, <8 x i16> %a2)
  ret <8 x i16> %1
}

declare <4 x i32> @llvm.sabsdiff.v4i32(<4 x i32>, <4 x i32>)

define <4 x i32> @test_sabsdiff_v4i32_expand(<4 x i32> %a1, <4 x i32> %a2) {
; CHECK-LABEL: test_sabsdiff_v4i32_expand
; CHECK:             psubd  %xmm1, %xmm0
; CHECK-NEXT:        pxor  %xmm1, %xmm1
; CHECK-NEXT:        pxor  %xmm2, %xmm2
; CHECK-NEXT:        psubd  %xmm0, %xmm2
; CHECK-NEXT:        pcmpgtd        %xmm2, %xmm1
; CHECK-NEXT:        pand    %xmm1, %xmm0
; CHECK-NEXT:        pandn   %xmm2, %xmm1
; CHECK-NEXT:        por    %xmm1, %xmm0
; CHECK-NEXT:        retq
  %1 = call <4 x i32> @llvm.sabsdiff.v4i32(<4 x i32> %a1, <4 x i32> %a2)
  ret <4 x i32> %1
}

declare <4 x i32> @llvm.uabsdiff.v4i32(<4 x i32>, <4 x i32>)

define <4 x i32> @test_uabsdiff_v4i32_expand(<4 x i32> %a1, <4 x i32> %a2) {
; CHECK-LABEL: test_uabsdiff_v4i32_expand
; CHECK:             psubd  %xmm1, %xmm0
; CHECK-NEXT:        pxor   %xmm1, %xmm1
; CHECK-NEXT:        psubd  %xmm0, %xmm1
; CHECK-NEXT:        movdqa  .LCPI{{[0-9_]*}}
; CHECK-NEXT:        movdqa  %xmm1, %xmm3
; CHECK-NEXT:        pxor   %xmm2, %xmm3
; CHECK-NEXT:        pcmpgtd        %xmm3, %xmm2
; CHECK-NEXT:        pand    %xmm2, %xmm0
; CHECK-NEXT:        pandn   %xmm1, %xmm2
; CHECK-NEXT:        por     %xmm2, %xmm0
; CHECK-NEXT:        retq
  %1 = call <4 x i32> @llvm.uabsdiff.v4i32(<4 x i32> %a1, <4 x i32> %a2)
  ret <4 x i32> %1
}

declare <2 x i32> @llvm.sabsdiff.v2i32(<2 x i32>, <2 x i32>)

define <2 x i32> @test_sabsdiff_v2i32_expand(<2 x i32> %a1, <2 x i32> %a2) {
; CHECK-LABEL: test_sabsdiff_v2i32_expand
; CHECK:        psubq   %xmm1, %xmm0
; CHECK-NEXT:   pxor    %xmm1, %xmm1
; CHECK-NEXT:   psubq   %xmm0, %xmm1
; CHECK-NEXT:   movdqa  .LCPI{{[0-9_]*}}
; CHECK-NEXT:   movdqa  %xmm1, %xmm3
; CHECK-NEXT:   pxor    %xmm2, %xmm3
; CHECK-NEXT:   movdqa  %xmm2, %xmm4
; CHECK-NEXT:   pcmpgtd %xmm3, %xmm4
; CHECK-NEXT:   pshufd  $160, %xmm4, %xmm5      # xmm5 = xmm4[0,0,2,2]
; CHECK-NEXT:   pcmpeqd %xmm2, %xmm3
; CHECK-NEXT:   pshufd  $245, %xmm3, %xmm2      # xmm2 = xmm3[1,1,3,3]
; CHECK-NEXT:   pand    %xmm5, %xmm2
; CHECK-NEXT:   pshufd  $245, %xmm4, %xmm3      # xmm3 = xmm4[1,1,3,3]
; CHECK-NEXT:   por     %xmm2, %xmm3
; CHECK-NEXT:   pand    %xmm3, %xmm0
; CHECK-NEXT:   pandn   %xmm1, %xmm3
; CHECK-NEXT:   por     %xmm3, %xmm0
; CHECK-NEXT:   retq
  %1 = call <2 x i32> @llvm.sabsdiff.v2i32(<2 x i32> %a1, <2 x i32> %a2)
  ret <2 x i32> %1
}

declare <2 x i64> @llvm.sabsdiff.v2i64(<2 x i64>, <2 x i64>)

define <2 x i64> @test_sabsdiff_v2i64_expand(<2 x i64> %a1, <2 x i64> %a2) {
; CHECK-LABEL: test_sabsdiff_v2i64_expand
; CHECK:        psubq   %xmm1, %xmm0
; CHECK-NEXT:   pxor    %xmm1, %xmm1
; CHECK-NEXT:   psubq   %xmm0, %xmm1
; CHECK-NEXT:   movdqa  .LCPI{{[0-9_]*}}
; CHECK-NEXT:   movdqa  %xmm1, %xmm3
; CHECK-NEXT:   pxor    %xmm2, %xmm3
; CHECK-NEXT:   movdqa  %xmm2, %xmm4
; CHECK-NEXT:   pcmpgtd %xmm3, %xmm4
; CHECK-NEXT:   pshufd  $160, %xmm4, %xmm5      # xmm5 = xmm4[0,0,2,2]
; CHECK-NEXT:   pcmpeqd %xmm2, %xmm3
; CHECK-NEXT:   pshufd  $245, %xmm3, %xmm2      # xmm2 = xmm3[1,1,3,3]
; CHECK-NEXT:   pand    %xmm5, %xmm2
; CHECK-NEXT:   pshufd  $245, %xmm4, %xmm3      # xmm3 = xmm4[1,1,3,3]
; CHECK-NEXT:   por     %xmm2, %xmm3
; CHECK-NEXT:   pand    %xmm3, %xmm0
; CHECK-NEXT:   pandn   %xmm1, %xmm3
; CHECK-NEXT:   por     %xmm3, %xmm0
; CHECK-NEXT:   retq
  %1 = call <2 x i64> @llvm.sabsdiff.v2i64(<2 x i64> %a1, <2 x i64> %a2)
  ret <2 x i64> %1
}

declare <16 x i32> @llvm.sabsdiff.v16i32(<16 x i32>, <16 x i32>)

define <16 x i32> @test_sabsdiff_v16i32_expand(<16 x i32> %a1, <16 x i32> %a2) {
; CHECK-LABEL: test_sabsdiff_v16i32_expand
; CHECK:             psubd  %xmm4, %xmm0
; CHECK-NEXT:        pxor    %xmm8, %xmm8
; CHECK-NEXT:        pxor    %xmm9, %xmm9
; CHECK-NEXT:        psubd   %xmm0, %xmm9
; CHECK-NEXT:        pxor    %xmm4, %xmm4
; CHECK-NEXT:        pcmpgtd %xmm9, %xmm4
; CHECK-NEXT:        pand    %xmm4, %xmm0
; CHECK-NEXT:        pandn   %xmm9, %xmm4
; CHECK-NEXT:        por     %xmm4, %xmm0
; CHECK-NEXT:        psubd   %xmm5, %xmm1
; CHECK-NEXT:        pxor    %xmm4, %xmm4
; CHECK-NEXT:        psubd   %xmm1, %xmm4
; CHECK-NEXT:        pxor    %xmm5, %xmm5
; CHECK-NEXT:        pcmpgtd %xmm4, %xmm5
; CHECK-NEXT:        pand    %xmm5, %xmm1
; CHECK-NEXT:        pandn   %xmm4, %xmm5
; CHECK-NEXT:        por     %xmm5, %xmm1
; CHECK-NEXT:        psubd   %xmm6, %xmm2
; CHECK-NEXT:        pxor    %xmm4, %xmm4
; CHECK-NEXT:        psubd   %xmm2, %xmm4
; CHECK-NEXT:        pxor    %xmm5, %xmm5
; CHECK-NEXT:        pcmpgtd %xmm4, %xmm5
; CHECK-NEXT:        pand    %xmm5, %xmm2
; CHECK-NEXT:        pandn   %xmm4, %xmm5
; CHECK-NEXT:        por     %xmm5, %xmm2
; CHECK-NEXT:        psubd   %xmm7, %xmm3
; CHECK-NEXT:        pxor    %xmm4, %xmm4
; CHECK-NEXT:        psubd   %xmm3, %xmm4
; CHECK-NEXT:        pcmpgtd %xmm4, %xmm8
; CHECK-NEXT:        pand    %xmm8, %xmm3
; CHECK-NEXT:        pandn   %xmm4, %xmm8
; CHECK-NEXT:        por     %xmm8, %xmm3
; CHECK-NEXT:        retq
  %1 = call <16 x i32> @llvm.sabsdiff.v16i32(<16 x i32> %a1, <16 x i32> %a2)
  ret <16 x i32> %1
}

