; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=skx | FileCheck %s --check-prefix=CHECK-SKX

; CHECK-LABEL: test1:
; CHECK: vpermps
; CHECK: ret
define <16 x float> @test1(<16 x float> %a) nounwind {
  %c = shufflevector <16 x float> %a, <16 x float> undef, <16 x i32> <i32 2, i32 5, i32 undef, i32 undef, i32 7, i32 undef, i32 10, i32 1,  i32 0, i32 5, i32 undef, i32 4, i32 7, i32 undef, i32 10, i32 1>
  ret <16 x float> %c
}

; CHECK-LABEL: test2:
; CHECK: vpermd
; CHECK: ret
define <16 x i32> @test2(<16 x i32> %a) nounwind {
  %c = shufflevector <16 x i32> %a, <16 x i32> undef, <16 x i32> <i32 2, i32 5, i32 undef, i32 undef, i32 7, i32 undef, i32 10, i32 1,  i32 0, i32 5, i32 undef, i32 4, i32 7, i32 undef, i32 10, i32 1>
  ret <16 x i32> %c
}

; CHECK-LABEL: test3:
; CHECK: vpermq
; CHECK: ret
define <8 x i64> @test3(<8 x i64> %a) nounwind {
  %c = shufflevector <8 x i64> %a, <8 x i64> undef, <8 x i32> <i32 2, i32 5, i32 1, i32 undef, i32 7, i32 undef, i32 3, i32 1>
  ret <8 x i64> %c
}

; CHECK-LABEL: test4:
; CHECK: vpermpd
; CHECK: ret
define <8 x double> @test4(<8 x double> %a) nounwind {
  %c = shufflevector <8 x double> %a, <8 x double> undef, <8 x i32> <i32 1, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <8 x double> %c
}

; CHECK-LABEL: test5:
; CHECK: vpermt2pd
; CHECK: ret
define <8 x double> @test5(<8 x double> %a, <8 x double> %b) nounwind {
  %c = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 2, i32 8, i32 0, i32 1, i32 6, i32 10, i32 4, i32 5>
  ret <8 x double> %c
}

; CHECK-LABEL: test6:
; CHECK: vpermq $30
; CHECK: ret
define <8 x i64> @test6(<8 x i64> %a) nounwind {
  %c = shufflevector <8 x i64> %a, <8 x i64> undef, <8 x i32> <i32 2, i32 3, i32 1, i32 0, i32 6, i32 7, i32 5, i32 4>
  ret <8 x i64> %c
}

; CHECK-LABEL: test7:
; CHECK: vpermt2q
; CHECK: ret
define <8 x i64> @test7(<8 x i64> %a, <8 x i64> %b) nounwind {
  %c = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 2, i32 8, i32 0, i32 1, i32 6, i32 10, i32 4, i32 5>
  ret <8 x i64> %c
}

; CHECK-LABEL: test8:
; CHECK: vpermt2d
; CHECK: ret
define <16 x i32> @test8(<16 x i32> %a, <16 x i32> %b) nounwind {
  %c = shufflevector <16 x i32> %a, <16 x i32> %b, <16 x i32> <i32 15, i32 31, i32 14, i32 22, i32 13, i32 29, i32 4, i32 28, i32 11, i32 27, i32 10, i32 26, i32 9, i32 25, i32 8, i32 24>
  ret <16 x i32> %c
}

; CHECK-LABEL: test9:
; CHECK: vpermt2ps
; CHECK: ret
define <16 x float> @test9(<16 x float> %a, <16 x float> %b) nounwind {
  %c = shufflevector <16 x float> %a, <16 x float> %b, <16 x i32> <i32 15, i32 31, i32 14, i32 22, i32 13, i32 29, i32 4, i32 28, i32 11, i32 27, i32 10, i32 26, i32 9, i32 25, i32 8, i32 24>
  ret <16 x float> %c
}

; CHECK-LABEL: test10:
; CHECK: vpermt2ps (
; CHECK: ret
define <16 x float> @test10(<16 x float> %a, <16 x float>* %b) nounwind {
  %c = load <16 x float>, <16 x float>* %b
  %d = shufflevector <16 x float> %a, <16 x float> %c, <16 x i32> <i32 15, i32 31, i32 14, i32 22, i32 13, i32 29, i32 4, i32 28, i32 11, i32 27, i32 10, i32 26, i32 9, i32 25, i32 8, i32 24>
  ret <16 x float> %d
}

; CHECK-LABEL: test11:
; CHECK: vpermt2d 
; CHECK: ret
define <16 x i32> @test11(<16 x i32> %a, <16 x i32>* %b) nounwind {
  %c = load <16 x i32>, <16 x i32>* %b
  %d = shufflevector <16 x i32> %a, <16 x i32> %c, <16 x i32> <i32 15, i32 31, i32 14, i32 22, i32 13, i32 29, i32 4, i32 28, i32 11, i32 27, i32 10, i32 26, i32 9, i32 25, i32 8, i32 24>
  ret <16 x i32> %d
}

; CHECK-LABEL: test13
; CHECK: vpermilps $177, %zmm
; CHECK: ret
define <16 x float> @test13(<16 x float> %a) {
 %b = shufflevector <16 x float> %a, <16 x float> undef, <16 x i32><i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6, i32 9, i32 8, i32 11, i32 10, i32 13, i32 12, i32 15, i32 14>
 ret <16 x float> %b
}

; CHECK-LABEL: test14
; CHECK: vpermilpd $203, %zmm
; CHECK: ret
define <8 x double> @test14(<8 x double> %a) {
 %b = shufflevector <8 x double> %a, <8 x double> undef, <8 x i32><i32 1, i32 1, i32 2, i32 3, i32 4, i32 4, i32 7, i32 7>
 ret <8 x double> %b
}

; CHECK-LABEL: test15
; CHECK: vpshufd $177, %zmm
; CHECK: ret
define <16 x i32> @test15(<16 x i32> %a) {
; mask 1-0-3-2 = 10110001 = 0xb1 = 177
 %b = shufflevector <16 x i32> %a, <16 x i32> undef, <16 x i32><i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6, i32 9, i32 8, i32 11, i32 10, i32 13, i32 12, i32 15, i32 14>
 ret <16 x i32> %b
}
; CHECK-LABEL: test16
; CHECK: valignq $3, %zmm0, %zmm1
; CHECK: ret
define <8 x double> @test16(<8 x double> %a, <8 x double> %b) nounwind {
  %c = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10>
  ret <8 x double> %c
}

; CHECK-LABEL: test17
; CHECK: vshufpd $19, %zmm1, %zmm0
; CHECK: ret
define <8 x double> @test17(<8 x double> %a, <8 x double> %b) nounwind {
  %c = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 9, i32 2, i32 10, i32 5, i32 undef, i32 undef, i32 undef>
  ret <8 x double> %c
}

; CHECK-LABEL: test18
; CHECK: vpunpckhdq %zmm
; CHECK: ret
define <16 x i32> @test18(<16 x i32> %a, <16 x i32> %c) {
 %b = shufflevector <16 x i32> %a, <16 x i32> %c, <16 x i32><i32 2, i32 18, i32 3, i32 19, i32 6, i32 22, i32 7, i32 23, i32 10, i32 26, i32 11, i32 27, i32 14, i32 30, i32 15, i32 31>
 ret <16 x i32> %b
}

; CHECK-LABEL: test19
; CHECK: vpunpckldq %zmm
; CHECK: ret
define <16 x i32> @test19(<16 x i32> %a, <16 x i32> %c) {
 %b = shufflevector <16 x i32> %a, <16 x i32> %c, <16 x i32><i32 0, i32 16, i32 1, i32 17, i32 4, i32 20, i32 5, i32 21, i32 8, i32 24, i32 9, i32 25, i32 12, i32 28, i32 13, i32 29>
 ret <16 x i32> %b
}

; CHECK-LABEL: test20
; CHECK: vpunpckhqdq  %zmm
; CHECK: ret
define <8 x i64> @test20(<8 x i64> %a, <8 x i64> %c) {
 %b = shufflevector <8 x i64> %a, <8 x i64> %c, <8 x i32><i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
 ret <8 x i64> %b
}

; CHECK-LABEL: test21
; CHECK: vbroadcastsd  %xmm0, %zmm
; CHECK: ret
define <8 x double> @test21(<8 x double> %a, <8 x double> %b) {
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x double> %shuffle
}

; CHECK-LABEL: test22
; CHECK: vpbroadcastq  %xmm0, %zmm
; CHECK: ret
define <8 x i64> @test22(<8 x i64> %a, <8 x i64> %b) {
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i64> %shuffle
}

; CHECK-LABEL: @test23
; CHECK: vshufps
; CHECK: vshufps
; CHECK: ret
define <16 x i32> @test23(<16 x i32> %a, <16 x i32> %b) nounwind {
  %c = shufflevector <16 x i32> %a, <16 x i32> %b, <16 x i32> <i32 0, i32 1, i32 2, i32 19, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <16 x i32> %c
}

; CHECK-LABEL: @test24
; CHECK: vpermt2d
; CHECK: ret
define <16 x i32> @test24(<16 x i32> %a, <16 x i32> %b) nounwind {
  %c = shufflevector <16 x i32> %a, <16 x i32> %b, <16 x i32> <i32 0, i32 1, i32 2, i32 19, i32 25, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <16 x i32> %c
}

; CHECK-LABEL: @test25
; CHECK: vshufps  $52
; CHECK: ret
define <16 x i32> @test25(<16 x i32> %a, <16 x i32> %b) nounwind {
; mask - 0-1-3-0 00110100 = 0x34 = 52
  %c = shufflevector <16 x i32> %a, <16 x i32> %b, <16 x i32> <i32 0, i32 1, i32 19, i32 16, i32 4, i32 5, i32 23, i32 undef, i32 8, i32 9, i32 27, i32 undef, i32 12, i32 13, i32 undef, i32 undef>
  ret <16 x i32> %c
}

; CHECK-LABEL: @test26
; CHECK: vmovshdup
; CHECK: ret
define <16 x i32> @test26(<16 x i32> %a) nounwind {
  %c = shufflevector <16 x i32> %a, <16 x i32> undef, <16 x i32> <i32 1, i32 1, i32 3, i32 3, i32 5, i32 5, i32 7, i32 undef, i32 9, i32 9, i32 undef, i32 11, i32 13, i32 undef, i32 undef, i32 undef>
  ret <16 x i32> %c
}

; CHECK-LABEL: @test27
; CHECK: ret
define <16 x i32> @test27(<4 x i32>%a) {
 %res = shufflevector <4 x i32> %a, <4 x i32> undef, <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
 ret <16 x i32> %res
}

; CHECK-LABEL: test28
; CHECK: vpshufhw $177, %ymm
; CHECK: ret
define <16 x i16> @test28(<16 x i16> %a) {
 %b = shufflevector <16 x i16> %a, <16 x i16> undef, <16 x i32><i32 0, i32 1, i32 2, i32 3, i32 5, i32 4, i32 7, i32 6, i32 8, i32 9, i32 10, i32 11, i32 13, i32 12, i32 15, i32 14>
 ret <16 x i16> %b
}

; CHECK-LABEL: test29
; CHECK: vunpcklps %zmm
; CHECK: ret
define <16 x float> @test29(<16 x float> %a, <16 x float> %c) {
 %b = shufflevector <16 x float> %a, <16 x float> %c, <16 x i32><i32 0, i32 16, i32 1, i32 17, i32 4, i32 20, i32 5, i32 21, i32 8, i32 24, i32 9, i32 25, i32 12, i32 28, i32 13, i32 29>
 ret <16 x float> %b
}

; CHECK-LABEL: @test30
; CHECK: vshufps $144, %zmm
; CHECK: ret
define <16 x float> @test30(<16 x float> %a, <16 x float> %c) {
 %b = shufflevector <16 x float> %a, <16 x float> %c, <16 x i32><i32 0, i32 0, i32 17, i32 18, i32 4, i32 4, i32 21, i32 22, i32 8, i32 8, i32 25, i32 26, i32 12, i32 12, i32 29, i32 30>
 ret <16 x float> %b
}

; CHECK-LABEL: test31
; CHECK: valignd $3, %zmm0, %zmm1
; CHECK: ret
define <16 x i32> @test31(<16 x i32> %a, <16 x i32> %b) nounwind {
  %c = shufflevector <16 x i32> %a, <16 x i32> %b, <16 x i32> <i32 3, i32 4, i32 5, i32 undef, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
  ret <16 x i32> %c
}

; CHECK-LABEL: test32
; CHECK: vshufpd $99, %zmm0, %zmm1
; CHECK: ret
define <8 x double> @test32(<8 x double> %a, <8 x double> %b) nounwind {
  %c = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 9, i32 1, i32 10, i32 2, i32 undef, i32 5, i32 15, i32 undef>
  ret <8 x double> %c
}

; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl | FileCheck %s
define <8 x double> @test_vshuff64x2_512(<8 x double> %x, <8 x double> %x1) nounwind {
; CHECK-LABEL: test_vshuff64x2_512:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vshuff64x2 $136, %zmm0, %zmm0, %zmm0
; CHECK-NEXT:    retq
  %res = shufflevector <8 x double> %x, <8 x double> %x1, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 0, i32 1,  i32 4, i32 5>
  ret <8 x double> %res
}

define <8 x double> @test_vshuff64x2_512_mask(<8 x double> %x, <8 x double> %x1, <8 x i1> %mask) nounwind {
; CHECK-LABEL: test_vshuff64x2_512_mask:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpmovsxwq %xmm2, %zmm1
; CHECK-NEXT:    vpandq {{.*}}(%rip){1to8}, %zmm1, %zmm1
; CHECK-NEXT:    vptestmq %zmm1, %zmm1, %k1
; CHECK-NEXT:    vshuff64x2 $136, %zmm0, %zmm0, %zmm0 {%k1} {z}
; CHECK-NEXT:    retq
  %y = shufflevector <8 x double> %x, <8 x double> %x1, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 0, i32 1,  i32 4, i32 5>
  %res = select <8 x i1> %mask, <8 x double> %y, <8 x double> zeroinitializer
  ret <8 x double> %res
}

define <8 x i64> @test_vshufi64x2_512_mask(<8 x i64> %x, <8 x i64> %x1, <8 x i1> %mask) nounwind {
; CHECK-LABEL: test_vshufi64x2_512_mask:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpmovsxwq %xmm2, %zmm1
; CHECK-NEXT:    vpandq {{.*}}(%rip){1to8}, %zmm1, %zmm1
; CHECK-NEXT:    vptestmq %zmm1, %zmm1, %k1
; CHECK-NEXT:    vshufi64x2 $168, %zmm0, %zmm0, %zmm0 {%k1}
; CHECK-NEXT:    retq
  %y = shufflevector <8 x i64> %x, <8 x i64> %x1, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 4, i32 5,  i32 4, i32 5>
  %res = select <8 x i1> %mask, <8 x i64> %y, <8 x i64> %x
  ret <8 x i64> %res
}

define <8 x double> @test_vshuff64x2_512_mem(<8 x double> %x, <8 x double> *%ptr) nounwind {
; CHECK-LABEL: test_vshuff64x2_512_mem:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vshuff64x2 $40, %zmm0, %zmm0, %zmm0
; CHECK-NEXT:    retq
  %x1   = load <8 x double>,<8 x double> *%ptr,align 1
  %res = shufflevector <8 x double> %x, <8 x double> %x1, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 4, i32 5,  i32 0, i32 1>
  ret <8 x double> %res
}

define <16 x float> @test_vshuff32x4_512_mem(<16 x float> %x, <16 x float> *%ptr) nounwind {
; CHECK-LABEL: test_vshuff32x4_512_mem:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vshuff64x2 $20, %zmm0, %zmm0, %zmm0
; CHECK-NEXT:    retq
  %x1   = load <16 x float>,<16 x float> *%ptr,align 1
  %res = shufflevector <16 x float> %x, <16 x float> %x1, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3>
  ret <16 x float> %res
}

define <16 x i32> @test_align_v16i32_rr(<16 x i32> %a, <16 x i32> %b) nounwind {
; CHECK-LABEL: test_align_v16i32_rr:
; CHECK:       ## BB#0:
; CHECK-NEXT:    valignd $3, %zmm0, %zmm1, %zmm0
; CHECK-NEXT:    retq
  %c = shufflevector <16 x i32> %a, <16 x i32> %b, <16 x i32> <i32 3, i32 4, i32 5, i32 undef, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
  ret <16 x i32> %c
}

define <16 x i32> @test_align_v16i32_rm(<16 x i32>* %a.ptr, <16 x i32> %b) nounwind {
; CHECK-LABEL: test_align_v16i32_rm:
; CHECK:       ## BB#0:
; CHECK-NEXT:    valignd $3, (%rdi), %zmm0, %zmm0
; CHECK-NEXT:    retq
  %a = load <16 x i32>, <16 x i32>* %a.ptr
  %c = shufflevector <16 x i32> %a, <16 x i32> %b, <16 x i32> <i32 3, i32 4, i32 5, i32 undef, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
  ret <16 x i32> %c
}

define <16 x i32> @test_align_v16i32_rm_mask(<16 x i32>* %a.ptr, <16 x i32> %b, <16 x i1> %mask) nounwind {
; CHECK-LABEL: test_align_v16i32_rm_mask:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpmovsxbd %xmm1, %zmm1
; CHECK-NEXT:    vpandd {{.*}}(%rip){1to16}, %zmm1, %zmm1
; CHECK-NEXT:    vptestmd %zmm1, %zmm1, %k1
; CHECK-NEXT:    vmovdqa32 (%rdi), %zmm1
; CHECK-NEXT:    valignd $3, %zmm1, %zmm0, %zmm1 {%k1}
; CHECK-NEXT:    vmovaps %zmm1, %zmm0
; CHECK-NEXT:    retq
;
; CHECK-SKX-LABEL: test_align_v16i32_rm_mask:
; CHECK-SKX:       ## BB#0:
; CHECK-SKX-NEXT:    vpmovb2m %xmm1, %k1
; CHECK-SKX-NEXT:    vmovdqa32 (%rdi), %zmm1
; CHECK-SKX-NEXT:    valignd $3, %zmm1, %zmm0, %zmm1 {%k1}
; CHECK-SKX-NEXT:    vmovaps %zmm1, %zmm0
; CHECK-SKX-NEXT:    retq
  %a = load <16 x i32>, <16 x i32>* %a.ptr
  %c = shufflevector <16 x i32> %a, <16 x i32> %b, <16 x i32> <i32 3, i32 4, i32 5, i32 undef, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
  %res = select <16 x i1> %mask,<16 x i32> %c, <16 x i32> %a
  ret <16 x i32> %res
}

define <8 x double> @test_align_v8f64_rr(<8 x double> %a, <8 x double> %b) nounwind {
; CHECK-LABEL: test_align_v8f64_rr:
; CHECK:       ## BB#0:
; CHECK-NEXT:    valignq $3, %zmm0, %zmm1, %zmm0
; CHECK-NEXT:    retq
  %c = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10>
  ret <8 x double> %c
}

define <8 x double> @test_align_v18f64_rm(<8 x double>* %a.ptr, <8 x double> %b) nounwind {
; CHECK-LABEL: test_align_v18f64_rm:
; CHECK:       ## BB#0:
; CHECK-NEXT:    valignq $3, (%rdi), %zmm0, %zmm0
; CHECK-NEXT:    retq
  %a = load <8 x double>, <8 x double>* %a.ptr
  %c = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10>
  ret <8 x double> %c
}

define <8 x double> @test_align_v18f64_rm_mask(<8 x double>* %a.ptr, <8 x double> %b, <8 x i1> %mask) nounwind {
; CHECK-LABEL: test_align_v18f64_rm_mask:
; CHECK:       ## BB#0:
; CHECK-NEXT:    vpmovsxwq %xmm1, %zmm1
; CHECK-NEXT:    vpandq {{.*}}(%rip){1to8}, %zmm1, %zmm1
; CHECK-NEXT:    vptestmq %zmm1, %zmm1, %k1
; CHECK-NEXT:    valignq $3, (%rdi), %zmm0, %zmm0 {%k1} {z}
; CHECK-NEXT:    retq
;
; CHECK-SKX-LABEL: test_align_v18f64_rm_mask:
; CHECK-SKX:       ## BB#0:
; CHECK-SKX-NEXT:    vpmovw2m %xmm1, %k1
; CHECK-SKX-NEXT:    valignq $3, (%rdi), %zmm0, %zmm0 {%k1} {z}
; CHECK-SKX-NEXT:    retq
  %a = load <8 x double>, <8 x double>* %a.ptr
  %c = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10>
  %res = select <8 x i1> %mask,<8 x double> %c, <8 x double> zeroinitializer
  ret <8 x double> %res
}

