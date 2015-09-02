; RUN: llc -mtriple=x86_64-apple-darwin  -mcpu=knl < %s | FileCheck %s -check-prefix=KNL

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; KNL-LABEL: test1
; KNL: kxnorw  %k1, %k1, %k1
; KNL: vgatherdps      (%rdi,%zmm0,4), %zmm1 {%k1}
define <16 x float> @test1(float* %base, <16 x i32> %ind) {

  %broadcast.splatinsert = insertelement <16 x float*> undef, float* %base, i32 0
  %broadcast.splat = shufflevector <16 x float*> %broadcast.splatinsert, <16 x float*> undef, <16 x i32> zeroinitializer

  %sext_ind = sext <16 x i32> %ind to <16 x i64>
  %gep.random = getelementptr float, <16 x float*> %broadcast.splat, <16 x i64> %sext_ind
  
  %res = call <16 x float> @llvm.masked.gather.v16f32(<16 x float*> %gep.random, i32 4, <16 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <16 x float> undef)
  ret <16 x float>%res
}

declare <16 x i32> @llvm.masked.gather.v16i32(<16 x i32*>, i32, <16 x i1>, <16 x i32>)
declare <16 x float> @llvm.masked.gather.v16f32(<16 x float*>, i32, <16 x i1>, <16 x float>)
declare <8 x i32> @llvm.masked.gather.v8i32(<8 x i32*> , i32, <8 x i1> , <8 x i32> )
  
; KNL-LABEL: test2
; KNL: kmovw %esi, %k1
; KNL: vgatherdps      (%rdi,%zmm0,4), %zmm1 {%k1}
define <16 x float> @test2(float* %base, <16 x i32> %ind, i16 %mask) {

  %broadcast.splatinsert = insertelement <16 x float*> undef, float* %base, i32 0
  %broadcast.splat = shufflevector <16 x float*> %broadcast.splatinsert, <16 x float*> undef, <16 x i32> zeroinitializer

  %sext_ind = sext <16 x i32> %ind to <16 x i64>
  %gep.random = getelementptr float, <16 x float*> %broadcast.splat, <16 x i64> %sext_ind
  %imask = bitcast i16 %mask to <16 x i1>
  %res = call <16 x float> @llvm.masked.gather.v16f32(<16 x float*> %gep.random, i32 4, <16 x i1> %imask, <16 x float>undef)
  ret <16 x float> %res
}

; KNL-LABEL: test3
; KNL: kmovw %esi, %k1
; KNL: vpgatherdd      (%rdi,%zmm0,4), %zmm1 {%k1}
define <16 x i32> @test3(i32* %base, <16 x i32> %ind, i16 %mask) {

  %broadcast.splatinsert = insertelement <16 x i32*> undef, i32* %base, i32 0
  %broadcast.splat = shufflevector <16 x i32*> %broadcast.splatinsert, <16 x i32*> undef, <16 x i32> zeroinitializer

  %sext_ind = sext <16 x i32> %ind to <16 x i64>
  %gep.random = getelementptr i32, <16 x i32*> %broadcast.splat, <16 x i64> %sext_ind
  %imask = bitcast i16 %mask to <16 x i1>
  %res = call <16 x i32> @llvm.masked.gather.v16i32(<16 x i32*> %gep.random, i32 4, <16 x i1> %imask, <16 x i32>undef)
  ret <16 x i32> %res
}

; KNL-LABEL: test4
; KNL: kmovw %esi, %k1
; KNL: kmovw
; KNL: vpgatherdd
; KNL: vpgatherdd

define <16 x i32> @test4(i32* %base, <16 x i32> %ind, i16 %mask) {

  %broadcast.splatinsert = insertelement <16 x i32*> undef, i32* %base, i32 0
  %broadcast.splat = shufflevector <16 x i32*> %broadcast.splatinsert, <16 x i32*> undef, <16 x i32> zeroinitializer

  %gep.random = getelementptr i32, <16 x i32*> %broadcast.splat, <16 x i32> %ind
  %imask = bitcast i16 %mask to <16 x i1>
  %gt1 = call <16 x i32> @llvm.masked.gather.v16i32(<16 x i32*> %gep.random, i32 4, <16 x i1> %imask, <16 x i32>undef)
  %gt2 = call <16 x i32> @llvm.masked.gather.v16i32(<16 x i32*> %gep.random, i32 4, <16 x i1> %imask, <16 x i32>%gt1)
  %res = add <16 x i32> %gt1, %gt2
  ret <16 x i32> %res
}

; KNL-LABEL: test5
; KNL: kmovw %k1, %k2
; KNL: vpscatterdd {{.*}}%k2
; KNL: vpscatterdd {{.*}}%k1

define void @test5(i32* %base, <16 x i32> %ind, i16 %mask, <16 x i32>%val) {

  %broadcast.splatinsert = insertelement <16 x i32*> undef, i32* %base, i32 0
  %broadcast.splat = shufflevector <16 x i32*> %broadcast.splatinsert, <16 x i32*> undef, <16 x i32> zeroinitializer

  %gep.random = getelementptr i32, <16 x i32*> %broadcast.splat, <16 x i32> %ind
  %imask = bitcast i16 %mask to <16 x i1>
  call void @llvm.masked.scatter.v16i32(<16 x i32>%val, <16 x i32*> %gep.random, i32 4, <16 x i1> %imask)
  call void @llvm.masked.scatter.v16i32(<16 x i32>%val, <16 x i32*> %gep.random, i32 4, <16 x i1> %imask)
  ret void
}

declare void @llvm.masked.scatter.v8i32(<8 x i32> , <8 x i32*> , i32 , <8 x i1> )
declare void @llvm.masked.scatter.v16i32(<16 x i32> , <16 x i32*> , i32 , <16 x i1> )

; KNL-LABEL: test6
; KNL: kxnorw  %k1, %k1, %k1
; KNL: kxnorw  %k2, %k2, %k2
; KNL: vpgatherqd      (,%zmm{{.*}}), %ymm{{.*}} {%k2}
; KNL: vpscatterqd     %ymm{{.*}}, (,%zmm{{.*}}) {%k1}
define <8 x i32> @test6(<8 x i32>%a1, <8 x i32*> %ptr) {

  %a = call <8 x i32> @llvm.masked.gather.v8i32(<8 x i32*> %ptr, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x i32> undef)

  call void @llvm.masked.scatter.v8i32(<8 x i32> %a1, <8 x i32*> %ptr, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
  ret <8 x i32>%a
}

; In this case the index should be promoted to <8 x i64> for KNL
; KNL-LABEL: test7
; KNL: vpmovsxdq %ymm0, %zmm0
; KNL: kmovw   %k1, %k2
; KNL: vpgatherqd {{.*}} {%k2}
; KNL: vpgatherqd {{.*}} {%k1}
define <8 x i32> @test7(i32* %base, <8 x i32> %ind, i8 %mask) {

  %broadcast.splatinsert = insertelement <8 x i32*> undef, i32* %base, i32 0
  %broadcast.splat = shufflevector <8 x i32*> %broadcast.splatinsert, <8 x i32*> undef, <8 x i32> zeroinitializer

  %gep.random = getelementptr i32, <8 x i32*> %broadcast.splat, <8 x i32> %ind
  %imask = bitcast i8 %mask to <8 x i1>
  %gt1 = call <8 x i32> @llvm.masked.gather.v8i32(<8 x i32*> %gep.random, i32 4, <8 x i1> %imask, <8 x i32>undef)
  %gt2 = call <8 x i32> @llvm.masked.gather.v8i32(<8 x i32*> %gep.random, i32 4, <8 x i1> %imask, <8 x i32>%gt1)
  %res = add <8 x i32> %gt1, %gt2
  ret <8 x i32> %res
}

; No uniform base in this case, index <8 x i64> contains addresses,
; each gather call will be split into two
; KNL-LABEL: test8
; KNL: kshiftrw        $8, %k1, %k2
; KNL: vpgatherqd
; KNL: vpgatherqd
; KNL: vinserti64x4
; KNL: vpgatherqd
; KNL: vpgatherqd
; KNL: vinserti64x4
define <16 x i32> @test8(<16 x i32*> %ptr.random, <16 x i32> %ind, i16 %mask) {
  %imask = bitcast i16 %mask to <16 x i1>
  %gt1 = call <16 x i32> @llvm.masked.gather.v16i32(<16 x i32*> %ptr.random, i32 4, <16 x i1> %imask, <16 x i32>undef)
  %gt2 = call <16 x i32> @llvm.masked.gather.v16i32(<16 x i32*> %ptr.random, i32 4, <16 x i1> %imask, <16 x i32>%gt1)
  %res = add <16 x i32> %gt1, %gt2
  ret <16 x i32> %res
}

%struct.RT = type { i8, [10 x [20 x i32]], i8 }
%struct.ST = type { i32, double, %struct.RT }

; Masked gather for agregate types
; Test9 and Test10 should give the same result (scalar and vector indices in GEP)

; KNL-LABEL: test9
; KNL: vpbroadcastq    %rdi, %zmm
; KNL: vpmovsxdq
; KNL: vpbroadcastq
; KNL: vpmuludq
; KNL: vpaddq
; KNL: vpaddq
; KNL: vpaddq
; KNL: vpaddq
; KNL: vpgatherqd      (,%zmm

define <8 x i32> @test9(%struct.ST* %base, <8 x i64> %ind1, <8 x i32>%ind5) {
entry:
  %broadcast.splatinsert = insertelement <8 x %struct.ST*> undef, %struct.ST* %base, i32 0
  %broadcast.splat = shufflevector <8 x %struct.ST*> %broadcast.splatinsert, <8 x %struct.ST*> undef, <8 x i32> zeroinitializer

  %arrayidx = getelementptr  %struct.ST, <8 x %struct.ST*> %broadcast.splat, <8 x i64> %ind1, <8 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>, <8 x i32><i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>, <8 x i32> %ind5, <8 x i64> <i64 13, i64 13, i64 13, i64 13, i64 13, i64 13, i64 13, i64 13>
  %res = call <8 x i32 >  @llvm.masked.gather.v8i32(<8 x i32*>%arrayidx, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x i32> undef)
  ret <8 x i32> %res
}

; KNL-LABEL: test10
; KNL: vpbroadcastq    %rdi, %zmm
; KNL: vpmovsxdq
; KNL: vpbroadcastq
; KNL: vpmuludq
; KNL: vpaddq
; KNL: vpaddq
; KNL: vpaddq
; KNL: vpaddq
; KNL: vpgatherqd      (,%zmm
define <8 x i32> @test10(%struct.ST* %base, <8 x i64> %i1, <8 x i32>%ind5) {
entry:
  %broadcast.splatinsert = insertelement <8 x %struct.ST*> undef, %struct.ST* %base, i32 0
  %broadcast.splat = shufflevector <8 x %struct.ST*> %broadcast.splatinsert, <8 x %struct.ST*> undef, <8 x i32> zeroinitializer

  %arrayidx = getelementptr  %struct.ST, <8 x %struct.ST*> %broadcast.splat, <8 x i64> %i1, i32 2, i32 1, <8 x i32> %ind5, i64 13
  %res = call <8 x i32 >  @llvm.masked.gather.v8i32(<8 x i32*>%arrayidx, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x i32> undef)
  ret <8 x i32> %res
}

; Splat index in GEP, requires broadcast
; KNL-LABEL: test11
; KNL: vpbroadcastd    %esi, %zmm
; KNL: vgatherdps      (%rdi,%zmm
define <16 x float> @test11(float* %base, i32 %ind) {

  %broadcast.splatinsert = insertelement <16 x float*> undef, float* %base, i32 0
  %broadcast.splat = shufflevector <16 x float*> %broadcast.splatinsert, <16 x float*> undef, <16 x i32> zeroinitializer

  %gep.random = getelementptr float, <16 x float*> %broadcast.splat, i32 %ind

  %res = call <16 x float> @llvm.masked.gather.v16f32(<16 x float*> %gep.random, i32 4, <16 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <16 x float> undef)
  ret <16 x float>%res
}

; We are checking the uniform base here. It is taken directly from input to vgatherdps
; KNL-LABEL: test12
; KNL: kxnorw  %k1, %k1, %k1
; KNL: vgatherdps      (%rdi,%zmm
define <16 x float> @test12(float* %base, <16 x i32> %ind) {

  %sext_ind = sext <16 x i32> %ind to <16 x i64>
  %gep.random = getelementptr float, float *%base, <16 x i64> %sext_ind

  %res = call <16 x float> @llvm.masked.gather.v16f32(<16 x float*> %gep.random, i32 4, <16 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <16 x float> undef)
  ret <16 x float>%res
}

; The same as the previous, but the mask is undefined
; KNL-LABEL: test13
; KNL-NOT: kxnorw
; KNL: vgatherdps      (%rdi,%zmm
define <16 x float> @test13(float* %base, <16 x i32> %ind) {

  %sext_ind = sext <16 x i32> %ind to <16 x i64>
  %gep.random = getelementptr float, float *%base, <16 x i64> %sext_ind

  %res = call <16 x float> @llvm.masked.gather.v16f32(<16 x float*> %gep.random, i32 4, <16 x i1> undef, <16 x float> undef)
  ret <16 x float>%res
}

; The base pointer is not splat, can't find unform base
; KNL-LABEL: test14
; KNL: vgatherqps      (,%zmm0)
; KNL: vgatherqps      (,%zmm0)
define <16 x float> @test14(float* %base, i32 %ind, <16 x float*> %vec) {

  %broadcast.splatinsert = insertelement <16 x float*> %vec, float* %base, i32 1
  %broadcast.splat = shufflevector <16 x float*> %broadcast.splatinsert, <16 x float*> undef, <16 x i32> zeroinitializer

  %gep.random = getelementptr float, <16 x float*> %broadcast.splat, i32 %ind

  %res = call <16 x float> @llvm.masked.gather.v16f32(<16 x float*> %gep.random, i32 4, <16 x i1> undef, <16 x float> undef)
  ret <16 x float>%res
}


