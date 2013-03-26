;RUN: llc < %s -march=r600 -mcpu=SI | FileCheck %s

;CHECK: IMAGE_SAMPLE
;CHECK: IMAGE_SAMPLE
;CHECK: IMAGE_SAMPLE
;CHECK: IMAGE_SAMPLE
;CHECK: IMAGE_SAMPLE
;CHECK: IMAGE_SAMPLE_C
;CHECK: IMAGE_SAMPLE_C
;CHECK: IMAGE_SAMPLE_C
;CHECK: IMAGE_SAMPLE
;CHECK: IMAGE_SAMPLE
;CHECK: IMAGE_SAMPLE_C
;CHECK: IMAGE_SAMPLE_C
;CHECK: IMAGE_SAMPLE_C
;CHECK: IMAGE_SAMPLE
;CHECK: IMAGE_SAMPLE
;CHECK: IMAGE_SAMPLE

define void @test(i32 %a1, i32 %a2, i32 %a3, i32 %a4) {
   %v1 = insertelement <4 x i32> undef, i32 %a1, i32 0
   %v2 = insertelement <4 x i32> undef, i32 %a1, i32 1
   %v3 = insertelement <4 x i32> undef, i32 %a1, i32 2
   %v4 = insertelement <4 x i32> undef, i32 %a1, i32 3
   %v5 = insertelement <4 x i32> undef, i32 %a2, i32 0
   %v6 = insertelement <4 x i32> undef, i32 %a2, i32 1
   %v7 = insertelement <4 x i32> undef, i32 %a2, i32 2
   %v8 = insertelement <4 x i32> undef, i32 %a2, i32 3
   %v9 = insertelement <4 x i32> undef, i32 %a3, i32 0
   %v10 = insertelement <4 x i32> undef, i32 %a3, i32 1
   %v11 = insertelement <4 x i32> undef, i32 %a3, i32 2
   %v12 = insertelement <4 x i32> undef, i32 %a3, i32 3
   %v13 = insertelement <4 x i32> undef, i32 %a4, i32 0
   %v14 = insertelement <4 x i32> undef, i32 %a4, i32 1
   %v15 = insertelement <4 x i32> undef, i32 %a4, i32 2
   %v16 = insertelement <4 x i32> undef, i32 %a4, i32 3
   %res1 = call <4 x float> @llvm.SI.sample.(i32 15, <4 x i32> %v1,
      <8 x i32> undef, <4 x i32> undef, i32 1)
   %res2 = call <4 x float> @llvm.SI.sample.(i32 15, <4 x i32> %v2,
      <8 x i32> undef, <4 x i32> undef, i32 2)
   %res3 = call <4 x float> @llvm.SI.sample.(i32 15, <4 x i32> %v3,
      <8 x i32> undef, <4 x i32> undef, i32 3)
   %res4 = call <4 x float> @llvm.SI.sample.(i32 15, <4 x i32> %v4,
      <8 x i32> undef, <4 x i32> undef, i32 4)
   %res5 = call <4 x float> @llvm.SI.sample.(i32 15, <4 x i32> %v5,
      <8 x i32> undef, <4 x i32> undef, i32 5)
   %res6 = call <4 x float> @llvm.SI.sample.(i32 15, <4 x i32> %v6,
      <8 x i32> undef, <4 x i32> undef, i32 6)
   %res7 = call <4 x float> @llvm.SI.sample.(i32 15, <4 x i32> %v7,
      <8 x i32> undef, <4 x i32> undef, i32 7)
   %res8 = call <4 x float> @llvm.SI.sample.(i32 15, <4 x i32> %v8,
      <8 x i32> undef, <4 x i32> undef, i32 8)
   %res9 = call <4 x float> @llvm.SI.sample.(i32 15, <4 x i32> %v9,
      <8 x i32> undef, <4 x i32> undef, i32 9)
   %res10 = call <4 x float> @llvm.SI.sample.(i32 15, <4 x i32> %v10,
      <8 x i32> undef, <4 x i32> undef, i32 10)
   %res11 = call <4 x float> @llvm.SI.sample.(i32 15, <4 x i32> %v11,
      <8 x i32> undef, <4 x i32> undef, i32 11)
   %res12 = call <4 x float> @llvm.SI.sample.(i32 15, <4 x i32> %v12,
      <8 x i32> undef, <4 x i32> undef, i32 12)
   %res13 = call <4 x float> @llvm.SI.sample.(i32 15, <4 x i32> %v13,
      <8 x i32> undef, <4 x i32> undef, i32 13)
   %res14 = call <4 x float> @llvm.SI.sample.(i32 15, <4 x i32> %v14,
      <8 x i32> undef, <4 x i32> undef, i32 14)
   %res15 = call <4 x float> @llvm.SI.sample.(i32 15, <4 x i32> %v15,
      <8 x i32> undef, <4 x i32> undef, i32 15)
   %res16 = call <4 x float> @llvm.SI.sample.(i32 15, <4 x i32> %v16,
      <8 x i32> undef, <4 x i32> undef, i32 16)
   %e1 = extractelement <4 x float> %res1, i32 0
   %e2 = extractelement <4 x float> %res2, i32 0
   %e3 = extractelement <4 x float> %res3, i32 0
   %e4 = extractelement <4 x float> %res4, i32 0
   %e5 = extractelement <4 x float> %res5, i32 0
   %e6 = extractelement <4 x float> %res6, i32 0
   %e7 = extractelement <4 x float> %res7, i32 0
   %e8 = extractelement <4 x float> %res8, i32 0
   %e9 = extractelement <4 x float> %res9, i32 0
   %e10 = extractelement <4 x float> %res10, i32 0
   %e11 = extractelement <4 x float> %res11, i32 0
   %e12 = extractelement <4 x float> %res12, i32 0
   %e13 = extractelement <4 x float> %res13, i32 0
   %e14 = extractelement <4 x float> %res14, i32 0
   %e15 = extractelement <4 x float> %res15, i32 0
   %e16 = extractelement <4 x float> %res16, i32 0
   %s1 = fadd float %e1, %e2
   %s2 = fadd float %s1, %e3
   %s3 = fadd float %s2, %e4
   %s4 = fadd float %s3, %e5
   %s5 = fadd float %s4, %e6
   %s6 = fadd float %s5, %e7
   %s7 = fadd float %s6, %e8
   %s8 = fadd float %s7, %e9
   %s9 = fadd float %s8, %e10
   %s10 = fadd float %s9, %e11
   %s11 = fadd float %s10, %e12
   %s12 = fadd float %s11, %e13
   %s13 = fadd float %s12, %e14
   %s14 = fadd float %s13, %e15
   %s15 = fadd float %s14, %e16
   call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 12, i32 0, float %s15, float %s15, float %s15, float %s15)
   ret void
}

declare <4 x float> @llvm.SI.sample.(i32, <4 x i32>, <8 x i32>, <4 x i32>, i32) readnone

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)
