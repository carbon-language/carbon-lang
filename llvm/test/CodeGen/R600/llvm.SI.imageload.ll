;RUN: llc < %s -march=r600 -mcpu=verde | FileCheck %s

;CHECK-DAG: IMAGE_LOAD_MIP {{VGPR[0-9]+_VGPR[0-9]+_VGPR[0-9]+_VGPR[0-9]+}}, 15, 0, 0, -1
;CHECK-DAG: IMAGE_LOAD_MIP {{VGPR[0-9]+_VGPR[0-9]+}}, 3, 0, 0, 0
;CHECK-DAG: IMAGE_LOAD_MIP {{VGPR[0-9]+}}, 2, 0, 0, 0
;CHECK-DAG: IMAGE_LOAD_MIP {{VGPR[0-9]+}}, 1, 0, 0, 0
;CHECK-DAG: IMAGE_LOAD_MIP {{VGPR[0-9]+}}, 4, 0, 0, 0
;CHECK-DAG: IMAGE_LOAD_MIP {{VGPR[0-9]+}}, 8, 0, 0, 0
;CHECK-DAG: IMAGE_LOAD_MIP {{VGPR[0-9]+_VGPR[0-9]+}}, 5, 0, 0, 0
;CHECK-DAG: IMAGE_LOAD_MIP {{VGPR[0-9]+_VGPR[0-9]+}}, 12, 0, 0, -1
;CHECK-DAG: IMAGE_LOAD_MIP {{VGPR[0-9]+_VGPR[0-9]+_VGPR[0-9]+}}, 7, 0, 0, 0
;CHECK-DAG: IMAGE_LOAD_MIP {{VGPR[0-9]+}}, 8, 0, 0, -1

define void @test(i32 %a1, i32 %a2, i32 %a3, i32 %a4) {
   %v1 = insertelement <4 x i32> undef, i32 %a1, i32 0
   %v2 = insertelement <4 x i32> undef, i32 %a1, i32 1
   %v3 = insertelement <4 x i32> undef, i32 %a1, i32 2
   %v4 = insertelement <4 x i32> undef, i32 %a1, i32 3
   %v5 = insertelement <4 x i32> undef, i32 %a2, i32 0
   %v6 = insertelement <4 x i32> undef, i32 %a2, i32 1
   %v10 = insertelement <4 x i32> undef, i32 %a3, i32 1
   %v11 = insertelement <4 x i32> undef, i32 %a3, i32 2
   %v15 = insertelement <4 x i32> undef, i32 %a4, i32 2
   %v16 = insertelement <4 x i32> undef, i32 %a4, i32 3
   %res1 = call <4 x i32> @llvm.SI.imageload.(<4 x i32> %v1,
      <8 x i32> undef, i32 1)
   %res2 = call <4 x i32> @llvm.SI.imageload.(<4 x i32> %v2,
      <8 x i32> undef, i32 2)
   %res3 = call <4 x i32> @llvm.SI.imageload.(<4 x i32> %v3,
      <8 x i32> undef, i32 3)
   %res4 = call <4 x i32> @llvm.SI.imageload.(<4 x i32> %v4,
      <8 x i32> undef, i32 4)
   %res5 = call <4 x i32> @llvm.SI.imageload.(<4 x i32> %v5,
      <8 x i32> undef, i32 5)
   %res6 = call <4 x i32> @llvm.SI.imageload.(<4 x i32> %v6,
      <8 x i32> undef, i32 6)
   %res10 = call <4 x i32> @llvm.SI.imageload.(<4 x i32> %v10,
      <8 x i32> undef, i32 10)
   %res11 = call <4 x i32> @llvm.SI.imageload.(<4 x i32> %v11,
      <8 x i32> undef, i32 11)
   %res15 = call <4 x i32> @llvm.SI.imageload.(<4 x i32> %v15,
      <8 x i32> undef, i32 15)
   %res16 = call <4 x i32> @llvm.SI.imageload.(<4 x i32> %v16,
      <8 x i32> undef, i32 16)
   %e1 = extractelement <4 x i32> %res1, i32 0
   %e2 = extractelement <4 x i32> %res2, i32 1
   %e3 = extractelement <4 x i32> %res3, i32 2
   %e4 = extractelement <4 x i32> %res4, i32 3
   %t0 = extractelement <4 x i32> %res5, i32 0
   %t1 = extractelement <4 x i32> %res5, i32 1
   %e5 = add i32 %t0, %t1
   %t2 = extractelement <4 x i32> %res6, i32 0
   %t3 = extractelement <4 x i32> %res6, i32 2
   %e6 = add i32 %t2, %t3
   %t10 = extractelement <4 x i32> %res10, i32 2
   %t11 = extractelement <4 x i32> %res10, i32 3
   %e10 = add i32 %t10, %t11
   %t12 = extractelement <4 x i32> %res11, i32 0
   %t13 = extractelement <4 x i32> %res11, i32 1
   %t14 = extractelement <4 x i32> %res11, i32 2
   %t15 = add i32 %t12, %t13
   %e11 = add i32 %t14, %t15
   %t28 = extractelement <4 x i32> %res15, i32 0
   %t29 = extractelement <4 x i32> %res15, i32 1
   %t30 = extractelement <4 x i32> %res15, i32 2
   %t31 = extractelement <4 x i32> %res15, i32 3
   %t32 = add i32 %t28, %t29
   %t33 = add i32 %t30, %t31
   %e15 = add i32 %t32, %t33
   %e16 = extractelement <4 x i32> %res16, i32 3
   %s1 = add i32 %e1, %e2
   %s2 = add i32 %s1, %e3
   %s3 = add i32 %s2, %e4
   %s4 = add i32 %s3, %e5
   %s5 = add i32 %s4, %e6
   %s9 = add i32 %s5, %e10
   %s10 = add i32 %s9, %e11
   %s14 = add i32 %s10, %e15
   %s15 = add i32 %s14, %e16
   %s16 = bitcast i32 %s15 to float
   call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 12, i32 0, float %s16, float %s16, float %s16, float %s16)
   ret void
}

declare <4 x i32> @llvm.SI.imageload.(<4 x i32>, <8 x i32>, i32) readnone

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)
