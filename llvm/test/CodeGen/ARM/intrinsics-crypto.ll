; RUN: llc < %s -mtriple=armv8 -mattr=+crypto | FileCheck %s

define arm_aapcs_vfpcc <16 x i8> @test_aesde(<16 x i8>* %a, <16 x i8> *%b) {
  %tmp = load <16 x i8>* %a
  %tmp2 = load <16 x i8>* %b
  %tmp3 = call <16 x i8> @llvm.arm.neon.aesd.v16i8(<16 x i8> %tmp, <16 x i8> %tmp2)
  ; CHECK: aesd.8 q{{[0-9]+}}, q{{[0-9]+}}
  %tmp4 = call <16 x i8> @llvm.arm.neon.aese.v16i8(<16 x i8> %tmp3, <16 x i8> %tmp2)
  ; CHECK: aese.8 q{{[0-9]+}}, q{{[0-9]+}}
  %tmp5 = call <16 x i8> @llvm.arm.neon.aesimc.v16i8(<16 x i8> %tmp4)
  ; CHECK: aesimc.8 q{{[0-9]+}}, q{{[0-9]+}}
  %tmp6 = call <16 x i8> @llvm.arm.neon.aesmc.v16i8(<16 x i8> %tmp5)
  ; CHECK: aesmc.8 q{{[0-9]+}}, q{{[0-9]+}}
  ret <16 x i8> %tmp6
}

define arm_aapcs_vfpcc <4 x i32> @test_sha(<4 x i32> *%a, <4 x i32> *%b, <4 x i32> *%c) {
  %tmp = load <4 x i32>* %a
  %tmp2 = load <4 x i32>* %b
  %tmp3 = load <4 x i32>* %c
  %res1 = call <4 x i32> @llvm.arm.neon.sha1h.v4i32(<4 x i32> %tmp)
  ; CHECK: sha1h.32 q{{[0-9]+}}, q{{[0-9]+}}
  %res2 = call <4 x i32> @llvm.arm.neon.sha1c.v4i32(<4 x i32> %tmp2, <4 x i32> %tmp3, <4 x i32> %res1)
  ; CHECK: sha1c.32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
  %res3 = call <4 x i32> @llvm.arm.neon.sha1m.v4i32(<4 x i32> %res2, <4 x i32> %tmp3, <4 x i32> %res1)
  ; CHECK: sha1m.32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
  %res4 = call <4 x i32> @llvm.arm.neon.sha1p.v4i32(<4 x i32> %res3, <4 x i32> %tmp3, <4 x i32> %res1)
  ; CHECK: sha1p.32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
  %res5 = call <4 x i32> @llvm.arm.neon.sha1su0.v4i32(<4 x i32> %res4, <4 x i32> %tmp3, <4 x i32> %res1)
  ; CHECK: sha1su0.32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
  %res6 = call <4 x i32> @llvm.arm.neon.sha1su1.v4i32(<4 x i32> %res5, <4 x i32> %res1)
  ; CHECK: sha1su1.32 q{{[0-9]+}}, q{{[0-9]+}}
  %res7 = call <4 x i32> @llvm.arm.neon.sha256h.v4i32(<4 x i32> %res6, <4 x i32> %tmp3, <4 x i32> %res1)
  ; CHECK: sha256h.32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
  %res8 = call <4 x i32> @llvm.arm.neon.sha256h2.v4i32(<4 x i32> %res7, <4 x i32> %tmp3, <4 x i32> %res1)
  ; CHECK: sha256h2.32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
  %res9 = call <4 x i32> @llvm.arm.neon.sha256su1.v4i32(<4 x i32> %res8, <4 x i32> %tmp3, <4 x i32> %res1)
  ; CHECK: sha256su1.32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
  %res10 = call <4 x i32> @llvm.arm.neon.sha256su0.v4i32(<4 x i32> %res9, <4 x i32> %tmp3)
  ; CHECK: sha256su0.32 q{{[0-9]+}}, q{{[0-9]+}}
  ret <4 x i32> %res10
}

declare <16 x i8> @llvm.arm.neon.aesd.v16i8(<16 x i8>, <16 x i8>)
declare <16 x i8> @llvm.arm.neon.aese.v16i8(<16 x i8>, <16 x i8>)
declare <16 x i8> @llvm.arm.neon.aesimc.v16i8(<16 x i8>)
declare <16 x i8> @llvm.arm.neon.aesmc.v16i8(<16 x i8>)
declare <4 x i32> @llvm.arm.neon.sha1h.v4i32(<4 x i32>)
declare <4 x i32> @llvm.arm.neon.sha1c.v4i32(<4 x i32>, <4 x i32>, <4 x i32>)
declare <4 x i32> @llvm.arm.neon.sha1m.v4i32(<4 x i32>, <4 x i32>, <4 x i32>)
declare <4 x i32> @llvm.arm.neon.sha1p.v4i32(<4 x i32>, <4 x i32>, <4 x i32>)
declare <4 x i32> @llvm.arm.neon.sha1su0.v4i32(<4 x i32>, <4 x i32>, <4 x i32>)
declare <4 x i32> @llvm.arm.neon.sha256h.v4i32(<4 x i32>, <4 x i32>, <4 x i32>)
declare <4 x i32> @llvm.arm.neon.sha256h2.v4i32(<4 x i32>, <4 x i32>, <4 x i32>)
declare <4 x i32> @llvm.arm.neon.sha256su1.v4i32(<4 x i32>, <4 x i32>, <4 x i32>)
declare <4 x i32> @llvm.arm.neon.sha256su0.v4i32(<4 x i32>, <4 x i32>)
declare <4 x i32> @llvm.arm.neon.sha1su1.v4i32(<4 x i32>, <4 x i32>)
