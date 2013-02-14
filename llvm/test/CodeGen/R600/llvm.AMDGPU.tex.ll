;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

;CHECK: TEX_SAMPLET{{[0-9]+\.XYZW, T[0-9]+\.XYZW}}, 0, 0, 1
;CHECK: TEX_SAMPLET{{[0-9]+\.XYZW, T[0-9]+\.XYZW}}, 0, 0, 2
;CHECK: TEX_SAMPLET{{[0-9]+\.XYZW, T[0-9]+\.XYZW}}, 0, 0, 3
;CHECK: TEX_SAMPLET{{[0-9]+\.XYZW, T[0-9]+\.XYZW}}, 0, 0, 4
;CHECK: TEX_SAMPLET{{[0-9]+\.XYZW, T[0-9]+\.XYZW}}, 0, 0, 5
;CHECK: TEX_SAMPLE_CT{{[0-9]+\.XYZW, T[0-9]+\.XYZW}}, 0, 0, 6
;CHECK: TEX_SAMPLE_CT{{[0-9]+\.XYZW, T[0-9]+\.XYZW}}, 0, 0, 7
;CHECK: TEX_SAMPLE_CT{{[0-9]+\.XYZW, T[0-9]+\.XYZW}}, 0, 0, 8
;CHECK: TEX_SAMPLET{{[0-9]+\.XYZW, T[0-9]+\.XYZW}}, 0, 0, 9
;CHECK: TEX_SAMPLET{{[0-9]+\.XYZW, T[0-9]+\.XYZW}}, 0, 0, 10
;CHECK: TEX_SAMPLE_CT{{[0-9]+\.XYZW, T[0-9]+\.XYZW}}, 0, 0, 11
;CHECK: TEX_SAMPLE_CT{{[0-9]+\.XYZW, T[0-9]+\.XYZW}}, 0, 0, 12
;CHECK: TEX_SAMPLE_CT{{[0-9]+\.XYZW, T[0-9]+\.XYZW}}, 0, 0, 13
;CHECK: TEX_SAMPLET{{[0-9]+\.XYZW, T[0-9]+\.XYZW}}, 0, 0, 14
;CHECK: TEX_SAMPLET{{[0-9]+\.XYZW, T[0-9]+\.XYZW}}, 0, 0, 15
;CHECK: TEX_SAMPLET{{[0-9]+\.XYZW, T[0-9]+\.XYZW}}, 0, 0, 16

define void @test(<4 x float> addrspace(1)* %out, <4 x float> addrspace(1)* %in) {
   %addr = load <4 x float> addrspace(1)* %in
   %res1 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %addr, i32 0, i32 0, i32 1)
   %res2 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %res1, i32 0, i32 0, i32 2)
   %res3 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %res2, i32 0, i32 0, i32 3)
   %res4 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %res3, i32 0, i32 0, i32 4)
   %res5 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %res4, i32 0, i32 0, i32 5)
   %res6 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %res5, i32 0, i32 0, i32 6)
   %res7 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %res6, i32 0, i32 0, i32 7)
   %res8 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %res7, i32 0, i32 0, i32 8)
   %res9 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %res8, i32 0, i32 0, i32 9)
   %res10 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %res9, i32 0, i32 0, i32 10)
   %res11 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %res10, i32 0, i32 0, i32 11)
   %res12 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %res11, i32 0, i32 0, i32 12)
   %res13 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %res12, i32 0, i32 0, i32 13)
   %res14 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %res13, i32 0, i32 0, i32 14)
   %res15 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %res14, i32 0, i32 0, i32 15)
   %res16 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %res15, i32 0, i32 0, i32 16)
   store <4 x float> %res16, <4 x float> addrspace(1)* %out
   ret void
}

declare <4 x float> @llvm.AMDGPU.tex(<4 x float>, i32, i32, i32) readnone
