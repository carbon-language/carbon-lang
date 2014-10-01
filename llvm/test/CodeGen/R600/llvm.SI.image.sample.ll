;RUN: llc < %s -march=r600 -mcpu=verde -verify-machineinstrs | FileCheck %s

;CHECK-LABEL: {{^}}sample:
;CHECK: IMAGE_SAMPLE {{v\[[0-9]+:[0-9]+\]}}, 15, 0, 0, 0, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @sample() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.image.sample.v4i32(<4 x i32> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}sample_cl:
;CHECK: IMAGE_SAMPLE_CL {{v\[[0-9]+:[0-9]+\]}}, 15, 0, 0, 0, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @sample_cl() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.image.sample.cl.v4i32(<4 x i32> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}sample_d:
;CHECK: IMAGE_SAMPLE_D {{v\[[0-9]+:[0-9]+\]}}, 15, 0, 0, 0, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @sample_d() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.image.sample.d.v4i32(<4 x i32> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}sample_d_cl:
;CHECK: IMAGE_SAMPLE_D_CL {{v\[[0-9]+:[0-9]+\]}}, 15, 0, 0, 0, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @sample_d_cl() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.image.sample.d.cl.v4i32(<4 x i32> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}sample_l:
;CHECK: IMAGE_SAMPLE_L {{v\[[0-9]+:[0-9]+\]}}, 15, 0, 0, 0, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @sample_l() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.image.sample.l.v4i32(<4 x i32> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}sample_b:
;CHECK: IMAGE_SAMPLE_B {{v\[[0-9]+:[0-9]+\]}}, 15, 0, 0, 0, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @sample_b() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.image.sample.b.v4i32(<4 x i32> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}sample_b_cl:
;CHECK: IMAGE_SAMPLE_B_CL {{v\[[0-9]+:[0-9]+\]}}, 15, 0, 0, 0, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @sample_b_cl() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.image.sample.b.cl.v4i32(<4 x i32> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}sample_lz:
;CHECK: IMAGE_SAMPLE_LZ {{v\[[0-9]+:[0-9]+\]}}, 15, 0, 0, 0, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @sample_lz() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.image.sample.lz.v4i32(<4 x i32> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}sample_cd:
;CHECK: IMAGE_SAMPLE_CD {{v\[[0-9]+:[0-9]+\]}}, 15, 0, 0, 0, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @sample_cd() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.image.sample.cd.v4i32(<4 x i32> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}sample_cd_cl:
;CHECK: IMAGE_SAMPLE_CD_CL {{v\[[0-9]+:[0-9]+\]}}, 15, 0, 0, 0, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @sample_cd_cl() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.image.sample.cd.cl.v4i32(<4 x i32> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}sample_c:
;CHECK: IMAGE_SAMPLE_C {{v\[[0-9]+:[0-9]+\]}}, 15, 0, 0, 0, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @sample_c() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.image.sample.c.v4i32(<4 x i32> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}sample_c_cl:
;CHECK: IMAGE_SAMPLE_C_CL {{v\[[0-9]+:[0-9]+\]}}, 15, 0, 0, 0, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @sample_c_cl() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.image.sample.c.cl.v4i32(<4 x i32> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}sample_c_d:
;CHECK: IMAGE_SAMPLE_C_D {{v\[[0-9]+:[0-9]+\]}}, 15, 0, 0, 0, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @sample_c_d() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.image.sample.c.d.v4i32(<4 x i32> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}sample_c_d_cl:
;CHECK: IMAGE_SAMPLE_C_D_CL {{v\[[0-9]+:[0-9]+\]}}, 15, 0, 0, 0, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @sample_c_d_cl() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.image.sample.c.d.cl.v4i32(<4 x i32> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}sample_c_l:
;CHECK: IMAGE_SAMPLE_C_L {{v\[[0-9]+:[0-9]+\]}}, 15, 0, 0, 0, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @sample_c_l() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.image.sample.c.l.v4i32(<4 x i32> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}sample_c_b:
;CHECK: IMAGE_SAMPLE_C_B {{v\[[0-9]+:[0-9]+\]}}, 15, 0, 0, 0, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @sample_c_b() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.image.sample.c.b.v4i32(<4 x i32> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}sample_c_b_cl:
;CHECK: IMAGE_SAMPLE_C_B_CL {{v\[[0-9]+:[0-9]+\]}}, 15, 0, 0, 0, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @sample_c_b_cl() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.image.sample.c.b.cl.v4i32(<4 x i32> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}sample_c_lz:
;CHECK: IMAGE_SAMPLE_C_LZ {{v\[[0-9]+:[0-9]+\]}}, 15, 0, 0, 0, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @sample_c_lz() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.image.sample.c.lz.v4i32(<4 x i32> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}sample_c_cd:
;CHECK: IMAGE_SAMPLE_C_CD {{v\[[0-9]+:[0-9]+\]}}, 15, 0, 0, 0, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @sample_c_cd() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.image.sample.c.cd.v4i32(<4 x i32> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}

;CHECK-LABEL: {{^}}sample_c_cd_cl:
;CHECK: IMAGE_SAMPLE_C_CD_CL {{v\[[0-9]+:[0-9]+\]}}, 15, 0, 0, 0, 0, 0, 0, 0, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @sample_c_cd_cl() #0 {
main_body:
  %r = call <4 x float> @llvm.SI.image.sample.c.cd.cl.v4i32(<4 x i32> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %r0 = extractelement <4 x float> %r, i32 0
  %r1 = extractelement <4 x float> %r, i32 1
  %r2 = extractelement <4 x float> %r, i32 2
  %r3 = extractelement <4 x float> %r, i32 3
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %r0, float %r1, float %r2, float %r3)
  ret void
}


declare <4 x float> @llvm.SI.image.sample.v4i32(<4 x i32>, <8 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.image.sample.cl.v4i32(<4 x i32>, <8 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.image.sample.d.v4i32(<4 x i32>, <8 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.image.sample.d.cl.v4i32(<4 x i32>, <8 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.image.sample.l.v4i32(<4 x i32>, <8 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.image.sample.b.v4i32(<4 x i32>, <8 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.image.sample.b.cl.v4i32(<4 x i32>, <8 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.image.sample.lz.v4i32(<4 x i32>, <8 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.image.sample.cd.v4i32(<4 x i32>, <8 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.image.sample.cd.cl.v4i32(<4 x i32>, <8 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #1

declare <4 x float> @llvm.SI.image.sample.c.v4i32(<4 x i32>, <8 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.image.sample.c.cl.v4i32(<4 x i32>, <8 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.image.sample.c.d.v4i32(<4 x i32>, <8 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.image.sample.c.d.cl.v4i32(<4 x i32>, <8 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.image.sample.c.l.v4i32(<4 x i32>, <8 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.image.sample.c.b.v4i32(<4 x i32>, <8 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.image.sample.c.b.cl.v4i32(<4 x i32>, <8 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.image.sample.c.lz.v4i32(<4 x i32>, <8 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.image.sample.c.cd.v4i32(<4 x i32>, <8 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare <4 x float> @llvm.SI.image.sample.c.cd.cl.v4i32(<4 x i32>, <8 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #1

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #0 = { "ShaderType"="0" }
attributes #1 = { nounwind readnone }
