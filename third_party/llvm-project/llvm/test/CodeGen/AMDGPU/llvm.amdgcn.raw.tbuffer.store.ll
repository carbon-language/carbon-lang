;RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck -check-prefixes=GCN,PREGFX10 %s
;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck -check-prefixes=GCN,PREGFX10 %s
;RUN: llc < %s -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs | FileCheck -check-prefixes=GCN,GFX10 %s

; GCN-LABEL: {{^}}tbuffer_store:
; PREGFX10: tbuffer_store_format_xyzw v[0:3], off, s[0:3], 0 format:[BUF_DATA_FORMAT_16_16_16_16,BUF_NUM_FORMAT_USCALED]
; PREGFX10: tbuffer_store_format_xyzw v[4:7], off, s[0:3], 0 format:[BUF_DATA_FORMAT_32_32_32,BUF_NUM_FORMAT_SSCALED] glc
; PREGFX10: tbuffer_store_format_xyzw v[8:11], off, s[0:3], 0 format:[BUF_DATA_FORMAT_32_32_32_32,BUF_NUM_FORMAT_UINT] slc
; PREGFX10: tbuffer_store_format_xyzw v[8:11], off, s[0:3], 0 format:[BUF_DATA_FORMAT_32_32_32_32,BUF_NUM_FORMAT_UINT] glc
; GFX10: tbuffer_store_format_xyzw v[0:3], off, s[0:3], 0 format:[BUF_FMT_10_10_10_2_UNORM]
; GFX10: tbuffer_store_format_xyzw v[4:7], off, s[0:3], 0 format:[BUF_FMT_8_8_8_8_SINT] glc
; GFX10: tbuffer_store_format_xyzw v[8:11], off, s[0:3], 0 format:78 slc
; GFX10: tbuffer_store_format_xyzw v[8:11], off, s[0:3], 0 format:78 glc dlc
define amdgpu_ps void @tbuffer_store(<4 x i32> inreg, <4 x float>, <4 x float>, <4 x float>) {
main_body:
  %in1 = bitcast <4 x float> %1 to <4 x i32>
  %in2 = bitcast <4 x float> %2 to <4 x i32>
  %in3 = bitcast <4 x float> %3 to <4 x i32>
  call void @llvm.amdgcn.raw.tbuffer.store.v4i32(<4 x i32> %in1, <4 x i32> %0, i32 0, i32 0, i32 44, i32 0)
  call void @llvm.amdgcn.raw.tbuffer.store.v4i32(<4 x i32> %in2, <4 x i32> %0, i32 0, i32 0, i32 61, i32 1)
  call void @llvm.amdgcn.raw.tbuffer.store.v4i32(<4 x i32> %in3, <4 x i32> %0, i32 0, i32 0, i32 78, i32 2)
  call void @llvm.amdgcn.raw.tbuffer.store.v4f32(<4 x float> %3, <4 x i32> %0, i32 0, i32 0, i32 78, i32 5)
  ret void
}

; GCN-LABEL: {{^}}tbuffer_store_immoffs:
; PREGFX10: tbuffer_store_format_xyzw v[0:3], off, s[0:3], 0 format:[BUF_DATA_FORMAT_16_16,BUF_NUM_FORMAT_FLOAT] offset:42
; GFX10: tbuffer_store_format_xyzw v[0:3], off, s[0:3], 0 format:117 offset:42
define amdgpu_ps void @tbuffer_store_immoffs(<4 x i32> inreg, <4 x float>) {
main_body:
  %in1 = bitcast <4 x float> %1 to <4 x i32>
  call void @llvm.amdgcn.raw.tbuffer.store.v4i32(<4 x i32> %in1, <4 x i32> %0, i32 42, i32 0, i32 117, i32 0)
  ret void
}

; GCN-LABEL: {{^}}tbuffer_store_scalar_and_imm_offs:
; PREGFX10: tbuffer_store_format_xyzw v[0:3], off, s[0:3], {{s[0-9]+}} format:[BUF_DATA_FORMAT_16_16,BUF_NUM_FORMAT_FLOAT] offset:42
; GFX10: tbuffer_store_format_xyzw v[0:3], off, s[0:3], {{s[0-9]+}} format:117 offset:42
define amdgpu_ps void @tbuffer_store_scalar_and_imm_offs(<4 x i32> inreg, <4 x float> %vdata, i32 inreg %soffset) {
main_body:
  %in1 = bitcast <4 x float> %vdata to <4 x i32>
  call void @llvm.amdgcn.raw.tbuffer.store.v4i32(<4 x i32> %in1, <4 x i32> %0, i32 42, i32 %soffset, i32 117, i32 0)
  ret void
}

; GCN-LABEL: {{^}}buffer_store_ofs:
; PREGFX10: tbuffer_store_format_xyzw v[0:3], v4, s[0:3], 0 format:[BUF_DATA_FORMAT_8_8,BUF_NUM_FORMAT_FLOAT] offen
; GFX10: tbuffer_store_format_xyzw v[0:3], v4, s[0:3], 0 format:115 offen
define amdgpu_ps void @buffer_store_ofs(<4 x i32> inreg, <4 x float> %vdata, i32 %voffset) {
main_body:
  %in1 = bitcast <4 x float> %vdata to <4 x i32>
  call void @llvm.amdgcn.raw.tbuffer.store.v4i32(<4 x i32> %in1, <4 x i32> %0, i32 %voffset, i32 0, i32 115, i32 0)
  ret void
}

; GCN-LABEL: {{^}}buffer_store_x1:
; PREGFX10: tbuffer_store_format_x v0, off, s[0:3], 0 format:[BUF_DATA_FORMAT_32_32_32,BUF_NUM_FORMAT_FLOAT]
; GFX10: tbuffer_store_format_x v0, off, s[0:3], 0 format:125
define amdgpu_ps void @buffer_store_x1(<4 x i32> inreg %rsrc, float %data) {
main_body:
  %data.i = bitcast float %data to i32
  call void @llvm.amdgcn.raw.tbuffer.store.i32(i32 %data.i, <4 x i32> %rsrc, i32 0, i32 0, i32 125, i32 0)
  ret void
}

; GCN-LABEL: {{^}}buffer_store_x2:
; PREGFX10: tbuffer_store_format_xy v[0:1], off, s[0:3], 0 format:[BUF_NUM_FORMAT_USCALED]
; GFX10: tbuffer_store_format_xy v[0:1], off, s[0:3], 0 format:[BUF_FMT_10_11_11_SSCALED]
define amdgpu_ps void @buffer_store_x2(<4 x i32> inreg %rsrc, <2 x float> %data) {
main_body:
  %data.i = bitcast <2 x float> %data to <2 x i32>
  call void @llvm.amdgcn.raw.tbuffer.store.v2i32(<2 x i32> %data.i, <4 x i32> %rsrc, i32 0, i32 0, i32 33, i32 0)
  ret void
}

declare void @llvm.amdgcn.raw.tbuffer.store.i32(i32, <4 x i32>, i32, i32, i32, i32) #0
declare void @llvm.amdgcn.raw.tbuffer.store.v2i32(<2 x i32>, <4 x i32>, i32, i32, i32, i32) #0
declare void @llvm.amdgcn.raw.tbuffer.store.v4i32(<4 x i32>, <4 x i32>, i32, i32, i32, i32) #0
declare void @llvm.amdgcn.raw.tbuffer.store.v4f32(<4 x float>, <4 x i32>, i32, i32, i32, i32) #0

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }
