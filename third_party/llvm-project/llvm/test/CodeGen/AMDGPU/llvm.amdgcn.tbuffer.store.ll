;RUN: llc -global-isel=0 < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck -check-prefixes=GCN,VERDE %s
;RUN: llc -global-isel=0 < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}tbuffer_store:
; GCN: tbuffer_store_format_xyzw v[0:3], off, s[0:3], 0 format:[BUF_DATA_FORMAT_16_16_16_16,BUF_NUM_FORMAT_USCALED]
; GCN: tbuffer_store_format_xyzw v[4:7], off, s[0:3], 0 format:[BUF_DATA_FORMAT_32_32_32,BUF_NUM_FORMAT_SSCALED] glc
; GCN: tbuffer_store_format_xyzw v[8:11], off, s[0:3], 0 format:[BUF_DATA_FORMAT_32_32_32_32,BUF_NUM_FORMAT_UINT] slc
; GCN: tbuffer_store_format_xyzw v[8:11], off, s[0:3], 0 format:[BUF_DATA_FORMAT_32_32_32_32,BUF_NUM_FORMAT_UINT]
define amdgpu_ps void @tbuffer_store(<4 x i32> inreg, <4 x float>, <4 x float>, <4 x float>) {
main_body:
  %in1 = bitcast <4 x float> %1 to <4 x i32>
  %in2 = bitcast <4 x float> %2 to <4 x i32>
  %in3 = bitcast <4 x float> %3 to <4 x i32>
  call void @llvm.amdgcn.tbuffer.store.v4i32(<4 x i32> %in1, <4 x i32> %0, i32 0, i32 0, i32 0, i32 0, i32 12, i32 2, i1 0, i1 0)
  call void @llvm.amdgcn.tbuffer.store.v4i32(<4 x i32> %in2, <4 x i32> %0, i32 0, i32 0, i32 0, i32 0, i32 13, i32 3, i1 1, i1 0)
  call void @llvm.amdgcn.tbuffer.store.v4i32(<4 x i32> %in3, <4 x i32> %0, i32 0, i32 0, i32 0, i32 0, i32 14, i32 4, i1 0, i1 1)
  call void @llvm.amdgcn.tbuffer.store.v4f32(<4 x float> %3, <4 x i32> %0, i32 0, i32 0, i32 0, i32 0, i32 14, i32 4, i1 0, i1 0)
  ret void
}

; GCN-LABEL: {{^}}tbuffer_store_immoffs:
; GCN: tbuffer_store_format_xyzw v[0:3], off, s[0:3], 0 format:[BUF_DATA_FORMAT_16_16,BUF_NUM_FORMAT_FLOAT] offset:42
define amdgpu_ps void @tbuffer_store_immoffs(<4 x i32> inreg, <4 x float>) {
main_body:
  %in1 = bitcast <4 x float> %1 to <4 x i32>
  call void @llvm.amdgcn.tbuffer.store.v4i32(<4 x i32> %in1, <4 x i32> %0, i32 0, i32 0, i32 0, i32 42, i32 5, i32 7, i1 0, i1 0)
  ret void
}

; GCN-LABEL: {{^}}tbuffer_store_scalar_and_imm_offs:
; GCN: tbuffer_store_format_xyzw v[0:3], off, s[0:3], {{s[0-9]+}} format:[BUF_DATA_FORMAT_16_16,BUF_NUM_FORMAT_FLOAT] offset:42
define amdgpu_ps void @tbuffer_store_scalar_and_imm_offs(<4 x i32> inreg, <4 x float> %vdata, i32 inreg %soffset) {
main_body:
  %in1 = bitcast <4 x float> %vdata to <4 x i32>
  call void @llvm.amdgcn.tbuffer.store.v4i32(<4 x i32> %in1, <4 x i32> %0, i32 0, i32 0, i32 %soffset, i32 42, i32 5, i32 7, i1 0, i1 0)
  ret void
}

; GCN-LABEL: {{^}}buffer_store_idx:
; GCN: tbuffer_store_format_xyzw v[0:3], v4, s[0:3], 0 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_USCALED] idxen
define amdgpu_ps void @buffer_store_idx(<4 x i32> inreg, <4 x float> %vdata, i32 %vindex) {
main_body:
  %in1 = bitcast <4 x float> %vdata to <4 x i32>
  call void @llvm.amdgcn.tbuffer.store.v4i32(<4 x i32> %in1, <4 x i32> %0, i32 %vindex, i32 0, i32 0, i32 0, i32 15, i32 2, i1 0, i1 0)
  ret void
}

; GCN-LABEL: {{^}}buffer_store_ofs:
; GCN: tbuffer_store_format_xyzw v[0:3], v4, s[0:3], 0 format:[BUF_DATA_FORMAT_8_8,BUF_NUM_FORMAT_FLOAT] offen
define amdgpu_ps void @buffer_store_ofs(<4 x i32> inreg, <4 x float> %vdata, i32 %voffset) {
main_body:
  %in1 = bitcast <4 x float> %vdata to <4 x i32>
  call void @llvm.amdgcn.tbuffer.store.v4i32(<4 x i32> %in1, <4 x i32> %0, i32 0, i32 %voffset, i32 0, i32 0, i32 3, i32 7, i1 0, i1 0)
  ret void
}

; GCN-LABEL: {{^}}buffer_store_both:
; GCN: tbuffer_store_format_xyzw v[0:3], v[4:5], s[0:3], 0 format:[BUF_DATA_FORMAT_10_11_11,BUF_NUM_FORMAT_UINT] idxen offen
define amdgpu_ps void @buffer_store_both(<4 x i32> inreg, <4 x float> %vdata, i32 %vindex, i32 %voffset) {
main_body:
  %in1 = bitcast <4 x float> %vdata to <4 x i32>
  call void @llvm.amdgcn.tbuffer.store.v4i32(<4 x i32> %in1, <4 x i32> %0, i32 %vindex, i32 %voffset, i32 0, i32 0, i32 6, i32 4, i1 0, i1 0)
  ret void
}

; Ideally, the register allocator would avoid the wait here
;
; GCN-LABEL: {{^}}buffer_store_wait:
; GCN: tbuffer_store_format_xyzw v[0:3], v4, s[0:3], 0 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_SSCALED] idxen
; VERDE: s_waitcnt expcnt(0)
; GCN: buffer_load_format_xyzw v[0:3], v5, s[0:3], 0 idxen
; GCN: s_waitcnt vmcnt(0)
; GCN: tbuffer_store_format_xyzw v[0:3], v6, s[0:3], 0 format:[BUF_DATA_FORMAT_32_32_32_32,BUF_NUM_FORMAT_USCALED] idxen
define amdgpu_ps void @buffer_store_wait(<4 x i32> inreg, <4 x float> %vdata, i32 %vindex.1, i32 %vindex.2, i32 %vindex.3) {
main_body:
  %in1 = bitcast <4 x float> %vdata to <4 x i32>
  call void @llvm.amdgcn.tbuffer.store.v4i32(<4 x i32> %in1, <4 x i32> %0, i32 %vindex.1, i32 0, i32 0, i32 0, i32 15, i32 3, i1 0, i1 0)
  %data = call <4 x float> @llvm.amdgcn.buffer.load.format.v4f32(<4 x i32> %0, i32 %vindex.2, i32 0, i1 0, i1 0)
  %data.i = bitcast <4 x float> %data to <4 x i32>
  call void @llvm.amdgcn.tbuffer.store.v4i32(<4 x i32> %data.i, <4 x i32> %0, i32 %vindex.3, i32 0, i32 0, i32 0, i32 14, i32 2, i1 0, i1 0)
  ret void
}

; GCN-LABEL: {{^}}buffer_store_x1:
; GCN: tbuffer_store_format_x v0, v1, s[0:3], 0 format:[BUF_DATA_FORMAT_32_32_32,BUF_NUM_FORMAT_FLOAT] idxen
define amdgpu_ps void @buffer_store_x1(<4 x i32> inreg %rsrc, float %data, i32 %vindex) {
main_body:
  %data.i = bitcast float %data to i32
  call void @llvm.amdgcn.tbuffer.store.i32(i32 %data.i, <4 x i32> %rsrc, i32 %vindex, i32 0, i32 0, i32 0, i32 13, i32 7, i1 0, i1 0)
  ret void
}

; GCN-LABEL: {{^}}buffer_store_x2:
; GCN: tbuffer_store_format_xy v[0:1], v2, s[0:3], 0 format:[BUF_NUM_FORMAT_USCALED] idxen
define amdgpu_ps void @buffer_store_x2(<4 x i32> inreg %rsrc, <2 x float> %data, i32 %vindex) {
main_body:
  %data.i = bitcast <2 x float> %data to <2 x i32>
  call void @llvm.amdgcn.tbuffer.store.v2i32(<2 x i32> %data.i, <4 x i32> %rsrc, i32 %vindex, i32 0, i32 0, i32 0, i32 1, i32 2, i1 0, i1 0)
  ret void
}

declare void @llvm.amdgcn.tbuffer.store.i32(i32, <4 x i32>, i32, i32, i32, i32, i32, i32, i1, i1) #0
declare void @llvm.amdgcn.tbuffer.store.v2i32(<2 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i1, i1) #0
declare void @llvm.amdgcn.tbuffer.store.v4i32(<4 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i1, i1) #0
declare void @llvm.amdgcn.tbuffer.store.v4f32(<4 x float>, <4 x i32>, i32, i32, i32, i32, i32, i32, i1, i1) #0
declare <4 x float> @llvm.amdgcn.buffer.load.format.v4f32(<4 x i32>, i32, i32, i1, i1) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }

