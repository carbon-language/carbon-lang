;RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck -check-prefix=GCN -check-prefix=PREGFX10 %s
;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck -check-prefix=GCN -check-prefix=PREGFX10 %s
;RUN: llc < %s -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs | FileCheck -check-prefix=GCN -check-prefix=GFX10 %s

; GCN-LABEL: {{^}}tbuffer_load:
; PREGFX10: tbuffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, off, {{s\[[0-9]+:[0-9]+\]}}, 0 format:[BUF_DATA_FORMAT_32_32_32_32,BUF_NUM_FORMAT_UINT]
; PREGFX10: tbuffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, off, {{s\[[0-9]+:[0-9]+\]}}, 0 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_SSCALED] glc
; PREGFX10: tbuffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, off, {{s\[[0-9]+:[0-9]+\]}}, 0 format:[BUF_DATA_FORMAT_10_11_11,BUF_NUM_FORMAT_SNORM] slc
; PREGFX10: tbuffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, off, {{s\[[0-9]+:[0-9]+\]}}, 0 format:[BUF_DATA_FORMAT_10_11_11,BUF_NUM_FORMAT_SNORM] glc
; GFX10-DAG: tbuffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, off, {{s\[[0-9]+:[0-9]+\]}}, 0 format:78
; GFX10-DAG: tbuffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, off, {{s\[[0-9]+:[0-9]+\]}}, 0 format:[BUF_FMT_32_32_SINT] glc
; GFX10-DAG: tbuffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, off, {{s\[[0-9]+:[0-9]+\]}}, 0 format:[BUF_FMT_32_FLOAT] slc
; GFX10-DAG: tbuffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, off, {{s\[[0-9]+:[0-9]+\]}}, 0 format:[BUF_FMT_32_FLOAT] glc dlc
; GCN: s_waitcnt
define amdgpu_vs {<4 x float>, <4 x float>, <4 x float>, <4 x float>} @tbuffer_load(<4 x i32> inreg) {
main_body:
    %vdata     = call <4 x i32> @llvm.amdgcn.raw.tbuffer.load.v4i32(<4 x i32> %0, i32 0, i32 0, i32 78, i32 0)
    %vdata_glc = call <4 x i32> @llvm.amdgcn.raw.tbuffer.load.v4i32(<4 x i32> %0, i32 0, i32 0, i32 63, i32 1)
    %vdata_slc = call <4 x i32> @llvm.amdgcn.raw.tbuffer.load.v4i32(<4 x i32> %0, i32 0, i32 0, i32 22, i32 2)
    %vdata_f32 = call <4 x float> @llvm.amdgcn.raw.tbuffer.load.v4f32(<4 x i32> %0, i32 0, i32 0, i32 22, i32 5)
    %vdata.f     = bitcast <4 x i32> %vdata to <4 x float>
    %vdata_glc.f = bitcast <4 x i32> %vdata_glc to <4 x float>
    %vdata_slc.f = bitcast <4 x i32> %vdata_slc to <4 x float>
    %r0 = insertvalue {<4 x float>, <4 x float>, <4 x float>, <4 x float>} undef, <4 x float> %vdata.f, 0
    %r1 = insertvalue {<4 x float>, <4 x float>, <4 x float>, <4 x float>} %r0, <4 x float> %vdata_glc.f, 1
    %r2 = insertvalue {<4 x float>, <4 x float>, <4 x float>, <4 x float>} %r1, <4 x float> %vdata_slc.f, 2
    %r3 = insertvalue {<4 x float>, <4 x float>, <4 x float>, <4 x float>} %r2, <4 x float> %vdata_f32, 3
    ret {<4 x float>, <4 x float>, <4 x float>, <4 x float>} %r3
}

; GCN-LABEL: {{^}}tbuffer_load_immoffs:
; PREGFX10: tbuffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, off, {{s\[[0-9]+:[0-9]+\]}}, 0 format:[BUF_DATA_FORMAT_32_32_32_32,BUF_NUM_FORMAT_UINT] offset:42
; GFX10: tbuffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, off, {{s\[[0-9]+:[0-9]+\]}}, 0 format:78 offset:42
define amdgpu_vs <4 x float> @tbuffer_load_immoffs(<4 x i32> inreg) {
main_body:
    %vdata   = call <4 x i32> @llvm.amdgcn.raw.tbuffer.load.v4i32(<4 x i32> %0, i32 42, i32 0, i32 78, i32 0)
    %vdata.f = bitcast <4 x i32> %vdata to <4 x float>
    ret <4 x float> %vdata.f
}

; GCN-LABEL: {{^}}tbuffer_load_immoffs_large
; PREGFX10: tbuffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, off, {{s\[[0-9]+:[0-9]+\]}}, 61 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_USCALED] offset:4095
; PREGFX10: tbuffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, off, {{s\[[0-9]+:[0-9]+\]}}, {{s[0-9]+}} format:[BUF_DATA_FORMAT_32_32_32_32,BUF_NUM_FORMAT_SSCALED] offset:73
; PREGFX10: tbuffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, off, {{s\[[0-9]+:[0-9]+\]}}, {{s[0-9]+}} format:[BUF_DATA_FORMAT_32_32_32,BUF_NUM_FORMAT_UINT] offset:1
; GFX10-DAG: tbuffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, off, {{s\[[0-9]+:[0-9]+\]}}, 61 format:[BUF_FMT_10_10_10_2_SSCALED] offset:4095
; GFX10-DAG: tbuffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, off, {{s\[[0-9]+:[0-9]+\]}}, {{s[0-9]+}} format:[BUF_FMT_32_32_UINT] offset:73
; GFX10-DAG: tbuffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, off, {{s\[[0-9]+:[0-9]+\]}}, {{s[0-9]+}} format:[BUF_FMT_32_32_32_32_FLOAT] offset:1
; GCN: s_waitcnt
define amdgpu_vs {<4 x float>, <4 x float>, <4 x float>} @tbuffer_load_immoffs_large(<4 x i32> inreg, i32 inreg %soffs) {
    %vdata     = call <4 x i32> @llvm.amdgcn.raw.tbuffer.load.v4i32(<4 x i32> %0, i32 4095, i32 61, i32 47, i32 0)
    %vdata_glc = call <4 x i32> @llvm.amdgcn.raw.tbuffer.load.v4i32(<4 x i32> %0, i32 73, i32 %soffs, i32 62, i32 0)
    %vdata_slc = call <4 x i32> @llvm.amdgcn.raw.tbuffer.load.v4i32(<4 x i32> %0, i32 1, i32 %soffs, i32 77, i32 0)
    %vdata.f     = bitcast <4 x i32> %vdata to <4 x float>
    %vdata_glc.f = bitcast <4 x i32> %vdata_glc to <4 x float>
    %vdata_slc.f = bitcast <4 x i32> %vdata_slc to <4 x float>
    %r0 = insertvalue {<4 x float>, <4 x float>, <4 x float>} undef, <4 x float> %vdata.f, 0
    %r1 = insertvalue {<4 x float>, <4 x float>, <4 x float>} %r0, <4 x float> %vdata_glc.f, 1
    %r2 = insertvalue {<4 x float>, <4 x float>, <4 x float>} %r1, <4 x float> %vdata_slc.f, 2
    ret {<4 x float>, <4 x float>, <4 x float>} %r2
}

; GCN-LABEL: {{^}}tbuffer_load_ofs:
; PREGFX10: tbuffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0 format:[BUF_DATA_FORMAT_32_32_32_32,BUF_NUM_FORMAT_UINT] offen
; GFX10: tbuffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0 format:78 offen
define amdgpu_vs <4 x float> @tbuffer_load_ofs(<4 x i32> inreg, i32 %voffs) {
main_body:
    %vdata   = call <4 x i32> @llvm.amdgcn.raw.tbuffer.load.v4i32(<4 x i32> %0, i32 %voffs, i32 0, i32 78, i32 0)
    %vdata.f = bitcast <4 x i32> %vdata to <4 x float>
    ret <4 x float> %vdata.f
}

; GCN-LABEL: {{^}}tbuffer_load_ofs_imm:
; PREGFX10: tbuffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0 format:[BUF_DATA_FORMAT_32_32_32_32,BUF_NUM_FORMAT_UINT] offen offset:52
; GFX10: tbuffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0 format:78 offen offset:52
define amdgpu_vs <4 x float> @tbuffer_load_ofs_imm(<4 x i32> inreg, i32 %voffs) {
main_body:
    %ofs = add i32 %voffs, 52
    %vdata   = call <4 x i32> @llvm.amdgcn.raw.tbuffer.load.v4i32(<4 x i32> %0, i32 %ofs, i32 0, i32 78, i32 0)
    %vdata.f = bitcast <4 x i32> %vdata to <4 x float>
    ret <4 x float> %vdata.f
}

; GCN-LABEL: {{^}}buffer_load_xy:
; PREGFX10: tbuffer_load_format_xy {{v\[[0-9]+:[0-9]+\]}}, off, {{s\[[0-9]+:[0-9]+\]}}, 0 format:[BUF_DATA_FORMAT_32_32_32,BUF_NUM_FORMAT_UINT]
; GFX10: tbuffer_load_format_xy {{v\[[0-9]+:[0-9]+\]}}, off, {{s\[[0-9]+:[0-9]+\]}}, 0 format:[BUF_FMT_32_32_32_32_FLOAT]
define amdgpu_vs <2 x float> @buffer_load_xy(<4 x i32> inreg %rsrc) {
    %vdata = call <2 x i32> @llvm.amdgcn.raw.tbuffer.load.v2i32(<4 x i32> %rsrc, i32 0, i32 0, i32 77, i32 0)
    %vdata.f = bitcast <2 x i32> %vdata to <2 x float>
    ret <2 x float> %vdata.f
}

; GCN-LABEL: {{^}}buffer_load_x:
; PREGFX10: tbuffer_load_format_x {{v[0-9]+}}, off, {{s\[[0-9]+:[0-9]+\]}}, 0 format:[BUF_DATA_FORMAT_32_32_32,BUF_NUM_FORMAT_UINT]
; GFX10: tbuffer_load_format_x {{v[0-9]+}}, off, {{s\[[0-9]+:[0-9]+\]}}, 0 format:[BUF_FMT_32_32_32_32_FLOAT]
define amdgpu_vs float @buffer_load_x(<4 x i32> inreg %rsrc) {
    %vdata = call i32 @llvm.amdgcn.raw.tbuffer.load.i32(<4 x i32> %rsrc, i32 0, i32 0, i32 77, i32 0)
    %vdata.f = bitcast i32 %vdata to float
    ret float %vdata.f
}

declare i32 @llvm.amdgcn.raw.tbuffer.load.i32(<4 x i32>, i32, i32, i32, i32)
declare <2 x i32> @llvm.amdgcn.raw.tbuffer.load.v2i32(<4 x i32>, i32, i32, i32, i32)
declare <4 x i32> @llvm.amdgcn.raw.tbuffer.load.v4i32(<4 x i32>, i32, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.raw.tbuffer.load.v4f32(<4 x i32>, i32, i32, i32, i32)

