; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs -show-mc-encoding < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX900,GFX9 %s
; RUN: llc -march=amdgcn -mcpu=gfx906 -verify-machineinstrs -show-mc-encoding < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX906,GFX9 %s
; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,CIVI,VI %s
; RUN: llc -march=amdgcn -mcpu=hawaii -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,CIVI,CI %s

; GCN-LABEL: {{^}}v_mad_mix_f32_f16lo_f16lo_f16lo:
; GFX900: v_mad_mix_f32 v0, v0, v1, v2 op_sel_hi:[1,1,1] ; encoding: [0x00,0x40,0xa0,0xd3,0x00,0x03,0x0a,0x1c]
; GFX906: v_fma_mix_f32 v0, v0, v1, v2 op_sel_hi:[1,1,1] ; encoding: [0x00,0x40,0xa0,0xd3,0x00,0x03,0x0a,0x1c]
; VI: v_mac_f32
; CI: v_mad_f32
define float @v_mad_mix_f32_f16lo_f16lo_f16lo(half %src0, half %src1, half %src2) #0 {
  %src0.ext = fpext half %src0 to float
  %src1.ext = fpext half %src1 to float
  %src2.ext = fpext half %src2 to float
  %result = tail call float @llvm.fmuladd.f32(float %src0.ext, float %src1.ext, float %src2.ext)
  ret float %result
}

; GCN-LABEL: {{^}}v_mad_mix_f32_f16hi_f16hi_f16hi_int:
; GFX900: v_mad_mix_f32 v0, v0, v1, v2 op_sel:[1,1,1] op_sel_hi:[1,1,1] ; encoding
; GFX906: v_fma_mix_f32 v0, v0, v1, v2 op_sel:[1,1,1] op_sel_hi:[1,1,1] ; encoding
; CIVI: v_mac_f32
define float @v_mad_mix_f32_f16hi_f16hi_f16hi_int(i32 %src0, i32 %src1, i32 %src2) #0 {
  %src0.hi = lshr i32 %src0, 16
  %src1.hi = lshr i32 %src1, 16
  %src2.hi = lshr i32 %src2, 16
  %src0.i16 = trunc i32 %src0.hi to i16
  %src1.i16 = trunc i32 %src1.hi to i16
  %src2.i16 = trunc i32 %src2.hi to i16
  %src0.fp16 = bitcast i16 %src0.i16 to half
  %src1.fp16 = bitcast i16 %src1.i16 to half
  %src2.fp16 = bitcast i16 %src2.i16 to half
  %src0.ext = fpext half %src0.fp16 to float
  %src1.ext = fpext half %src1.fp16 to float
  %src2.ext = fpext half %src2.fp16 to float
  %result = tail call float @llvm.fmuladd.f32(float %src0.ext, float %src1.ext, float %src2.ext)
  ret float %result
}

; GCN-LABEL: {{^}}v_mad_mix_f32_f16hi_f16hi_f16hi_elt:
; GFX900: v_mad_mix_f32 v0, v0, v1, v2 op_sel:[1,1,1] op_sel_hi:[1,1,1] ; encoding
; GFX906: v_fma_mix_f32 v0, v0, v1, v2 op_sel:[1,1,1] op_sel_hi:[1,1,1] ; encoding
; VI: v_mac_f32
; CI: v_mad_f32
define float @v_mad_mix_f32_f16hi_f16hi_f16hi_elt(<2 x half> %src0, <2 x half> %src1, <2 x half> %src2) #0 {
  %src0.hi = extractelement <2 x half> %src0, i32 1
  %src1.hi = extractelement <2 x half> %src1, i32 1
  %src2.hi = extractelement <2 x half> %src2, i32 1
  %src0.ext = fpext half %src0.hi to float
  %src1.ext = fpext half %src1.hi to float
  %src2.ext = fpext half %src2.hi to float
  %result = tail call float @llvm.fmuladd.f32(float %src0.ext, float %src1.ext, float %src2.ext)
  ret float %result
}

; GCN-LABEL: {{^}}v_mad_mix_v2f32:
; GFX900: v_mad_mix_f32 v3, v0, v1, v2 op_sel:[1,1,1] op_sel_hi:[1,1,1]
; GFX900-NEXT: v_mad_mix_f32 v0, v0, v1, v2 op_sel_hi:[1,1,1]
; GFX900-NEXT: v_mov_b32_e32 v1, v3

; GFX906: v_fma_mix_f32 v3, v0, v1, v2 op_sel:[1,1,1] op_sel_hi:[1,1,1]
; GFX906-NEXT: v_fma_mix_f32 v0, v0, v1, v2 op_sel_hi:[1,1,1]
; GFX906-NEXT: v_mov_b32_e32 v1, v3

; CIVI: v_mac_f32
define <2 x float> @v_mad_mix_v2f32(<2 x half> %src0, <2 x half> %src1, <2 x half> %src2) #0 {
  %src0.ext = fpext <2 x half> %src0 to <2 x float>
  %src1.ext = fpext <2 x half> %src1 to <2 x float>
  %src2.ext = fpext <2 x half> %src2 to <2 x float>
  %result = tail call <2 x float> @llvm.fmuladd.v2f32(<2 x float> %src0.ext, <2 x float> %src1.ext, <2 x float> %src2.ext)
  ret <2 x float> %result
}

; GCN-LABEL: {{^}}v_mad_mix_v2f32_shuffle:
; GCN: s_waitcnt
; GFX900: v_mad_mix_f32 v3, v0, v1, v2 op_sel:[1,0,1] op_sel_hi:[1,1,1]
; GFX900-NEXT: v_mad_mix_f32 v1, v0, v1, v2 op_sel:[0,1,1] op_sel_hi:[1,1,1]
; GFX900-NEXT: v_mov_b32_e32 v0, v3
; GFX900-NEXT: s_setpc_b64

; GFX906-NEXT: v_fma_mix_f32 v3, v0, v1, v2 op_sel:[1,0,1] op_sel_hi:[1,1,1]
; GFX906-NEXT: v_fma_mix_f32 v1, v0, v1, v2 op_sel:[0,1,1] op_sel_hi:[1,1,1]
; GFX906-NEXT: v_mov_b32_e32 v0, v3
; GFX906-NEXT: s_setpc_b64

; CIVI: v_mac_f32
define <2 x float> @v_mad_mix_v2f32_shuffle(<2 x half> %src0, <2 x half> %src1, <2 x half> %src2) #0 {
  %src0.shuf = shufflevector <2 x half> %src0, <2 x half> undef, <2 x i32> <i32 1, i32 0>
  %src1.shuf = shufflevector <2 x half> %src1, <2 x half> undef, <2 x i32> <i32 0, i32 1>
  %src2.shuf = shufflevector <2 x half> %src2, <2 x half> undef, <2 x i32> <i32 1, i32 1>
  %src0.ext = fpext <2 x half> %src0.shuf to <2 x float>
  %src1.ext = fpext <2 x half> %src1.shuf to <2 x float>
  %src2.ext = fpext <2 x half> %src2.shuf to <2 x float>
  %result = tail call <2 x float> @llvm.fmuladd.v2f32(<2 x float> %src0.ext, <2 x float> %src1.ext, <2 x float> %src2.ext)
  ret <2 x float> %result
}

; GCN-LABEL: {{^}}v_mad_mix_f32_negf16lo_f16lo_f16lo:
; GFX900: s_waitcnt
; GFX900-NEXT: v_mad_mix_f32 v0, -v0, v1, v2 op_sel_hi:[1,1,1] ; encoding
; GFX900-NEXT: s_setpc_b64

; GFX906: s_waitcnt
; GFX906-NEXT: v_fma_mix_f32 v0, -v0, v1, v2  op_sel_hi:[1,1,1] ; encoding
; GFX906-NEXT: s_setpc_b64

; CIVI: v_mad_f32
define float @v_mad_mix_f32_negf16lo_f16lo_f16lo(half %src0, half %src1, half %src2) #0 {
  %src0.ext = fpext half %src0 to float
  %src1.ext = fpext half %src1 to float
  %src2.ext = fpext half %src2 to float
  %src0.ext.neg = fneg float %src0.ext
  %result = tail call float @llvm.fmuladd.f32(float %src0.ext.neg, float %src1.ext, float %src2.ext)
  ret float %result
}

; GCN-LABEL: {{^}}v_mad_mix_f32_absf16lo_f16lo_f16lo:
; GFX900: v_mad_mix_f32 v0, |v0|, v1, v2 op_sel_hi:[1,1,1]
; GFX906: v_fma_mix_f32 v0, |v0|, v1, v2 op_sel_hi:[1,1,1]

; CIVI: v_mad_f32
define float @v_mad_mix_f32_absf16lo_f16lo_f16lo(half %src0, half %src1, half %src2) #0 {
  %src0.ext = fpext half %src0 to float
  %src1.ext = fpext half %src1 to float
  %src2.ext = fpext half %src2 to float
  %src0.ext.abs = call float @llvm.fabs.f32(float %src0.ext)
  %result = tail call float @llvm.fmuladd.f32(float %src0.ext.abs, float %src1.ext, float %src2.ext)
  ret float %result
}

; GCN-LABEL: {{^}}v_mad_mix_f32_negabsf16lo_f16lo_f16lo:
; GFX900: s_waitcnt
; GFX900-NEXT: v_mad_mix_f32 v0, -|v0|, v1, v2 op_sel_hi:[1,1,1]
; GFX900-NEXT: s_setpc_b64

; GFX906: s_waitcnt
; GFX906-NEXT: v_fma_mix_f32 v0, -|v0|, v1, v2 op_sel_hi:[1,1,1]
; GFX906-NEXT: s_setpc_b64

; CIVI: v_mad_f32
define float @v_mad_mix_f32_negabsf16lo_f16lo_f16lo(half %src0, half %src1, half %src2) #0 {
  %src0.ext = fpext half %src0 to float
  %src1.ext = fpext half %src1 to float
  %src2.ext = fpext half %src2 to float
  %src0.ext.abs = call float @llvm.fabs.f32(float %src0.ext)
  %src0.ext.neg.abs = fneg float %src0.ext.abs
  %result = tail call float @llvm.fmuladd.f32(float %src0.ext.neg.abs, float %src1.ext, float %src2.ext)
  ret float %result
}

; GCN-LABEL: {{^}}v_mad_mix_f32_f16lo_f16lo_f32:
; GCN: s_waitcnt
; GFX900-NEXT: v_mad_mix_f32 v0, v0, v1, v2 op_sel_hi:[1,1,0] ; encoding
; GFX906-NEXT: v_fma_mix_f32 v0, v0, v1, v2 op_sel_hi:[1,1,0] ; encoding
; GFX9-NEXT: s_setpc_b64

; CIVI: v_mad_f32
define float @v_mad_mix_f32_f16lo_f16lo_f32(half %src0, half %src1, float %src2) #0 {
  %src0.ext = fpext half %src0 to float
  %src1.ext = fpext half %src1 to float
  %result = tail call float @llvm.fmuladd.f32(float %src0.ext, float %src1.ext, float %src2)
  ret float %result
}

; GCN-LABEL: {{^}}v_mad_mix_f32_f16lo_f16lo_negf32:
; GCN: s_waitcnt
; GFX900-NEXT: v_mad_mix_f32 v0, v0, v1, -v2 op_sel_hi:[1,1,0] ; encoding
; GFX906-NEXT: v_fma_mix_f32 v0, v0, v1, -v2 op_sel_hi:[1,1,0] ; encoding
; GFX9-NEXT: s_setpc_b64

; CIVI: v_mad_f32
define float @v_mad_mix_f32_f16lo_f16lo_negf32(half %src0, half %src1, float %src2) #0 {
  %src0.ext = fpext half %src0 to float
  %src1.ext = fpext half %src1 to float
  %src2.neg = fneg float %src2
  %result = tail call float @llvm.fmuladd.f32(float %src0.ext, float %src1.ext, float %src2.neg)
  ret float %result
}

; GCN-LABEL: {{^}}v_mad_mix_f32_f16lo_f16lo_absf32:
; GCN: s_waitcnt
; GFX900-NEXT: v_mad_mix_f32 v0, v0, v1, |v2| op_sel_hi:[1,1,0] ; encoding
; GFX906-NEXT: v_fma_mix_f32 v0, v0, v1, |v2| op_sel_hi:[1,1,0] ; encoding
; GFX9-NEXT: s_setpc_b64

; CIVI: v_mad_f32
define float @v_mad_mix_f32_f16lo_f16lo_absf32(half %src0, half %src1, float %src2) #0 {
  %src0.ext = fpext half %src0 to float
  %src1.ext = fpext half %src1 to float
  %src2.abs = call float @llvm.fabs.f32(float %src2)
  %result = tail call float @llvm.fmuladd.f32(float %src0.ext, float %src1.ext, float %src2.abs)
  ret float %result
}

; GCN-LABEL: {{^}}v_mad_mix_f32_f16lo_f16lo_negabsf32:
; GCN: s_waitcnt
; GFX900-NEXT: v_mad_mix_f32 v0, v0, v1, -|v2| op_sel_hi:[1,1,0] ; encoding
; GFX906-NEXT: v_fma_mix_f32 v0, v0, v1, -|v2| op_sel_hi:[1,1,0] ; encoding
; GFX9-NEXT: s_setpc_b64

; CIVI: v_mad_f32
define float @v_mad_mix_f32_f16lo_f16lo_negabsf32(half %src0, half %src1, float %src2) #0 {
  %src0.ext = fpext half %src0 to float
  %src1.ext = fpext half %src1 to float
  %src2.abs = call float @llvm.fabs.f32(float %src2)
  %src2.neg.abs = fneg float %src2.abs
  %result = tail call float @llvm.fmuladd.f32(float %src0.ext, float %src1.ext, float %src2.neg.abs)
  ret float %result
}

; TODO: Fold inline immediates. Need to be careful because it is an
; f16 inline immediate that may be converted to f32, not an actual f32
; inline immediate.

; GCN-LABEL: {{^}}v_mad_mix_f32_f16lo_f16lo_f32imm1:
; GCN: s_waitcnt
; GFX9: s_mov_b32 [[SREG:s[0-9]+]], 1.0
; GFX900-NEXT: v_mad_mix_f32 v0, v0, v1, [[SREG]]  op_sel_hi:[1,1,0] ; encoding
; GFX906-NEXT: v_fma_mix_f32 v0, v0, v1, [[SREG]]  op_sel_hi:[1,1,0] ; encoding

; CIVI: v_mad_f32 v0, v0, v1, 1.0
; GCN-NEXT: s_setpc_b64
define float @v_mad_mix_f32_f16lo_f16lo_f32imm1(half %src0, half %src1) #0 {
  %src0.ext = fpext half %src0 to float
  %src1.ext = fpext half %src1 to float
  %result = tail call float @llvm.fmuladd.f32(float %src0.ext, float %src1.ext, float 1.0)
  ret float %result
}

; GCN-LABEL: {{^}}v_mad_mix_f32_f16lo_f16lo_f32imminv2pi:
; GCN: s_waitcnt
; GFX9: s_mov_b32 [[SREG:s[0-9]+]], 0.15915494
; GFX900: v_mad_mix_f32 v0, v0, v1, [[SREG]]  op_sel_hi:[1,1,0] ; encoding
; GFX906: v_fma_mix_f32 v0, v0, v1, [[SREG]]  op_sel_hi:[1,1,0] ; encoding
; VI: v_mad_f32 v0, v0, v1, 0.15915494
define float @v_mad_mix_f32_f16lo_f16lo_f32imminv2pi(half %src0, half %src1) #0 {
  %src0.ext = fpext half %src0 to float
  %src1.ext = fpext half %src1 to float
  %result = tail call float @llvm.fmuladd.f32(float %src0.ext, float %src1.ext, float 0x3FC45F3060000000)
  ret float %result
}

; Attempt to break inline immediate folding. If the operand is
; interpreted as f32, the inline immediate is really the f16 inline
; imm value converted to f32.
;	fpext f16 1/2pi = 0x3e230000
;	      f32 1/2pi = 0x3e22f983
; GCN-LABEL: {{^}}v_mad_mix_f32_f16lo_f16lo_cvtf16imminv2pi:
; GFX9: s_mov_b32 [[SREG:s[0-9]+]], 0x3e230000
; GFX900: v_mad_mix_f32 v0, v0, v1, [[SREG]]  op_sel_hi:[1,1,0] ; encoding
; GFX906: v_fma_mix_f32 v0, v0, v1, [[SREG]]  op_sel_hi:[1,1,0] ; encoding

; CIVI: v_madak_f32 v0, v0, v1, 0x3e230000
define float @v_mad_mix_f32_f16lo_f16lo_cvtf16imminv2pi(half %src0, half %src1) #0 {
  %src0.ext = fpext half %src0 to float
  %src1.ext = fpext half %src1 to float
  %src2 = fpext half 0xH3118 to float
  %result = tail call float @llvm.fmuladd.f32(float %src0.ext, float %src1.ext, float %src2)
  ret float %result
}

; GCN-LABEL: {{^}}v_mad_mix_f32_f16lo_f16lo_cvtf16imm63:
; GFX9: s_mov_b32 [[SREG:s[0-9]+]], 0x367c0000
; GFX900: v_mad_mix_f32 v0, v0, v1, [[SREG]]  op_sel_hi:[1,1,0] ; encoding
; GFX906: v_fma_mix_f32 v0, v0, v1, [[SREG]]  op_sel_hi:[1,1,0] ; encoding

; CIVI: v_madak_f32 v0, v0, v1, 0x367c0000
define float @v_mad_mix_f32_f16lo_f16lo_cvtf16imm63(half %src0, half %src1) #0 {
  %src0.ext = fpext half %src0 to float
  %src1.ext = fpext half %src1 to float
  %src2 = fpext half 0xH003F to float
  %result = tail call float @llvm.fmuladd.f32(float %src0.ext, float %src1.ext, float %src2)
  ret float %result
}

; GCN-LABEL: {{^}}v_mad_mix_v2f32_f32imm1:
; GFX9: s_mov_b32 [[SREG:s[0-9]+]], 1.0
; GFX900: v_mad_mix_f32 v2, v0, v1, [[SREG]] op_sel:[1,1,0] op_sel_hi:[1,1,0] ; encoding
; GFX900: v_mad_mix_f32 v0, v0, v1, [[SREG]] op_sel_hi:[1,1,0] ; encoding
; GFX900: v_mov_b32_e32 v1, v2

; GFX906: v_fma_mix_f32 v2, v0, v1, [[SREG]] op_sel:[1,1,0] op_sel_hi:[1,1,0] ; encoding
; GFX906: v_fma_mix_f32 v0, v0, v1, [[SREG]] op_sel_hi:[1,1,0] ; encoding
; GFX906: v_mov_b32_e32 v1, v2
define <2 x float> @v_mad_mix_v2f32_f32imm1(<2 x half> %src0, <2 x half> %src1) #0 {
  %src0.ext = fpext <2 x half> %src0 to <2 x float>
  %src1.ext = fpext <2 x half> %src1 to <2 x float>
  %result = tail call <2 x float> @llvm.fmuladd.v2f32(<2 x float> %src0.ext, <2 x float> %src1.ext, <2 x float> <float 1.0, float 1.0>)
  ret <2 x float> %result
}

; GCN-LABEL: {{^}}v_mad_mix_v2f32_cvtf16imminv2pi:
; GFX9: s_mov_b32 [[SREG:s[0-9]+]], 0x3e230000

; GFX900: v_mad_mix_f32 v2, v0, v1, [[SREG]] op_sel:[1,1,0] op_sel_hi:[1,1,0] ; encoding
; GFX900: v_mad_mix_f32 v0, v0, v1, [[SREG]] op_sel_hi:[1,1,0] ; encoding
; GFX900: v_mov_b32_e32 v1, v2

; GFX906: v_fma_mix_f32 v2, v0, v1, [[SREG]] op_sel:[1,1,0] op_sel_hi:[1,1,0] ; encoding
; GFX906: v_fma_mix_f32 v0, v0, v1, [[SREG]] op_sel_hi:[1,1,0] ; encoding
; GFX906: v_mov_b32_e32 v1, v2
define <2 x float> @v_mad_mix_v2f32_cvtf16imminv2pi(<2 x half> %src0, <2 x half> %src1) #0 {
  %src0.ext = fpext <2 x half> %src0 to <2 x float>
  %src1.ext = fpext <2 x half> %src1 to <2 x float>
  %src2 = fpext <2 x half> <half 0xH3118, half 0xH3118> to <2 x float>
  %result = tail call <2 x float> @llvm.fmuladd.v2f32(<2 x float> %src0.ext, <2 x float> %src1.ext, <2 x float> %src2)
  ret <2 x float> %result
}

; GCN-LABEL: {{^}}v_mad_mix_v2f32_f32imminv2pi:
; GFX9: s_mov_b32 [[SREG:s[0-9]+]], 0.15915494

; GFX900: v_mad_mix_f32 v2, v0, v1, [[SREG]] op_sel:[1,1,0] op_sel_hi:[1,1,0] ; encoding
; GFX900: v_mad_mix_f32 v0, v0, v1, [[SREG]] op_sel_hi:[1,1,0] ; encoding
; GFX900: v_mov_b32_e32 v1, v2

; GFX906: v_fma_mix_f32 v2, v0, v1, [[SREG]] op_sel:[1,1,0] op_sel_hi:[1,1,0] ; encoding
; GFX906: v_fma_mix_f32 v0, v0, v1, [[SREG]] op_sel_hi:[1,1,0] ; encoding
; GFX906: v_mov_b32_e32 v1, v2
define <2 x float> @v_mad_mix_v2f32_f32imminv2pi(<2 x half> %src0, <2 x half> %src1) #0 {
  %src0.ext = fpext <2 x half> %src0 to <2 x float>
  %src1.ext = fpext <2 x half> %src1 to <2 x float>
  %src2 = fpext <2 x half> <half 0xH3118, half 0xH3118> to <2 x float>
  %result = tail call <2 x float> @llvm.fmuladd.v2f32(<2 x float> %src0.ext, <2 x float> %src1.ext, <2 x float> <float 0x3FC45F3060000000, float 0x3FC45F3060000000>)
  ret <2 x float> %result
}

; GCN-LABEL: {{^}}v_mad_mix_clamp_f32_f16hi_f16hi_f16hi_elt:
; GFX900: v_mad_mix_f32 v0, v0, v1, v2 op_sel:[1,1,1] op_sel_hi:[1,1,1] clamp ; encoding
; GFX906: v_fma_mix_f32 v0, v0, v1, v2 op_sel:[1,1,1] op_sel_hi:[1,1,1] clamp ; encoding
; CIVI: v_mad_f32 v{{[0-9]}}, v{{[0-9]}}, v{{[0-9]}}, v{{[0-9]}} clamp{{$}}
define float @v_mad_mix_clamp_f32_f16hi_f16hi_f16hi_elt(<2 x half> %src0, <2 x half> %src1, <2 x half> %src2) #0 {
  %src0.hi = extractelement <2 x half> %src0, i32 1
  %src1.hi = extractelement <2 x half> %src1, i32 1
  %src2.hi = extractelement <2 x half> %src2, i32 1
  %src0.ext = fpext half %src0.hi to float
  %src1.ext = fpext half %src1.hi to float
  %src2.ext = fpext half %src2.hi to float
  %result = tail call float @llvm.fmuladd.f32(float %src0.ext, float %src1.ext, float %src2.ext)
  %max = call float @llvm.maxnum.f32(float %result, float 0.0)
  %clamp = call float @llvm.minnum.f32(float %max, float 1.0)
  ret float %clamp
}

; GCN-LABEL: no_mix_simple:
; GCN: s_waitcnt
; GCN-NEXT: v_{{mad|fma}}_f32 v0, v0, v1, v2
; GCN-NEXT: s_setpc_b64
define float @no_mix_simple(float %src0, float %src1, float %src2) #0 {
  %result = call float @llvm.fmuladd.f32(float %src0, float %src1, float %src2)
  ret float %result
}

; GCN-LABEL: no_mix_simple_fabs:
; GCN: s_waitcnt
; CIVI-NEXT: v_mad_f32 v0, |v0|, v1, v2
; GFX900-NEXT: v_mad_f32 v0, |v0|, v1, v2
; GFX906-NEXT: v_fma_f32 v0, |v0|, v1, v2
; GCN-NEXT: s_setpc_b64
define float @no_mix_simple_fabs(float %src0, float %src1, float %src2) #0 {
  %src0.fabs = call float @llvm.fabs.f32(float %src0)
  %result = call float @llvm.fmuladd.f32(float %src0.fabs, float %src1, float %src2)
  ret float %result
}

; FIXME: Should abe able to select in thits case
; All sources are converted from f16, so it doesn't matter
; v_mad_mix_f32 flushes.

; GCN-LABEL: {{^}}v_mad_mix_f32_f16lo_f16lo_f16lo_f32_denormals:
; GFX900: v_cvt_f32_f16
; GFX900: v_cvt_f32_f16
; GFX900: v_cvt_f32_f16
; GFX900: v_fma_f32
define float @v_mad_mix_f32_f16lo_f16lo_f16lo_f32_denormals(half %src0, half %src1, half %src2) #1 {
  %src0.ext = fpext half %src0 to float
  %src1.ext = fpext half %src1 to float
  %src2.ext = fpext half %src2 to float
  %result = tail call float @llvm.fmuladd.f32(float %src0.ext, float %src1.ext, float %src2.ext)
  ret float %result
}

; GCN-LABEL: {{^}}v_mad_mix_f32_f16lo_f16lo_f32_denormals:
; GFX900: v_cvt_f32_f16
; GFX900: v_cvt_f32_f16
; GFX900: v_fma_f32

; GFX906-NOT: v_cvt_f32_f16
; GFX906: v_fma_mix_f32 v0, v0, v1, v2 op_sel_hi:[1,1,0]
define float @v_mad_mix_f32_f16lo_f16lo_f32_denormals(half %src0, half %src1, float %src2) #1 {
  %src0.ext = fpext half %src0 to float
  %src1.ext = fpext half %src1 to float
  %result = tail call float @llvm.fmuladd.f32(float %src0.ext, float %src1.ext, float %src2)
  ret float %result
}

; GCN-LABEL: {{^}}v_mad_mix_f32_f16lo_f16lo_f16lo_f32_denormals_fmulfadd:
; GFX9: v_cvt_f32_f16
; GFX9: v_cvt_f32_f16
; GFX9: v_cvt_f32_f16
; GFX9: v_mul_f32
; GFX9: v_add_f32
define float @v_mad_mix_f32_f16lo_f16lo_f16lo_f32_denormals_fmulfadd(half %src0, half %src1, half %src2) #1 {
  %src0.ext = fpext half %src0 to float
  %src1.ext = fpext half %src1 to float
  %src2.ext = fpext half %src2 to float
  %mul = fmul float %src0.ext, %src1.ext
  %result = fadd float %mul, %src2.ext
  ret float %result
}

; GCN-LABEL: {{^}}v_mad_mix_f32_f16lo_f16lo_f32_denormals_fmulfadd:
; GFX9: v_cvt_f32_f16
; GFX9: v_cvt_f32_f16
; GFX9: v_mul_f32
; GFX9: v_add_f32
define float @v_mad_mix_f32_f16lo_f16lo_f32_denormals_fmulfadd(half %src0, half %src1, float %src2) #1 {
  %src0.ext = fpext half %src0 to float
  %src1.ext = fpext half %src1 to float
  %mul = fmul float %src0.ext, %src1.ext
  %result = fadd float %mul, %src2
  ret float %result
}

; GCN-LABEL: {{^}}v_mad_mix_f32_f16lo_f16lo_f16lo_f32_flush_fmulfadd:
; GCN: s_waitcnt
; GFX900-NEXT: v_mad_mix_f32 v0, v0, v1, v2 op_sel_hi:[1,1,1] ; encoding
; GFX906-NEXT: v_fma_mix_f32 v0, v0, v1, v2 op_sel_hi:[1,1,1] ; encoding
; GFX9-NEXT: s_setpc_b64
define float @v_mad_mix_f32_f16lo_f16lo_f16lo_f32_flush_fmulfadd(half %src0, half %src1, half %src2) #0 {
  %src0.ext = fpext half %src0 to float
  %src1.ext = fpext half %src1 to float
  %src2.ext = fpext half %src2 to float
  %mul = fmul contract float %src0.ext, %src1.ext
  %result = fadd contract float %mul, %src2.ext
  ret float %result
}

; GCN-LABEL: {{^}}v_mad_mix_f32_f16lo_f16lo_f32_flush_fmulfadd:
; GCN: s_waitcnt
; GFX900-NEXT: v_mad_mix_f32 v0, v0, v1, v2 op_sel_hi:[1,1,0] ; encoding
; GFX906-NEXT: v_fma_mix_f32 v0, v0, v1, v2 op_sel_hi:[1,1,0] ; encoding
; GFX9-NEXT: s_setpc_b64
define float @v_mad_mix_f32_f16lo_f16lo_f32_flush_fmulfadd(half %src0, half %src1, float %src2) #0 {
  %src0.ext = fpext half %src0 to float
  %src1.ext = fpext half %src1 to float
  %mul = fmul contract float %src0.ext, %src1.ext
  %result = fadd contract float %mul, %src2
  ret float %result
}

; GCN-LABEL: {{^}}v_mad_mix_f32_negprecvtf16lo_f16lo_f16lo:
; GFX9: s_waitcnt
; GFX900-NEXT: v_mad_mix_f32 v0, -v0, v1, v2 op_sel_hi:[1,1,1] ; encoding
; GFX906-NEXT: v_fma_mix_f32 v0, -v0, v1, v2 op_sel_hi:[1,1,1] ; encoding
; GFX9-NEXT: s_setpc_b64

; CIVI: v_mad_f32
define float @v_mad_mix_f32_negprecvtf16lo_f16lo_f16lo(i32 %src0.arg, half %src1, half %src2) #0 {
  %src0.arg.bc = bitcast i32 %src0.arg to <2 x half>
  %src0 = extractelement <2 x half> %src0.arg.bc, i32 0
  %src0.neg = fsub half -0.0, %src0
  %src0.ext = fpext half %src0.neg to float
  %src1.ext = fpext half %src1 to float
  %src2.ext = fpext half %src2 to float
;  %src0.ext.neg = fsub float -0.0, %src0.ext
  %result = tail call float @llvm.fmuladd.f32(float %src0.ext, float %src1.ext, float %src2.ext)
  ret float %result
}

; Make sure we don't fold pre-cvt fneg if we already have a fabs
; GCN-LABEL: {{^}}v_mad_mix_f32_precvtnegf16hi_abs_f16lo_f16lo:
; GFX900: s_waitcnt
define float @v_mad_mix_f32_precvtnegf16hi_abs_f16lo_f16lo(i32 %src0.arg, half %src1, half %src2) #0 {
  %src0.arg.bc = bitcast i32 %src0.arg to <2 x half>
  %src0 = extractelement <2 x half> %src0.arg.bc, i32 1
  %src0.neg = fsub half -0.0, %src0
  %src0.ext = fpext half %src0.neg to float
  %src0.ext.abs = call float @llvm.fabs.f32(float %src0.ext)
  %src1.ext = fpext half %src1 to float
  %src2.ext = fpext half %src2 to float
  %result = tail call float @llvm.fmuladd.f32(float %src0.ext.abs, float %src1.ext, float %src2.ext)
  ret float %result
}

; GCN-LABEL: {{^}}v_mad_mix_f32_precvtabsf16hi_f16lo_f16lo:
; GFX9: s_waitcnt
; GFX900-NEXT: v_mad_mix_f32 v0, |v0|, v1, v2 op_sel:[1,0,0] op_sel_hi:[1,1,1]
; GFX906-NEXT: v_fma_mix_f32 v0, |v0|, v1, v2 op_sel:[1,0,0] op_sel_hi:[1,1,1]
; GFX9-NEXT: s_setpc_b64
define float @v_mad_mix_f32_precvtabsf16hi_f16lo_f16lo(i32 %src0.arg, half %src1, half %src2) #0 {
  %src0.arg.bc = bitcast i32 %src0.arg to <2 x half>
  %src0 = extractelement <2 x half> %src0.arg.bc, i32 1
  %src0.abs = call half @llvm.fabs.f16(half %src0)
  %src0.ext = fpext half %src0.abs to float
  %src1.ext = fpext half %src1 to float
  %src2.ext = fpext half %src2 to float
  %result = tail call float @llvm.fmuladd.f32(float %src0.ext, float %src1.ext, float %src2.ext)
  ret float %result
}

; GCN-LABEL: {{^}}v_mad_mix_f32_preextractfneg_f16hi_f16lo_f16lo:
; GFX9: s_waitcnt
; GFX900-NEXT: v_mad_mix_f32 v0, -v0, v1, v2 op_sel:[1,0,0] op_sel_hi:[1,1,1]
; GFX906-NEXT: v_fma_mix_f32 v0, -v0, v1, v2 op_sel:[1,0,0] op_sel_hi:[1,1,1]
; GFX9-NEXT: s_setpc_b64
define float @v_mad_mix_f32_preextractfneg_f16hi_f16lo_f16lo(i32 %src0.arg, half %src1, half %src2) #0 {
  %src0.arg.bc = bitcast i32 %src0.arg to <2 x half>
  %fneg = fsub <2 x half> <half -0.0, half -0.0>, %src0.arg.bc
  %src0 = extractelement <2 x half> %fneg, i32 1
  %src0.ext = fpext half %src0 to float
  %src1.ext = fpext half %src1 to float
  %src2.ext = fpext half %src2 to float
  %result = tail call float @llvm.fmuladd.f32(float %src0.ext, float %src1.ext, float %src2.ext)
  ret float %result
}

; GCN-LABEL: {{^}}v_mad_mix_f32_preextractfabs_f16hi_f16lo_f16lo:
; GFX9: s_waitcnt
; GFX900-NEXT: v_mad_mix_f32 v0, |v0|, v1, v2 op_sel:[1,0,0] op_sel_hi:[1,1,1]
; GFX906-NEXT: v_fma_mix_f32 v0, |v0|, v1, v2 op_sel:[1,0,0] op_sel_hi:[1,1,1]
; GFX9-NEXT: s_setpc_b64
define float @v_mad_mix_f32_preextractfabs_f16hi_f16lo_f16lo(i32 %src0.arg, half %src1, half %src2) #0 {
  %src0.arg.bc = bitcast i32 %src0.arg to <2 x half>
  %fabs = call <2 x half> @llvm.fabs.v2f16(<2 x half> %src0.arg.bc)
  %src0 = extractelement <2 x half> %fabs, i32 1
  %src0.ext = fpext half %src0 to float
  %src1.ext = fpext half %src1 to float
  %src2.ext = fpext half %src2 to float
  %result = tail call float @llvm.fmuladd.f32(float %src0.ext, float %src1.ext, float %src2.ext)
  ret float %result
}

; GCN-LABEL: {{^}}v_mad_mix_f32_preextractfabsfneg_f16hi_f16lo_f16lo:
; GFX9: s_waitcnt
; GFX900-NEXT: v_mad_mix_f32 v0, -|v0|, v1, v2 op_sel:[1,0,0] op_sel_hi:[1,1,1]
; GFX906-NEXT: v_fma_mix_f32 v0, -|v0|, v1, v2 op_sel:[1,0,0] op_sel_hi:[1,1,1]
; GFX9-NEXT: s_setpc_b64
define float @v_mad_mix_f32_preextractfabsfneg_f16hi_f16lo_f16lo(i32 %src0.arg, half %src1, half %src2) #0 {
  %src0.arg.bc = bitcast i32 %src0.arg to <2 x half>
  %fabs = call <2 x half> @llvm.fabs.v2f16(<2 x half> %src0.arg.bc)
  %fneg.fabs = fsub <2 x half> <half -0.0, half -0.0>, %fabs
  %src0 = extractelement <2 x half> %fneg.fabs, i32 1
  %src0.ext = fpext half %src0 to float
  %src1.ext = fpext half %src1 to float
  %src2.ext = fpext half %src2 to float
  %result = tail call float @llvm.fmuladd.f32(float %src0.ext, float %src1.ext, float %src2.ext)
  ret float %result
}

declare half @llvm.fabs.f16(half) #2
declare <2 x half> @llvm.fabs.v2f16(<2 x half>) #2
declare float @llvm.fabs.f32(float) #2
declare float @llvm.minnum.f32(float, float) #2
declare float @llvm.maxnum.f32(float, float) #2
declare float @llvm.fmuladd.f32(float, float, float) #2
declare <2 x float> @llvm.fmuladd.v2f32(<2 x float>, <2 x float>, <2 x float>) #2

attributes #0 = { nounwind "denormal-fp-math-f32"="preserve-sign,preserve-sign" }
attributes #1 = { nounwind "denormal-fp-math-f32"="ieee,ieee" }
attributes #2 = { nounwind readnone speculatable }
