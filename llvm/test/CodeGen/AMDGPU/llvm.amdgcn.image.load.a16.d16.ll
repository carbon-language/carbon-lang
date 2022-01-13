; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX9 %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX10 %s

; GCN-LABEL: {{^}}load.f16.1d:
; GFX9: image_load v0, v0, s[0:7] dmask:0x1 unorm a16 d16
; GFX10: image_load v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D unorm a16 d16
define amdgpu_ps <4 x half> @load.f16.1d(<8 x i32> inreg %rsrc, <2 x i16> %coords) {
main_body:
  %x = extractelement <2 x i16> %coords, i32 0
  %v = call <4 x half> @llvm.amdgcn.image.load.1d.v4f16.i16(i32 1, i16 %x, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x half> %v
}

; GCN-LABEL: {{^}}load.v2f16.1d:
; GFX9: image_load v0, v0, s[0:7] dmask:0x3 unorm a16 d16
; GFX10: image_load v0, v0, s[0:7] dmask:0x3 dim:SQ_RSRC_IMG_1D unorm a16 d16
define amdgpu_ps <4 x half> @load.v2f16.1d(<8 x i32> inreg %rsrc, <2 x i16> %coords) {
main_body:
  %x = extractelement <2 x i16> %coords, i32 0
  %v = call <4 x half> @llvm.amdgcn.image.load.1d.v4f16.i16(i32 3, i16 %x, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x half> %v
}

; GCN-LABEL: {{^}}load.v3f16.1d:
; GFX9: image_load v[0:1], v0, s[0:7] dmask:0x7 unorm a16 d16
; GFX10: image_load v[0:1], v0, s[0:7] dmask:0x7 dim:SQ_RSRC_IMG_1D unorm a16 d16
define amdgpu_ps <4 x half> @load.v3f16.1d(<8 x i32> inreg %rsrc, <2 x i16> %coords) {
main_body:
  %x = extractelement <2 x i16> %coords, i32 0
  %v = call <4 x half> @llvm.amdgcn.image.load.1d.v4f16.i16(i32 7, i16 %x, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x half> %v
}

; GCN-LABEL: {{^}}load.v4f16.1d:
; GFX9: image_load v[0:1], v0, s[0:7] dmask:0xf unorm a16 d16
; GFX10: image_load v[0:1], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D unorm a16 d16
define amdgpu_ps <4 x half> @load.v4f16.1d(<8 x i32> inreg %rsrc, <2 x i16> %coords) {
main_body:
  %x = extractelement <2 x i16> %coords, i32 0
  %v = call <4 x half> @llvm.amdgcn.image.load.1d.v4f16.i16(i32 15, i16 %x, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x half> %v
}

; GCN-LABEL: {{^}}load.f16.2d:
; GFX9: image_load v0, v0, s[0:7] dmask:0x1 unorm a16 d16
; GFX10: image_load v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm a16 d16
define amdgpu_ps <4 x half> @load.f16.2d(<8 x i32> inreg %rsrc, <2 x i16> %coords) {
main_body:
  %x = extractelement <2 x i16> %coords, i32 0
  %y = extractelement <2 x i16> %coords, i32 1
  %v = call <4 x half> @llvm.amdgcn.image.load.2d.v4f16.i16(i32 1, i16 %x, i16 %y, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x half> %v
}

; GCN-LABEL: {{^}}load.v2f16.2d:
; GFX9: image_load v0, v0, s[0:7] dmask:0x3 unorm a16 d16
; GFX10: image_load v0, v0, s[0:7] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm a16 d16
define amdgpu_ps <4 x half> @load.v2f16.2d(<8 x i32> inreg %rsrc, <2 x i16> %coords) {
main_body:
  %x = extractelement <2 x i16> %coords, i32 0
  %y = extractelement <2 x i16> %coords, i32 1
  %v = call <4 x half> @llvm.amdgcn.image.load.2d.v4f16.i16(i32 3, i16 %x, i16 %y, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x half> %v
}

; GCN-LABEL: {{^}}load.v3f16.2d:
; GFX9: image_load v[0:1], v0, s[0:7] dmask:0x7 unorm a16 d16
; GFX10: image_load v[0:1], v0, s[0:7] dmask:0x7 dim:SQ_RSRC_IMG_2D unorm a16 d16
define amdgpu_ps <4 x half> @load.v3f16.2d(<8 x i32> inreg %rsrc, <2 x i16> %coords) {
main_body:
  %x = extractelement <2 x i16> %coords, i32 0
  %y = extractelement <2 x i16> %coords, i32 1
  %v = call <4 x half> @llvm.amdgcn.image.load.2d.v4f16.i16(i32 7, i16 %x, i16 %y, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x half> %v
}

; GCN-LABEL: {{^}}load.v4f16.2d:
; GFX9: image_load v[0:1], v0, s[0:7] dmask:0xf unorm a16 d16
; GFX10: image_load v[0:1], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D unorm a16 d16
define amdgpu_ps <4 x half> @load.v4f16.2d(<8 x i32> inreg %rsrc, <2 x i16> %coords) {
main_body:
  %x = extractelement <2 x i16> %coords, i32 0
  %y = extractelement <2 x i16> %coords, i32 1
  %v = call <4 x half> @llvm.amdgcn.image.load.2d.v4f16.i16(i32 15, i16 %x, i16 %y, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x half> %v
}

; GCN-LABEL: {{^}}load.f16.3d:
; GFX9: image_load v0, v[0:1], s[0:7] dmask:0x1 unorm a16 d16
; GFX10: image_load v0, v[0:1], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm a16 d16
define amdgpu_ps <4 x half> @load.f16.3d(<8 x i32> inreg %rsrc, <2 x i16> %coords_lo, <2 x i16> %coords_hi) {
main_body:
  %x = extractelement <2 x i16> %coords_lo, i32 0
  %y = extractelement <2 x i16> %coords_lo, i32 1
  %z = extractelement <2 x i16> %coords_hi, i32 0
  %v = call <4 x half> @llvm.amdgcn.image.load.3d.v4f16.i16(i32 1, i16 %x, i16 %y, i16 %z, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x half> %v
}

; GCN-LABEL: {{^}}load.v2f16.3d:
; GFX9: image_load v0, v[0:1], s[0:7] dmask:0x3 unorm a16 d16
; GFX10: image_load v0, v[0:1], s[0:7] dmask:0x3 dim:SQ_RSRC_IMG_3D unorm a16 d16
define amdgpu_ps <4 x half> @load.v2f16.3d(<8 x i32> inreg %rsrc, <2 x i16> %coords_lo, <2 x i16> %coords_hi) {
main_body:
  %x = extractelement <2 x i16> %coords_lo, i32 0
  %y = extractelement <2 x i16> %coords_lo, i32 1
  %z = extractelement <2 x i16> %coords_hi, i32 0
  %v = call <4 x half> @llvm.amdgcn.image.load.3d.v4f16.i16(i32 3, i16 %x, i16 %y, i16 %z, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x half> %v
}

; GCN-LABEL: {{^}}load.v3f16.3d:
; GFX9: image_load v[0:1], v[0:1], s[0:7] dmask:0x7 unorm a16 d16
; GFX10: image_load v[0:1], v[0:1], s[0:7] dmask:0x7 dim:SQ_RSRC_IMG_3D unorm a16 d16
define amdgpu_ps <4 x half> @load.v3f16.3d(<8 x i32> inreg %rsrc, <2 x i16> %coords_lo, <2 x i16> %coords_hi) {
main_body:
  %x = extractelement <2 x i16> %coords_lo, i32 0
  %y = extractelement <2 x i16> %coords_lo, i32 1
  %z = extractelement <2 x i16> %coords_hi, i32 0
  %v = call <4 x half> @llvm.amdgcn.image.load.3d.v4f16.i16(i32 7, i16 %x, i16 %y, i16 %z, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x half> %v
}

; GCN-LABEL: {{^}}load.v4f16.3d:
; GFX9: image_load v[0:1], v[0:1], s[0:7] dmask:0xf unorm a16 d16
; GFX10: image_load v[0:1], v[0:1], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_3D unorm a16 d16
define amdgpu_ps <4 x half> @load.v4f16.3d(<8 x i32> inreg %rsrc, <2 x i16> %coords_lo, <2 x i16> %coords_hi) {
main_body:
  %x = extractelement <2 x i16> %coords_lo, i32 0
  %y = extractelement <2 x i16> %coords_lo, i32 1
  %z = extractelement <2 x i16> %coords_hi, i32 0
  %v = call <4 x half> @llvm.amdgcn.image.load.3d.v4f16.i16(i32 15, i16 %x, i16 %y, i16 %z, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x half> %v
}

declare <4 x half> @llvm.amdgcn.image.load.1d.v4f16.i16(i32, i16, <8 x i32>, i32, i32) #2
declare <4 x half> @llvm.amdgcn.image.load.2d.v4f16.i16(i32, i16, i16, <8 x i32>, i32, i32) #2
declare <4 x half> @llvm.amdgcn.image.load.3d.v4f16.i16(i32, i16, i16, i16, <8 x i32>, i32, i32) #2

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }
