; RUN: llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx1010 < %s | FileCheck -check-prefix=GFX10-PAL %s
; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx1010 < %s | FileCheck -check-prefix=GFX10-MESA %s

; Check EXTRA_LDS_SIZE in SPI_SHADER_PGM_RSRC2_PS.

; GFX10-PAL: 0x2c0b (SPI_SHADER_PGM_RSRC2_PS): 0x800

; GFX10-MESA: .long 45100
; GFX10-MESA-NEXT: .long 2048

@lds = internal addrspace(3) global [4096 x i8] undef

define amdgpu_ps void @global_store_saddr_uniform_ptr_in_vgprs(i32 %voffset) {
  %ptr = getelementptr [4096 x i8], [4096 x i8] addrspace(3)* @lds, i32 0, i32 %voffset
  store i8 0, i8 addrspace(3)* %ptr
  ret void
}
