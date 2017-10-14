; RUN: llc -mtriple=amdgcn-amd-unknown -mcpu=gfx800 < %s | FileCheck --check-prefix=GCN --check-prefix=OSABI-UNK --check-prefix=GFX800 %s
; RUN: llc -mtriple=amdgcn-amd-unknown -mcpu=iceland < %s | FileCheck --check-prefix=GCN --check-prefix=OSABI-UNK --check-prefix=GFX800 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx800 < %s | FileCheck --check-prefix=GCN --check-prefix=OSABI-HSA --check-prefix=GFX800 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=iceland < %s | FileCheck --check-prefix=GCN --check-prefix=OSABI-HSA --check-prefix=GFX800 %s
; RUN: llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx800 < %s | FileCheck --check-prefix=GCN --check-prefix=OSABI-PAL --check-prefix=GFX800 %s
; RUN: llc -mtriple=amdgcn-amd-amdpal -mcpu=iceland < %s | FileCheck --check-prefix=GCN --check-prefix=OSABI-PAL --check-prefix=GFX800 %s
; RUN: llc -march=r600 < %s | FileCheck --check-prefix=R600 %s

; OSABI-UNK: .amd_amdgpu_isa "amdgcn-amd-unknown--gfx800"
; OSABI-UNK-NOT: .amd_amdgpu_hsa_metadata
; OSABI-UNK-NOT: .amd_amdgpu_pal_metadata

; OSABI-HSA: .amd_amdgpu_isa "amdgcn-amd-amdhsa--gfx800"
; OSABI-HSA: .amd_amdgpu_hsa_metadata

; OSABI-PAL: .amd_amdgpu_isa "amdgcn-amd-amdpal--gfx800"
; OSABI-PAL: .amd_amdgpu_pal_metadata

; R600-NOT: .amd_amdgpu_isa
; R600-NOT: .amd_amdgpu_hsa_metadata
; R600-NOT: .amd_amdgpu_hsa_metadata

define amdgpu_kernel void @elf_notes() {
  ret void
}
