; RUN: llc -march=amdgcn -mcpu=bonaire < %s | FileCheck -check-prefix=GCN -check-prefix=CI -check-prefix=ALL %s
; RUN: llc -march=amdgcn -mcpu=tonga < %s | FileCheck -check-prefix=GCN -check-prefix=VI -check-prefix=ALL %s
; XUN: llc -march=amdgcn -mcpu=bonaire -mtriple=amdgcn-unknown-amdhsa < %s | FileCheck -check-prefix=GCNHSA -check-prefix=CIHSA -check-prefix=ALL %s
; XUN: llc -march=amdgcn -mcpu=tonga -mtriple=amdgcn-unknown-amdhsa < %s | FileCheck -check-prefix=GCNHSA -check-prefix=VIHSA -check-prefix=ALL %s

; FIXME: align on alloca seems to be ignored for private_segment_alignment

; ALL-LABEL: {{^}}large_alloca_compute_shader:

; GCN: s_mov_b32 s12, SCRATCH_RSRC_DWORD0
; GCN: s_mov_b32 s13, SCRATCH_RSRC_DWORD1
; GCN: s_mov_b32 s14, -1
; CI: s_mov_b32 s15, 0x80f000
; VI: s_mov_b32 s15, 0x800000


; GCNHSA: .amd_kernel_code_t
; GCNHSA: private_segment_alignment = 4
; GCNHSA: .end_amd_kernel_code_t

; GCNHSA: s_mov_b32 s8, SCRATCH_RSRC_DWORD0
; GCNHSA: s_mov_b32 s9, SCRATCH_RSRC_DWORD1
; GCNHSA: s_mov_b32 s10, -1
; CIHSA: s_mov_b32 s11, 0x180f000
; VIHSA: s_mov_b32 s11, 0x11800000

; GCNHSA: buffer_store_dword {{v[0-9]+}}, {{v[0-9]+}}, s[8:11], s6 offen
; GCNHSA: buffer_load_dword {{v[0-9]+}}, {{v[0-9]+}}, s[8:11], s6 offen

; Scratch size = alloca size + emergency stack slot
; ALL: ; ScratchSize: 32772
define void @large_alloca_compute_shader(i32 %x, i32 %y) #0 {
  %large = alloca [8192 x i32], align 4
  %gep = getelementptr [8192 x i32], [8192 x i32]* %large, i32 0, i32 8191
  store volatile i32 %x, i32* %gep
  %gep1 = getelementptr [8192 x i32], [8192 x i32]* %large, i32 0, i32 %y
  %val = load volatile i32, i32* %gep1
  store volatile i32 %val, i32 addrspace(1)* undef
  ret void
}

attributes #0 = { nounwind  }
