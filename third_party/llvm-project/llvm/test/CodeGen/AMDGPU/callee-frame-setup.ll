; RUN: llc -march=amdgcn -mcpu=hawaii -verify-machineinstrs < %s | FileCheck  -enable-var-scope -check-prefixes=GCN,MUBUF %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck  -enable-var-scope -check-prefixes=GCN,MUBUF %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs -mattr=+enable-flat-scratch < %s | FileCheck  -enable-var-scope -check-prefixes=GCN,FLATSCR %s

; GCN-LABEL: {{^}}callee_no_stack:
; GCN: ; %bb.0:
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @callee_no_stack() #0 {
  ret void
}

; GCN-LABEL: {{^}}callee_no_stack_no_fp_elim_all:
; GCN: ; %bb.0:
; GCN-NEXT: s_waitcnt
; MUBUF-NEXT:   s_mov_b32 [[FP_COPY:s4]], s33
; FLATSCR-NEXT: s_mov_b32 [[FP_COPY:s0]], s33
; GCN-NEXT: s_mov_b32 s33, s32
; GCN-NEXT: s_mov_b32 s33, [[FP_COPY]]
; GCN-NEXT: s_setpc_b64
define void @callee_no_stack_no_fp_elim_all() #1 {
  ret void
}

; GCN-LABEL: {{^}}callee_no_stack_no_fp_elim_nonleaf:
; GCN: ; %bb.0:
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @callee_no_stack_no_fp_elim_nonleaf() #2 {
  ret void
}

; GCN-LABEL: {{^}}callee_with_stack:
; GCN: ; %bb.0:
; GCN-NEXT: s_waitcnt
; GCN-NEXT: v_mov_b32_e32 v0, 0{{$}}
; MUBUF-NEXT:   buffer_store_dword v0, off, s[0:3], s32{{$}}
; FLATSCR-NEXT: scratch_store_dword off, v0, s32
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @callee_with_stack() #0 {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca
  ret void
}

; Can use free call clobbered register to preserve original FP value.

; GCN-LABEL: {{^}}callee_with_stack_no_fp_elim_all:
; GCN: ; %bb.0:
; GCN-NEXT: s_waitcnt
; MUBUF-NEXT:   s_mov_b32 [[FP_COPY:s4]], s33
; FLATSCR-NEXT: s_mov_b32 [[FP_COPY:s0]], s33
; GCN-NEXT: s_mov_b32 s33, s32
; MUBUF-NEXT:   s_addk_i32 s32, 0x200
; FLATSCR-NEXT: s_add_i32 s32, s32, 8
; GCN-NEXT: v_mov_b32_e32 v0, 0{{$}}
; MUBUF-NEXT:   buffer_store_dword v0, off, s[0:3], s33{{$}}
; FLATSCR-NEXT: scratch_store_dword off, v0, s33{{$}}
; GCN-NEXT: s_waitcnt vmcnt(0)
; MUBUF-NEXT:   s_addk_i32 s32, 0xfe00
; FLATSCR-NEXT: s_add_i32 s32, s32, -8
; GCN-NEXT: s_mov_b32 s33, [[FP_COPY]]
; GCN-NEXT: s_setpc_b64
define void @callee_with_stack_no_fp_elim_all() #1 {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca
  ret void
}

; GCN-LABEL: {{^}}callee_with_stack_no_fp_elim_non_leaf:
; GCN: ; %bb.0:
; GCN-NEXT: s_waitcnt
; GCN-NEXT: v_mov_b32_e32 v0, 0{{$}}
; MUBUF-NEXT:   buffer_store_dword v0, off, s[0:3], s32{{$}}
; FLATSCR-NEXT: scratch_store_dword off, v0, s32{{$}}
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @callee_with_stack_no_fp_elim_non_leaf() #2 {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca
  ret void
}

; GCN-LABEL: {{^}}callee_with_stack_and_call:
; GCN: ; %bb.0:
; GCN-NEXT: s_waitcnt
; GCN: s_or_saveexec_b64 [[COPY_EXEC0:s\[[0-9]+:[0-9]+\]]], -1{{$}}
; MUBUF-NEXT:   buffer_store_dword [[CSR_VGPR:v[0-9]+]], off, s[0:3], s32 offset:4 ; 4-byte Folded Spill
; FLATSCR-NEXT: scratch_store_dword off, [[CSR_VGPR:v[0-9]+]], s32 offset:4 ; 4-byte Folded Spill
; GCN-NEXT: s_mov_b64 exec, [[COPY_EXEC0]]
; GCN: v_writelane_b32 [[CSR_VGPR]], s33, 2
; GCN-DAG: s_mov_b32 s33, s32
; MUBUF-DAG:   s_addk_i32 s32, 0x400{{$}}
; FLATSCR-DAG: s_add_i32 s32, s32, 16{{$}}
; GCN-DAG: v_mov_b32_e32 [[ZERO:v[0-9]+]], 0{{$}}
; GCN-DAG: v_writelane_b32 [[CSR_VGPR]], s30,
; GCN-DAG: v_writelane_b32 [[CSR_VGPR]], s31,

; MUBUF-DAG:   buffer_store_dword [[ZERO]], off, s[0:3], s33{{$}}
; FLATSCR-DAG: scratch_store_dword off, [[ZERO]], s33{{$}}

; GCN: s_swappc_b64

; GCN-DAG: v_readlane_b32 s30, [[CSR_VGPR]]
; GCN-DAG: v_readlane_b32 s31, [[CSR_VGPR]]

; MUBUF:    s_addk_i32 s32, 0xfc00{{$}}
; FLATSCR:  s_add_i32 s32, s32, -16{{$}}
; GCN-NEXT: v_readlane_b32 s33, [[CSR_VGPR]], 2
; GCN-NEXT: s_or_saveexec_b64 [[COPY_EXEC1:s\[[0-9]+:[0-9]+\]]], -1{{$}}
; MUBUF-NEXT:   buffer_load_dword [[CSR_VGPR]], off, s[0:3], s32 offset:4 ; 4-byte Folded Reload
; FLATSCR-NEXT: scratch_load_dword [[CSR_VGPR]], off, s32 offset:4 ; 4-byte Folded Reload
; GCN-NEXT: s_mov_b64 exec, [[COPY_EXEC1]]
; GCN-NEXT: s_waitcnt vmcnt(0)

; GCN-NEXT: s_setpc_b64 s[30:31]
define void @callee_with_stack_and_call() #0 {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca
  call void @external_void_func_void()
  ret void
}

; Should be able to copy incoming stack pointer directly to inner
; call's stack pointer argument.

; There is stack usage only because of the need to evict a VGPR for
; spilling CSR SGPRs.

; GCN-LABEL: {{^}}callee_no_stack_with_call:
; GCN: s_waitcnt
; GCN-NEXT: s_or_saveexec_b64 [[COPY_EXEC0:s\[[0-9]+:[0-9]+\]]], -1{{$}}
; MUBUF-NEXT:   buffer_store_dword [[CSR_VGPR:v[0-9]+]], off, s[0:3], s32 ; 4-byte Folded Spill
; FLATSCR-NEXT: scratch_store_dword off, [[CSR_VGPR:v[0-9]+]], s32 ; 4-byte Folded Spill
; GCN-NEXT: s_mov_b64 exec, [[COPY_EXEC0]]
; MUBUF-DAG:   s_addk_i32 s32, 0x400
; FLATSCR-DAG: s_add_i32 s32, s32, 16
; GCN-DAG: v_writelane_b32 [[CSR_VGPR]], s33, [[FP_SPILL_LANE:[0-9]+]]

; GCN-DAG: v_writelane_b32 [[CSR_VGPR]], s30, 0
; GCN-DAG: v_writelane_b32 [[CSR_VGPR]], s31, 1
; GCN: s_swappc_b64

; GCN-DAG: v_readlane_b32 s30, [[CSR_VGPR]], 0
; GCN-DAG: v_readlane_b32 s31, [[CSR_VGPR]], 1

; MUBUF:   s_addk_i32 s32, 0xfc00
; FLATSCR: s_add_i32 s32, s32, -16
; GCN-NEXT: v_readlane_b32 s33, [[CSR_VGPR]], [[FP_SPILL_LANE]]
; GCN-NEXT: s_or_saveexec_b64 [[COPY_EXEC1:s\[[0-9]+:[0-9]+\]]], -1{{$}}
; MUBUF-NEXT:   buffer_load_dword [[CSR_VGPR]], off, s[0:3], s32 ; 4-byte Folded Reload
; FLATSCR-NEXT: scratch_load_dword [[CSR_VGPR]], off, s32 ; 4-byte Folded Reload
; GCN-NEXT: s_mov_b64 exec, [[COPY_EXEC1]]
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64 s[30:31]
define void @callee_no_stack_with_call() #0 {
  call void @external_void_func_void()
  ret void
}

declare hidden void @external_void_func_void() #0

; Make sure if a CSR vgpr is used for SGPR spilling, it is saved and
; restored. No FP is required.
;
; GCN-LABEL: {{^}}callee_func_sgpr_spill_no_calls:
; GCN: s_or_saveexec_b64 [[COPY_EXEC0:s\[[0-9]+:[0-9]+\]]], -1{{$}}
; MUBUF-NEXT:   buffer_store_dword [[CSR_VGPR:v[0-9]+]], off, s[0:3], s32 ; 4-byte Folded Spill
; FLATSCR-NEXT: scratch_store_dword off, [[CSR_VGPR:v[0-9]+]], s32 ; 4-byte Folded Spill
; GCN-NEXT: s_mov_b64 exec, [[COPY_EXEC0]]
; GCN: v_writelane_b32 [[CSR_VGPR]], s
; GCN: v_writelane_b32 [[CSR_VGPR]], s

; GCN: ;;#ASMSTART
; GCN: v_readlane_b32 s{{[0-9]+}}, [[CSR_VGPR]]
; GCN: v_readlane_b32 s{{[0-9]+}}, [[CSR_VGPR]]

; GCN: s_or_saveexec_b64 [[COPY_EXEC1:s\[[0-9]+:[0-9]+\]]], -1{{$}}
; MUBUF-NEXT:   buffer_load_dword [[CSR_VGPR]], off, s[0:3], s32 ; 4-byte Folded Reload
; FLATSCR-NEXT: scratch_load_dword [[CSR_VGPR]], off, s32 ; 4-byte Folded Reload
; GCN-NEXT: s_mov_b64 exec, [[COPY_EXEC1]]
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @callee_func_sgpr_spill_no_calls(i32 %in) #0 {
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7}"() #0
  call void asm sideeffect "", "~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}"() #0
  call void asm sideeffect "", "~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23}"() #0
  call void asm sideeffect "", "~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31}"() #0
  call void asm sideeffect "", "~{v32},~{v33},~{v34},~{v35},~{v36},~{v37},~{v38},~{v39}"() #0

  %wide.sgpr0 = call <16 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr1 = call <16 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr2 = call <16 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr5 = call <16 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr3 = call <8 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr4 = call <2 x i32> asm sideeffect "; def $0", "=s" () #0

  call void asm sideeffect "; use $0", "s"(<16 x i32> %wide.sgpr0) #0
  call void asm sideeffect "; use $0", "s"(<16 x i32> %wide.sgpr1) #0
  call void asm sideeffect "; use $0", "s"(<16 x i32> %wide.sgpr2) #0
  call void asm sideeffect "; use $0", "s"(<8 x i32> %wide.sgpr3) #0
  call void asm sideeffect "; use $0", "s"(<2 x i32> %wide.sgpr4) #0
  call void asm sideeffect "; use $0", "s"(<16 x i32> %wide.sgpr5) #0
  ret void
}

; Has no spilled CSR VGPRs used for SGPR spilling, so no need to
; enable all lanes and restore.

; GCN-LABEL: {{^}}spill_only_csr_sgpr:
; GCN: s_waitcnt
; GCN-NEXT: s_or_saveexec_b64
; MUBUF-NEXT: buffer_store_dword v0, off, s[0:3], s32 ; 4-byte Folded Spill
; FLATSCR-NEXT: scratch_store_dword off, v0, s32 ; 4-byte Folded Spill
; GCN-NEXT: s_mov_b64 exec,
; GCN-NEXT: v_writelane_b32 v0, s42, 0
; GCN-NEXT: ;;#ASMSTART
; GCN-NEXT: ; clobber s42
; GCN-NEXT: ;;#ASMEND
; GCN-NEXT: v_readlane_b32 s42, v0, 0
; GCN-NEXT: s_or_saveexec_b64
; MUBUF-NEXT: buffer_load_dword v0, off, s[0:3], s32 ; 4-byte Folded Reload
; FLATSCR-NEXT: scratch_load_dword v0, off, s32 ; 4-byte Folded Reload
; GCN-NEXT: s_mov_b64 exec,
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define void @spill_only_csr_sgpr() {
  call void asm sideeffect "; clobber s42", "~{s42}"()
  ret void
}

; TODO: Can the SP inc/deec be remvoed?
; GCN-LABEL: {{^}}callee_with_stack_no_fp_elim_csr_vgpr:
; GCN: s_waitcnt
; GCN-NEXT:s_mov_b32 [[FP_COPY:s[0-9]+]], s33
; GCN-NEXT: s_mov_b32 s33, s32
; GCN: v_mov_b32_e32 [[ZERO:v[0-9]+]], 0
; MUBUF-DAG:   buffer_store_dword v41, off, s[0:3], s33 ; 4-byte Folded Spill
; FLATSCR-DAG: scratch_store_dword off, v41, s33 ; 4-byte Folded Spill
; MUBUF-DAG:   buffer_store_dword [[ZERO]], off, s[0:3], s33 offset:4
; FLATSCR-DAG: scratch_store_dword off, [[ZERO]], s33 offset:4

; GCN:	;;#ASMSTART
; GCN-NEXT: ; clobber v41
; GCN-NEXT: ;;#ASMEND

; MUBUF:   buffer_load_dword v41, off, s[0:3], s33 ; 4-byte Folded Reload
; FLATSCR: scratch_load_dword v41, off, s33 ; 4-byte Folded Reload
; MUBUF:        s_addk_i32 s32, 0x300
; MUBUF-NEXT:   s_addk_i32 s32, 0xfd00
; MUBUF-NEXT:   s_mov_b32 s33, s4
; FLATSCR:      s_add_i32 s32, s32, 12
; FLATSCR-NEXT: s_add_i32 s32, s32, -12
; FLATSCR-NEXT: s_mov_b32 s33, s0
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define void @callee_with_stack_no_fp_elim_csr_vgpr() #1 {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca
  call void asm sideeffect "; clobber v41", "~{v41}"()
  ret void
}

; Use a copy to a free SGPR instead of introducing a second CSR VGPR.
; GCN-LABEL: {{^}}last_lane_vgpr_for_fp_csr:
; GCN: s_waitcnt
; GCN-NEXT: s_or_saveexec_b64 [[COPY_EXEC0:s\[[0-9]+:[0-9]+\]]], -1{{$}}
; MUBUF-NEXT:   buffer_store_dword [[CSR_VGPR:v[0-9]+]], off, s[0:3], s32 offset:8 ; 4-byte Folded Spill
; FLATSCR-NEXT: scratch_store_dword off, [[CSR_VGPR:v[0-9]+]], s32 offset:8 ; 4-byte Folded Spill
; GCN-NEXT: s_mov_b64 exec, [[COPY_EXEC0]]
; GCN-NEXT: v_writelane_b32 v0, s33, 63
; GCN-COUNT-60: v_writelane_b32 v0
; GCN: s_mov_b32 s33, s32
; GCN: v_writelane_b32 v0
; MUBUF:   buffer_store_dword v41, off, s[0:3], s33 ; 4-byte Folded Spill
; FLATSCR: scratch_store_dword off, v41, s33 ; 4-byte Folded Spill
; GCN: v_writelane_b32 v0
; MUBUF:   buffer_store_dword v{{[0-9]+}}, off, s[0:3], s33 offset:4
; FLATSCR: scratch_store_dword off, v{{[0-9]+}}, s33 offset:4
; GCN: ;;#ASMSTART
; GCN: v_writelane_b32 v0

; MUBUF:        s_addk_i32 s32, 0x400
; MUBUF:        s_addk_i32 s32, 0xfc00
; FLATSCR:      s_add_i32 s32, s32, 16
; FLATSCR:      s_add_i32 s32, s32, -16
; GCN-NEXT: v_readlane_b32 s33, v0, 63
; GCN-NEXT: s_or_saveexec_b64 [[COPY_EXEC1:s\[[0-9]+:[0-9]+\]]], -1{{$}}
; MUBUF-NEXT:   buffer_load_dword [[CSR_VGPR]], off, s[0:3], s32 offset:8 ; 4-byte Folded Reload
; FLATSCR-NEXT: scratch_load_dword [[CSR_VGPR]], off, s32 offset:8 ; 4-byte Folded Reload
; GCN-NEXT: s_mov_b64 exec, [[COPY_EXEC1]]
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define void @last_lane_vgpr_for_fp_csr() #1 {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca
  call void asm sideeffect "; clobber v41", "~{v41}"()
  call void asm sideeffect "",
    "~{s40},~{s41},~{s42},~{s43},~{s44},~{s45},~{s46},~{s47},~{s48},~{s49}
    ,~{s50},~{s51},~{s52},~{s53},~{s54},~{s55},~{s56},~{s57},~{s58},~{s59}
    ,~{s60},~{s61},~{s62},~{s63},~{s64},~{s65},~{s66},~{s67},~{s68},~{s69}
    ,~{s70},~{s71},~{s72},~{s73},~{s74},~{s75},~{s76},~{s77},~{s78},~{s79}
    ,~{s80},~{s81},~{s82},~{s83},~{s84},~{s85},~{s86},~{s87},~{s88},~{s89}
    ,~{s90},~{s91},~{s92},~{s93},~{s94},~{s95},~{s96},~{s97},~{s98},~{s99}
    ,~{s100},~{s101},~{s102}"() #1

  ret void
}

; Use a copy to a free SGPR instead of introducing a second CSR VGPR.
; GCN-LABEL: {{^}}no_new_vgpr_for_fp_csr:
; GCN: s_waitcnt
; GCN-NEXT: s_or_saveexec_b64 [[COPY_EXEC0:s\[[0-9]+:[0-9]+\]]], -1{{$}}
; MUBUF-NEXT:   buffer_store_dword [[CSR_VGPR:v[0-9]+]], off, s[0:3], s32 offset:8 ; 4-byte Folded Spill
; FLATSCR-NEXT: scratch_store_dword off, [[CSR_VGPR:v[0-9]+]], s32 offset:8 ; 4-byte Folded Spill
; GCN-NEXT: s_mov_b64 exec, [[COPY_EXEC0]]
; GCN-COUNT-61: v_writelane_b32 v0,
; FLATSCR: s_mov_b32 [[FP_COPY:s[0-9]+]], s33
; FLATSCR-NEXT: s_mov_b32 s33, s32
; MUBUF: s_mov_b32 [[FP_COPY:s[0-9]+]], s33
; MUBUF-NEXT: s_mov_b32 s33, s32
; MUBUF:   buffer_store_dword v41, off, s[0:3], s33 ; 4-byte Folded Spill
; FLATSCR: scratch_store_dword off, v41, s33 ; 4-byte Folded Spill
; GCN: v_writelane_b32 v0,
; MUBUF:   buffer_store_dword
; FLATSCR: scratch_store_dword
; GCN: ;;#ASMSTART
; GCN: v_writelane_b32 v0,
; MUBUF:   buffer_load_dword v41, off, s[0:3], s33 ; 4-byte Folded Reload
; FLATSCR: scratch_load_dword v41, off, s33 ; 4-byte Folded Reload
; MUBUF:        s_addk_i32 s32, 0x400
; FLATSCR:      s_add_i32 s32, s32, 16
; GCN-COUNT-64: v_readlane_b32 s{{[0-9]+}}, v0
; MUBUF-NEXT:   s_addk_i32 s32, 0xfc00
; FLATSCR-NEXT: s_add_i32 s32, s32, -16
; GCN-NEXT: s_mov_b32 s33, [[FP_COPY]]
; GCN-NEXT: s_or_saveexec_b64 [[COPY_EXEC1:s\[[0-9]+:[0-9]+\]]], -1{{$}}
; MUBUF-NEXT:   buffer_load_dword [[CSR_VGPR]], off, s[0:3], s32 offset:8 ; 4-byte Folded Reload
; FLATSCR-NEXT: scratch_load_dword [[CSR_VGPR]], off, s32 offset:8 ; 4-byte Folded Reload
; GCN-NEXT: s_mov_b64 exec, [[COPY_EXEC1]]
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define void @no_new_vgpr_for_fp_csr() #1 {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca
  call void asm sideeffect "; clobber v41", "~{v41}"()
  call void asm sideeffect "",
    "~{s39},~{s40},~{s41},~{s42},~{s43},~{s44},~{s45},~{s46},~{s47},~{s48},~{s49}
    ,~{s50},~{s51},~{s52},~{s53},~{s54},~{s55},~{s56},~{s57},~{s58},~{s59}
    ,~{s60},~{s61},~{s62},~{s63},~{s64},~{s65},~{s66},~{s67},~{s68},~{s69}
    ,~{s70},~{s71},~{s72},~{s73},~{s74},~{s75},~{s76},~{s77},~{s78},~{s79}
    ,~{s80},~{s81},~{s82},~{s83},~{s84},~{s85},~{s86},~{s87},~{s88},~{s89}
    ,~{s90},~{s91},~{s92},~{s93},~{s94},~{s95},~{s96},~{s97},~{s98},~{s99}
    ,~{s100},~{s101},~{s102}"() #1

  ret void
}

; GCN-LABEL: {{^}}realign_stack_no_fp_elim:
; GCN: s_waitcnt
; MUBUF-NEXT:   s_mov_b32 [[FP_COPY:s4]], s33
; FLATSCR-NEXT: s_mov_b32 [[FP_COPY:s0]], s33
; MUBUF-NEXT:   s_add_i32 s33, s32, 0x7ffc0
; FLATSCR-NEXT: s_add_i32 s33, s32, 0x1fff
; MUBUF-NEXT:   s_and_b32 s33, s33, 0xfff80000
; FLATSCR-NEXT: s_and_b32 s33, s33, 0xffffe000
; MUBUF-NEXT:   s_add_i32 s32, s32, 0x180000
; FLATSCR-NEXT: s_addk_i32 s32, 0x6000
; GCN-NEXT:     v_mov_b32_e32 [[ZERO:v[0-9]+]], 0
; MUBUF-NEXT:   v_mov_b32_e32 [[OFFSET:v[0-9]+]], 0x2000{{$}}
; MUBUF-NEXT:   buffer_store_dword [[ZERO]], [[OFFSET]], s[0:3], s33 offen{{$}}
; FLATSCR-NEXT: s_add_i32 vcc_hi, s33, 0x2000
; FLATSCR-NEXT: scratch_store_dword off, [[ZERO]], vcc_hi
; GCN-NEXT: s_waitcnt vmcnt(0)
; MUBUF-NEXT:   s_add_i32 s32, s32, 0xffe80000
; FLATSCR-NEXT: s_addk_i32 s32, 0xa000
; GCN-NEXT: s_mov_b32 s33, [[FP_COPY]]
; GCN-NEXT: s_setpc_b64
define void @realign_stack_no_fp_elim() #1 {
  %alloca = alloca i32, align 8192, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca
  ret void
}

; GCN-LABEL: {{^}}no_unused_non_csr_sgpr_for_fp:
; GCN: s_waitcnt
; GCN-NEXT: s_or_saveexec_b64 [[COPY_EXEC0:s\[[0-9]+:[0-9]+\]]], -1{{$}}
; MUBUF-NEXT:   buffer_store_dword [[CSR_VGPR:v[0-9]+]], off, s[0:3], s32 offset:4 ; 4-byte Folded Spill
; FLATSCR-NEXT: scratch_store_dword off, [[CSR_VGPR:v[0-9]+]], s32 offset:4 ; 4-byte Folded Spill
; GCN-NEXT: s_mov_b64 exec, [[COPY_EXEC0]]
; GCN-NEXT: v_writelane_b32 [[CSR_VGPR]], s33, 2
; GCN-NEXT: s_mov_b32 s33, s32
; GCN: v_writelane_b32 [[CSR_VGPR]], s30, 0
; GCN: v_mov_b32_e32 [[ZERO:v[0-9]+]], 0
; MUBUF:        s_addk_i32 s32, 0x300
; FLATSCR:      s_add_i32 s32, s32, 12
; GCN: v_writelane_b32 [[CSR_VGPR]], s31, 1
; MUBUF:   buffer_store_dword [[ZERO]], off, s[0:3], s33{{$}}
; FLATSCR: scratch_store_dword off, [[ZERO]], s33{{$}}
; GCN-NEXT:     s_waitcnt vmcnt(0)
; GCN: ;;#ASMSTART
; GCN:     v_readlane_b32 s31, [[CSR_VGPR]], 1
; GCN:     v_readlane_b32 s30, [[CSR_VGPR]], 0
; MUBUF:   s_addk_i32 s32, 0xfd00
; FLATSCR: s_add_i32 s32, s32, -12
; GCN-NEXT:     v_readlane_b32 s33, [[CSR_VGPR]], 2
; GCN-NEXT: s_or_saveexec_b64 [[COPY_EXEC1:s\[[0-9]+:[0-9]+\]]], -1{{$}}
; MUBUF-NEXT:   buffer_load_dword [[CSR_VGPR]], off, s[0:3], s32 offset:4 ; 4-byte Folded Reload
; FLATSCR-NEXT: scratch_load_dword [[CSR_VGPR]], off, s32 offset:4 ; 4-byte Folded Reload
; GCN-NEXT: s_mov_b64 exec, [[COPY_EXEC1]]
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64 s[30:31]
define void @no_unused_non_csr_sgpr_for_fp() #1 {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca

  ; Use all clobberable registers, so FP has to spill to a VGPR.
  call void asm sideeffect "",
    "~{s0},~{s1},~{s2},~{s3},~{s4},~{s5},~{s6},~{s7},~{s8},~{s9}
    ,~{s10},~{s11},~{s12},~{s13},~{s14},~{s15},~{s16},~{s17},~{s18},~{s19}
    ,~{s20},~{s21},~{s22},~{s23},~{s24},~{s25},~{s26},~{s27},~{s28},~{s29}
    ,~{s30},~{s31}"() #0

  ret void
}

; Need a new CSR VGPR to satisfy the FP spill.
; GCN-LABEL: {{^}}no_unused_non_csr_sgpr_for_fp_no_scratch_vgpr:
; GCN: s_waitcnt
; GCN-NEXT: s_or_saveexec_b64 [[COPY_EXEC0:s\[[0-9]+:[0-9]+\]]], -1{{$}}
; MUBUF-NEXT:   buffer_store_dword [[CSR_VGPR:v[0-9]+]], off, s[0:3], s32 offset:4 ; 4-byte Folded Spill
; FLATSCR-NEXT: scratch_store_dword off, [[CSR_VGPR:v[0-9]+]], s32 offset:4 ; 4-byte Folded Spill
; GCN-NEXT: s_mov_b64 exec, [[COPY_EXEC0]]
; GCN-NEXT: v_writelane_b32 [[CSR_VGPR]], s33, 2
; GCN-NEXT: s_mov_b32 s33, s32
; MUBUF:       s_addk_i32 s32, 0x300{{$}}
; FLATSCR:     s_add_i32 s32, s32, 12{{$}}

; MUBUF-DAG:   buffer_store_dword
; FLATSCR-DAG: scratch_store_dword

; GCN: ;;#ASMSTART
; MUBUF:   s_addk_i32 s32, 0xfd00{{$}}
; FLATSCR: s_add_i32 s32, s32, -12{{$}}
; GCN-NEXT: v_readlane_b32 s33, [[CSR_VGPR]], 2
; GCN-NEXT: s_or_saveexec_b64 [[COPY_EXEC1:s\[[0-9]+:[0-9]+\]]], -1{{$}}
; MUBUF-NEXT:   buffer_load_dword [[CSR_VGPR]], off, s[0:3], s32 offset:4 ; 4-byte Folded Reload
; FLATSCR-NEXT: scratch_load_dword [[CSR_VGPR]], off, s32 offset:4 ; 4-byte Folded Reload
; GCN-NEXT: s_mov_b64 exec, [[COPY_EXEC1]]
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64 s[30:31]
define void @no_unused_non_csr_sgpr_for_fp_no_scratch_vgpr() #1 {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca

  ; Use all clobberable registers, so FP has to spill to a VGPR.
  call void asm sideeffect "",
    "~{s0},~{s1},~{s2},~{s3},~{s4},~{s5},~{s6},~{s7},~{s8},~{s9}
    ,~{s10},~{s11},~{s12},~{s13},~{s14},~{s15},~{s16},~{s17},~{s18},~{s19}
    ,~{s20},~{s21},~{s22},~{s23},~{s24},~{s25},~{s26},~{s27},~{s28},~{s29}
    ,~{s30},~{s31}"() #0

  call void asm sideeffect "; clobber nonpreserved initial VGPRs",
    "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9}
    ,~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19}
    ,~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29}
    ,~{v30},~{v31},~{v32},~{v33},~{v34},~{v35},~{v36},~{v37},~{v38},~{v39}"() #1

  ret void
}

; The byval argument exceeds the MUBUF constant offset, so a scratch
; register is needed to access the CSR VGPR slot.
; GCN-LABEL: {{^}}scratch_reg_needed_mubuf_offset:
; GCN: s_waitcnt
; GCN-NEXT: s_or_saveexec_b64 [[COPY_EXEC0:s\[[0-9]+:[0-9]+\]]], -1{{$}}
; MUBUF-NEXT: s_add_i32 [[SCRATCH_SGPR:s[0-9]+]], s32, 0x40100
; MUBUF-NEXT: buffer_store_dword [[CSR_VGPR:v[0-9]+]], off, s[0:3], [[SCRATCH_SGPR]] ; 4-byte Folded Spill
; FLATSCR-NEXT: s_add_i32 [[SCRATCH_SGPR:s[0-9]+]], s32, 0x1004
; FLATSCR-NEXT: scratch_store_dword off, [[CSR_VGPR:v[0-9]+]], [[SCRATCH_SGPR]] ; 4-byte Folded Spill
; GCN-NEXT: s_mov_b64 exec, [[COPY_EXEC0]]
; GCN-NEXT: v_writelane_b32 [[CSR_VGPR]], s33, 2
; GCN-DAG:  s_mov_b32 s33, s32
; MUBUF-DAG:   s_add_i32 s32, s32, 0x40300{{$}}
; FLATSCR-DAG: s_addk_i32 s32, 0x100c{{$}}
; MUBUF-DAG:   buffer_store_dword
; FLATSCR-DAG: scratch_store_dword

; GCN: ;;#ASMSTART
; MUBUF:   s_add_i32 s32, s32, 0xfffbfd00{{$}}
; FLATSCR: s_addk_i32 s32, 0xeff4{{$}}
; GCN-NEXT: v_readlane_b32 s33, [[CSR_VGPR]], 2
; GCN-NEXT: s_or_saveexec_b64 [[COPY_EXEC1:s\[[0-9]+:[0-9]+\]]], -1{{$}}
; MUBUF-NEXT: s_add_i32 [[SCRATCH_SGPR:s[0-9]+]], s32, 0x40100
; MUBUF-NEXT: buffer_load_dword [[CSR_VGPR]], off, s[0:3], [[SCRATCH_SGPR]] ; 4-byte Folded Reload
; FLATSCR-NEXT: s_add_i32 [[SCRATCH_SGPR:s[0-9]+]], s32, 0x1004
; FLATSCR-NEXT: scratch_load_dword [[CSR_VGPR]], off, [[SCRATCH_SGPR]] ; 4-byte Folded Reload
; GCN-NEXT: s_mov_b64 exec, [[COPY_EXEC1]]
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64 s[30:31]
define void @scratch_reg_needed_mubuf_offset([4096 x i8] addrspace(5)* byval([4096 x i8]) align 4 %arg) #1 {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca

  ; Use all clobberable registers, so FP has to spill to a VGPR.
  call void asm sideeffect "; clobber nonpreserved SGPRs",
    "~{s0},~{s1},~{s2},~{s3},~{s4},~{s5},~{s6},~{s7},~{s8},~{s9}
    ,~{s10},~{s11},~{s12},~{s13},~{s14},~{s15},~{s16},~{s17},~{s18},~{s19}
    ,~{s20},~{s21},~{s22},~{s23},~{s24},~{s25},~{s26},~{s27},~{s28},~{s29}
    ,~{s30},~{s31}"() #0

  ; Use all clobberable VGPRs, so a CSR spill is needed for the VGPR
  call void asm sideeffect "; clobber nonpreserved VGPRs",
    "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9}
    ,~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19}
    ,~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29}
    ,~{v30},~{v31},~{v32},~{v33},~{v34},~{v35},~{v36},~{v37},~{v38},~{v39}"() #1

  ret void
}

; GCN-LABEL: {{^}}local_empty_func:
; GCN: s_waitcnt
; GCN-NEXT: s_setpc_b64
define internal void @local_empty_func() #0 {
  ret void
}

; An FP is needed, despite not needing any spills
; TODO: Ccould see callee does not use stack and omit FP.
; GCN-LABEL: {{^}}ipra_call_with_stack:
; GCN: v_writelane_b32 v0, s33, 2
; GCN: s_mov_b32 s33, s32
; MUBUF:   s_addk_i32 s32, 0x400
; FLATSCR: s_add_i32 s32, s32, 16
; MUBUF:   buffer_store_dword v{{[0-9]+}}, off, s[0:3], s33{{$}}
; FLATSCR: scratch_store_dword off, v{{[0-9]+}}, s33{{$}}
; GCN:     s_swappc_b64
; MUBUF:   s_addk_i32 s32, 0xfc00
; FLATSCR: s_add_i32 s32, s32, -16
; GCN: v_readlane_b32 s33, v0, 2
define void @ipra_call_with_stack() #0 {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca
  call void @local_empty_func()
  ret void
}

; With no free registers, we must spill the FP to memory.
; GCN-LABEL: {{^}}callee_need_to_spill_fp_to_memory:
; MUBUF:   v_mov_b32_e32 [[TMP_VGPR1:v[0-9]+]], s33
; MUBUF:   buffer_store_dword [[TMP_VGPR1]], off, s[0:3], s32 ; 4-byte Folded Spill
; FLATSCR: s_mov_b32 s0, s33
; GCN:     s_mov_b32 s33, s32
; MUBUF:   buffer_load_dword [[TMP_VGPR2:v[0-9]+]], off, s[0:3], s32 ; 4-byte Folded Reload
; FLATSCR: s_mov_b32 s33, s0
; MUBUF:   s_waitcnt vmcnt(0)
; MUBUF:   v_readfirstlane_b32 s33, [[TMP_VGPR2]]
; GCN:     s_setpc_b64
; MUBUF:   ScratchSize: 8
; FLATSCR: ScratchSize: 0
define void @callee_need_to_spill_fp_to_memory() #3 {
  call void asm sideeffect "; clobber nonpreserved SGPRs",
    "~{s4},~{s5},~{s6},~{s7},~{s8},~{s9}
    ,~{s10},~{s11},~{s12},~{s13},~{s14},~{s15},~{s16},~{s17},~{s18},~{s19}
    ,~{s20},~{s21},~{s22},~{s23},~{s24},~{s25},~{s26},~{s27},~{s28},~{s29}
    ,~{vcc}"()

  call void asm sideeffect "; clobber all VGPRs",
    "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9}
    ,~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19}
    ,~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29}
    ,~{v30},~{v31},~{v32},~{v33},~{v34},~{v35},~{v36},~{v37},~{v38},~{v39}"()
  ret void
}

; If we have a reserved VGPR that can be used for SGPR spills, we may still
; need to spill the FP to memory if there are no free lanes in the reserved
; VGPR.
; GCN-LABEL: {{^}}callee_need_to_spill_fp_to_memory_full_reserved_vgpr:
; MUBUF:   s_or_saveexec_b64 [[COPY_EXEC1:s\[[0-9]+:[0-9]+\]]], -1{{$}}
; MUBUF:   s_mov_b64 exec, [[COPY_EXEC1]]
; MUBUF:   v_mov_b32_e32 [[TMP_VGPR1:v[0-9]+]], s33
; MUBUF:   buffer_store_dword [[TMP_VGPR1]], off, s[0:3], s32 offset:[[OFF:[0-9]+]]
; GCN-NOT: v_writelane_b32 v40, s33
; MUBUF:   s_mov_b32 s33, s32
; FLATSCR: s_mov_b32 s33, s0
; GCN-NOT: v_readlane_b32 s33, v40
; MUBUF:   buffer_load_dword [[TMP_VGPR2:v[0-9]+]], off, s[0:3], s32 offset:[[OFF]]
; MUBUF:   v_readfirstlane_b32 s33, [[TMP_VGPR2]]
; MUBUF:   s_or_saveexec_b64 [[COPY_EXEC2:s\[[0-9]+:[0-9]+\]]], -1{{$}}
; MUBUF:   s_mov_b64 exec, [[COPY_EXEC2]]
; GCN:     s_setpc_b64
define void @callee_need_to_spill_fp_to_memory_full_reserved_vgpr() #3 {
  call void asm sideeffect "; clobber nonpreserved SGPRs and 64 CSRs",
    "~{s4},~{s5},~{s6},~{s7},~{s8},~{s9}
    ,~{s10},~{s11},~{s12},~{s13},~{s14},~{s15},~{s16},~{s17},~{s18},~{s19}
    ,~{s20},~{s21},~{s22},~{s23},~{s24},~{s25},~{s26},~{s27},~{s28},~{s29}
    ,~{s40},~{s41},~{s42},~{s43},~{s44},~{s45},~{s46},~{s47},~{s48},~{s49}
    ,~{s50},~{s51},~{s52},~{s53},~{s54},~{s55},~{s56},~{s57},~{s58},~{s59}
    ,~{s60},~{s61},~{s62},~{s63},~{s64},~{s65},~{s66},~{s67},~{s68},~{s69}
    ,~{s70},~{s71},~{s72},~{s73},~{s74},~{s75},~{s76},~{s77},~{s78},~{s79}
    ,~{s80},~{s81},~{s82},~{s83},~{s84},~{s85},~{s86},~{s87},~{s88},~{s89}
    ,~{s90},~{s91},~{s92},~{s93},~{s94},~{s95},~{s96},~{s97},~{s98},~{s99}
    ,~{s100},~{s101},~{s102},~{s39},~{vcc}"()

  call void asm sideeffect "; clobber all VGPRs except CSR v40",
    "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9}
    ,~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19}
    ,~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29}
    ,~{v30},~{v31},~{v32},~{v33},~{v34},~{v35},~{v36},~{v37},~{v38}"()
  ret void
}

; When flat-scratch is enabled, we save the FP to s0. At the same time,
; the exec register is saved to s0 when saving CSR in the function prolog.
; Make sure that the FP save happens after restoring exec from the same
; register.
; GCN-LABEL: {{^}}callee_need_to_spill_fp_to_reg:
; GCN-NOT: v_writelane_b32 v40, s33
; FLATSCR: s_or_saveexec_b64 s[0:1], -1
; FLATSCR: s_mov_b64 exec, s[0:1]
; FLATSCR: s_mov_b32 s0, s33
; FLATSCR: s_mov_b32 s33, s32
; FLATSCR: s_mov_b32 s33, s0
; FLATSCR: s_or_saveexec_b64 s[0:1], -1
; GCN-NOT: v_readlane_b32 s33, v40
; GCN:     s_setpc_b64
define void @callee_need_to_spill_fp_to_reg() #1 {
  call void asm sideeffect "; clobber nonpreserved SGPRs and 64 CSRs",
    "~{s4},~{s5},~{s6},~{s7},~{s8},~{s9}
    ,~{s10},~{s11},~{s12},~{s13},~{s14},~{s15},~{s16},~{s17},~{s18},~{s19}
    ,~{s20},~{s21},~{s22},~{s23},~{s24},~{s25},~{s26},~{s27},~{s28},~{s29}
    ,~{s40},~{s41},~{s42},~{s43},~{s44},~{s45},~{s46},~{s47},~{s48},~{s49}
    ,~{s50},~{s51},~{s52},~{s53},~{s54},~{s55},~{s56},~{s57},~{s58},~{s59}
    ,~{s60},~{s61},~{s62},~{s63},~{s64},~{s65},~{s66},~{s67},~{s68},~{s69}
    ,~{s70},~{s71},~{s72},~{s73},~{s74},~{s75},~{s76},~{s77},~{s78},~{s79}
    ,~{s80},~{s81},~{s82},~{s83},~{s84},~{s85},~{s86},~{s87},~{s88},~{s89}
    ,~{s90},~{s91},~{s92},~{s93},~{s94},~{s95},~{s96},~{s97},~{s98},~{s99}
    ,~{s100},~{s101},~{s102},~{s39},~{vcc}"()

  call void asm sideeffect "; clobber all VGPRs except CSR v40",
    "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9}
    ,~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19}
    ,~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29}
    ,~{v30},~{v31},~{v32},~{v33},~{v34},~{v35},~{v36},~{v37},~{v38},~{v39}"()
  ret void
}

; If the size of the offset exceeds the MUBUF offset field we need another
; scratch VGPR to hold the offset.
; GCN-LABEL: {{^}}spill_fp_to_memory_scratch_reg_needed_mubuf_offset
; MUBUF: s_or_saveexec_b64 s[4:5], -1
; MUBUF-NEXT: s_add_i32 [[SCRATCH_SGPR:s[0-9]+]], s32, 0x40100
; MUBUF-NEXT: buffer_store_dword v39, off, s[0:3], [[SCRATCH_SGPR]] ; 4-byte Folded Spill
; MUBUF: v_mov_b32_e32 v0, s33
; GCN-NOT: v_mov_b32_e32 v0, 0x100c
; MUBUF-NEXT: s_add_i32 [[SCRATCH_SGPR:s[0-9]+]], s32, 0x40200
; MUBUF: buffer_store_dword v0, off, s[0:3], [[SCRATCH_SGPR]] ; 4-byte Folded Spill
; FLATSCR: v_mov_b32_e32 v0, 0
; FLATSCR: s_add_i32 [[SOFF:s[0-9]+]], s33, 0x1000
; FLATSCR: scratch_store_dword off, v0, [[SOFF]]
define void @spill_fp_to_memory_scratch_reg_needed_mubuf_offset([4096 x i8] addrspace(5)* byval([4096 x i8]) align 4 %arg) #3 {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca

  call void asm sideeffect "; clobber nonpreserved SGPRs and 64 CSRs",
    "~{s4},~{s5},~{s6},~{s7},~{s8},~{s9}
    ,~{s10},~{s11},~{s12},~{s13},~{s14},~{s15},~{s16},~{s17},~{s18},~{s19}
    ,~{s20},~{s21},~{s22},~{s23},~{s24},~{s25},~{s26},~{s27},~{s28},~{s29}
    ,~{s40},~{s41},~{s42},~{s43},~{s44},~{s45},~{s46},~{s47},~{s48},~{s49}
    ,~{s50},~{s51},~{s52},~{s53},~{s54},~{s55},~{s56},~{s57},~{s58},~{s59}
    ,~{s60},~{s61},~{s62},~{s63},~{s64},~{s65},~{s66},~{s67},~{s68},~{s69}
    ,~{s70},~{s71},~{s72},~{s73},~{s74},~{s75},~{s76},~{s77},~{s78},~{s79}
    ,~{s80},~{s81},~{s82},~{s83},~{s84},~{s85},~{s86},~{s87},~{s88},~{s89}
    ,~{s90},~{s91},~{s92},~{s93},~{s94},~{s95},~{s96},~{s97},~{s98},~{s99}
    ,~{s100},~{s101},~{s102},~{s39},~{vcc}"()

  call void asm sideeffect "; clobber all VGPRs except CSR v40",
    "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9}
    ,~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19}
    ,~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29}
    ,~{v30},~{v31},~{v32},~{v33},~{v34},~{v35},~{v36},~{v37},~{v38}"()
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind "frame-pointer"="all" }
attributes #2 = { nounwind "frame-pointer"="non-leaf" }
attributes #3 = { nounwind "frame-pointer"="all" "amdgpu-waves-per-eu"="6,6" }
