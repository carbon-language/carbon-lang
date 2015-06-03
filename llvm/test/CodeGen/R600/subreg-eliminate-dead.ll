; RUN: llc -mtriple=amdgcn-- -verify-machineinstrs -o - %s | FileCheck %s
; LiveRangeEdit::eliminateDeadDef did not update LiveInterval sub ranges
; properly.

; CHECK-LABEL: foobar:
; Output with subregister liveness disabled:
; CHECK: v_mov_b32_e32 v1, 1
; CHECK: v_mov_b32_e32 v0, 0
; CHECK: v_cmp_eq_i32_e32 vcc, s0, v0
; CHECK: v_cmp_eq_i32_e64 s[0:1], s0, v1
; CHECK: v_mov_b32_e32 v1, 3
; CHECK: v_mov_b32_e32 v0, 2
; CHECK: v_cmp_eq_i32_e64 s[2:3], s0, v0
; CHECK: v_cmp_eq_i32_e64 s[4:5], s0, v1
; CHECK: v_cndmask_b32_e64 v3, 0, -1, s[4:5]
; CHECK: v_cndmask_b32_e64 v2, 0, -1, s[2:3]
; CHECK: v_cndmask_b32_e64 v1, 0, -1, s[0:1]
; CHECK: v_cndmask_b32_e64 v0, 0, -1, vcc
; CHECK: v_cmp_ne_i32_e32 vcc, 0, v1
; CHECK: v_cndmask_b32_e64 v1, 0, v0, vcc
; CHECK: s_mov_b32 s3, 0xf000
; CHECK: s_mov_b32 s2, -1
; CHECK: buffer_store_dwordx2 v[0:1], s[0:3], 0
; CHECK: s_endpgm
; Output with subregister liveness enabled:
; XCHECK: v_mov_b32_e32 v1, 1
; XCHECK: v_mov_b32_e32 v0, 0
; XCHECK: v_cmp_eq_i32_e32 vcc, s0, v1
; XCHECK: v_mov_b32_e32 v1, 3
; XCHECK: v_mov_b32_e32 v0, 2
; XCHECK: v_cmp_eq_i32_e64 s[0:1], s0, v0
; XCHECK: v_cmp_eq_i32_e64 s[2:3], s0, v1
; XCHECK: v_cndmask_b32_e64 v3, 0, -1, s[2:3]
; XCHECK: v_cndmask_b32_e64 v2, 0, -1, s[0:1]
; XCHECK: v_cndmask_b32_e64 v1, 0, -1, vcc
; XCHECK: v_cmp_ne_i32_e32 vcc, 0, v1
; XCHECK: v_cndmask_b32_e64 v1, 0, v0, vcc
; XCHECK: s_mov_b32 s3, 0xf000
; XCHECK: s_mov_b32 s2, -1
; XCHECK: buffer_store_dwordx2 v[0:1], s[0:3], 0
; XCHECK: s_endpgm
define void @foobar() {
  %v0 = icmp eq <4 x i32> undef, <i32 0, i32 1, i32 2, i32 3>
  %v3 = sext <4 x i1> %v0 to <4 x i32>
  %v4 = extractelement <4 x i32> %v3, i32 1
  %v5 = icmp ne i32 %v4, 0
  %v6 = select i1 %v5, i32 undef, i32 0
  %v15 = insertelement <2 x i32> undef, i32 %v6, i32 1
  store <2 x i32> %v15, <2 x i32> addrspace(1)* undef, align 8
  ret void
}

declare double @llvm.fma.f64(double, double, double)
