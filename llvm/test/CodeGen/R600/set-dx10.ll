; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; These tests check that floating point comparisons which are used by select
; to store integer true (-1) and false (0) values are lowered to one of the
; SET*DX10 instructions.

; CHECK: @fcmp_une_select_fptosi
; CHECK: SETNE_DX10 * T{{[0-9]+\.[XYZW]}}, T{{[0-9]+\.[XYZW]}}, literal.x,
; CHECK-NEXT: 1084227584(5.000000e+00)
define void @fcmp_une_select_fptosi(i32 addrspace(1)* %out, float %in) {
entry:
  %0 = fcmp une float %in, 5.0
  %1 = select i1 %0, float 1.000000e+00, float 0.000000e+00
  %2 = fsub float -0.000000e+00, %1
  %3 = fptosi float %2 to i32
  store i32 %3, i32 addrspace(1)* %out
  ret void
}

; CHECK: @fcmp_une_select_i32
; CHECK: SETNE_DX10 * T{{[0-9]+\.[XYZW]}}, T{{[0-9]+\.[XYZW]}}, literal.x,
; CHECK-NEXT: 1084227584(5.000000e+00)
define void @fcmp_une_select_i32(i32 addrspace(1)* %out, float %in) {
entry:
  %0 = fcmp une float %in, 5.0
  %1 = select i1 %0, i32 -1, i32 0
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; CHECK: @fcmp_ueq_select_fptosi
; CHECK: SETE_DX10 * T{{[0-9]+\.[XYZW]}}, T{{[0-9]+\.[XYZW]}}, literal.x,
; CHECK-NEXT: 1084227584(5.000000e+00)
define void @fcmp_ueq_select_fptosi(i32 addrspace(1)* %out, float %in) {
entry:
  %0 = fcmp ueq float %in, 5.0
  %1 = select i1 %0, float 1.000000e+00, float 0.000000e+00
  %2 = fsub float -0.000000e+00, %1
  %3 = fptosi float %2 to i32
  store i32 %3, i32 addrspace(1)* %out
  ret void
}

; CHECK: @fcmp_ueq_select_i32
; CHECK: SETE_DX10 * T{{[0-9]+\.[XYZW]}}, T{{[0-9]+\.[XYZW]}}, literal.x,
; CHECK-NEXT: 1084227584(5.000000e+00)
define void @fcmp_ueq_select_i32(i32 addrspace(1)* %out, float %in) {
entry:
  %0 = fcmp ueq float %in, 5.0
  %1 = select i1 %0, i32 -1, i32 0
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; CHECK: @fcmp_ugt_select_fptosi
; CHECK: SETGT_DX10 * T{{[0-9]+\.[XYZW]}}, T{{[0-9]+\.[XYZW]}}, literal.x,
; CHECK-NEXT: 1084227584(5.000000e+00)
define void @fcmp_ugt_select_fptosi(i32 addrspace(1)* %out, float %in) {
entry:
  %0 = fcmp ugt float %in, 5.0
  %1 = select i1 %0, float 1.000000e+00, float 0.000000e+00
  %2 = fsub float -0.000000e+00, %1
  %3 = fptosi float %2 to i32
  store i32 %3, i32 addrspace(1)* %out
  ret void
}

; CHECK: @fcmp_ugt_select_i32
; CHECK: SETGT_DX10 * T{{[0-9]+\.[XYZW]}}, T{{[0-9]+\.[XYZW]}}, literal.x,
; CHECK-NEXT: 1084227584(5.000000e+00)
define void @fcmp_ugt_select_i32(i32 addrspace(1)* %out, float %in) {
entry:
  %0 = fcmp ugt float %in, 5.0
  %1 = select i1 %0, i32 -1, i32 0
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; CHECK: @fcmp_uge_select_fptosi
; CHECK: SETGE_DX10 * T{{[0-9]+\.[XYZW]}}, T{{[0-9]+\.[XYZW]}}, literal.x,
; CHECK-NEXT: 1084227584(5.000000e+00)
define void @fcmp_uge_select_fptosi(i32 addrspace(1)* %out, float %in) {
entry:
  %0 = fcmp uge float %in, 5.0
  %1 = select i1 %0, float 1.000000e+00, float 0.000000e+00
  %2 = fsub float -0.000000e+00, %1
  %3 = fptosi float %2 to i32
  store i32 %3, i32 addrspace(1)* %out
  ret void
}

; CHECK: @fcmp_uge_select_i32
; CHECK: SETGE_DX10 * T{{[0-9]+\.[XYZW]}}, T{{[0-9]+\.[XYZW]}}, literal.x,
; CHECK-NEXT: 1084227584(5.000000e+00)
define void @fcmp_uge_select_i32(i32 addrspace(1)* %out, float %in) {
entry:
  %0 = fcmp uge float %in, 5.0
  %1 = select i1 %0, i32 -1, i32 0
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; CHECK: @fcmp_ule_select_fptosi
; CHECK: SETGE_DX10 * T{{[0-9]+\.[XYZW]}}, literal.x, T{{[0-9]+\.[XYZW]}},
; CHECK-NEXT: 1084227584(5.000000e+00)
define void @fcmp_ule_select_fptosi(i32 addrspace(1)* %out, float %in) {
entry:
  %0 = fcmp ule float %in, 5.0
  %1 = select i1 %0, float 1.000000e+00, float 0.000000e+00
  %2 = fsub float -0.000000e+00, %1
  %3 = fptosi float %2 to i32
  store i32 %3, i32 addrspace(1)* %out
  ret void
}

; CHECK: @fcmp_ule_select_i32
; CHECK: SETGE_DX10 * T{{[0-9]+\.[XYZW]}}, literal.x, T{{[0-9]+\.[XYZW]}},
; CHECK-NEXT: 1084227584(5.000000e+00)
define void @fcmp_ule_select_i32(i32 addrspace(1)* %out, float %in) {
entry:
  %0 = fcmp ule float %in, 5.0
  %1 = select i1 %0, i32 -1, i32 0
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; CHECK: @fcmp_ult_select_fptosi
; CHECK: SETGT_DX10 * T{{[0-9]+\.[XYZW]}}, literal.x, T{{[0-9]+\.[XYZW]}},
; CHECK-NEXT: 1084227584(5.000000e+00)
define void @fcmp_ult_select_fptosi(i32 addrspace(1)* %out, float %in) {
entry:
  %0 = fcmp ult float %in, 5.0
  %1 = select i1 %0, float 1.000000e+00, float 0.000000e+00
  %2 = fsub float -0.000000e+00, %1
  %3 = fptosi float %2 to i32
  store i32 %3, i32 addrspace(1)* %out
  ret void
}

; CHECK: @fcmp_ult_select_i32
; CHECK: SETGT_DX10 * T{{[0-9]+\.[XYZW]}}, literal.x, T{{[0-9]+\.[XYZW]}},
; CHECK-NEXT: 1084227584(5.000000e+00)
define void @fcmp_ult_select_i32(i32 addrspace(1)* %out, float %in) {
entry:
  %0 = fcmp ult float %in, 5.0
  %1 = select i1 %0, i32 -1, i32 0
  store i32 %1, i32 addrspace(1)* %out
  ret void
}
