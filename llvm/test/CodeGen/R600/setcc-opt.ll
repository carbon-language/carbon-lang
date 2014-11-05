; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

; SI-LABEL: {{^}}sext_bool_icmp_ne:
; SI: v_cmp_ne_i32
; SI-NEXT: v_cndmask_b32
; SI-NOT: v_cmp_ne_i32
; SI-NOT: v_cndmask_b32
; SI: s_endpgm
define void @sext_bool_icmp_ne(i1 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %icmp0 = icmp ne i32 %a, %b
  %ext = sext i1 %icmp0 to i32
  %icmp1 = icmp ne i32 %ext, 0
  store i1 %icmp1, i1 addrspace(1)* %out
  ret void
}
