; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

; SI-LABEL: {{^}}sext_bool_icmp_ne:
; SI: V_CMP_NE_I32
; SI-NEXT: V_CNDMASK_B32
; SI-NOT: V_CMP_NE_I32
; SI-NOT: V_CNDMASK_B32
; SI: S_ENDPGM
define void @sext_bool_icmp_ne(i1 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %icmp0 = icmp ne i32 %a, %b
  %ext = sext i1 %icmp0 to i32
  %icmp1 = icmp ne i32 %ext, 0
  store i1 %icmp1, i1 addrspace(1)* %out
  ret void
}
