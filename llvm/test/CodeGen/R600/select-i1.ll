; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

; FIXME: This should go in existing select.ll test, except the current testcase there is broken on SI

; FUNC-LABEL: @select_i1
; SI: V_CNDMASK_B32
; SI-NOT: V_CNDMASK_B32
define void @select_i1(i1 addrspace(1)* %out, i32 %cond, i1 %a, i1 %b) nounwind {
  %cmp = icmp ugt i32 %cond, 5
  %sel = select i1 %cmp, i1 %a, i1 %b
  store i1 %sel, i1 addrspace(1)* %out, align 4
  ret void
}

