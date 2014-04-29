; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs< %s | FileCheck -check-prefix=SI %s
;
;
; Most SALU instructions ignore control flow, so we need to make sure
; they don't overwrite values from other blocks.

; SI-NOT: S_ADD

define void @sgpr_if_else(i32 addrspace(1)* %out, i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) {
entry:
  %0 = icmp eq i32 %a, 0
  br i1 %0, label %if, label %else

if:
  %1 = add i32 %b, %c
  br label %endif

else:
  %2 = add i32 %d, %e
  br label %endif

endif:
  %3 = phi i32 [%1, %if], [%2, %else]
  %4 = add i32 %3, %a
  store i32 %4, i32 addrspace(1)* %out
  ret void
}
