; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs< %s | FileCheck -check-prefix=SI %s
;
;
; Most SALU instructions ignore control flow, so we need to make sure
; they don't overwrite values from other blocks.

; If the branch decision is made based on a value in an SGPR then all
; threads will execute the same code paths, so we don't need to worry
; about instructions in different blocks overwriting each other.
; SI-LABEL: {{^}}sgpr_if_else_salu_br:
; SI: S_ADD
; SI: S_ADD

define void @sgpr_if_else_salu_br(i32 addrspace(1)* %out, i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) {
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

; The two S_ADD instructions should write to different registers, since
; different threads will take different control flow paths.

; SI-LABEL: {{^}}sgpr_if_else_valu_br:
; SI: S_ADD_I32 [[SGPR:s[0-9]+]]
; SI-NOT: S_ADD_I32 [[SGPR]]

define void @sgpr_if_else_valu_br(i32 addrspace(1)* %out, float %a, i32 %b, i32 %c, i32 %d, i32 %e) {
entry:
  %tid = call i32 @llvm.r600.read.tidig.x() #0
  %tid_f = uitofp i32 %tid to float
  %tmp1 = fcmp ueq float %tid_f, 0.0
  br i1 %tmp1, label %if, label %else

if:
  %tmp2 = add i32 %b, %c
  br label %endif

else:
  %tmp3 = add i32 %d, %e
  br label %endif

endif:
  %tmp4 = phi i32 [%tmp2, %if], [%tmp3, %else]
  store i32 %tmp4, i32 addrspace(1)* %out
  ret void
}

declare i32 @llvm.r600.read.tidig.x() #0

attributes #0 = { readnone }
