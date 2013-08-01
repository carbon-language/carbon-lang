; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck --check-prefix=EG-CHECK %s
; RUN: llc < %s -march=r600 -mcpu=cayman | FileCheck --check-prefix=CM-CHECK %s
; RUN: llc < %s -march=r600 -mcpu=verde | FileCheck --check-prefix=SI-CHECK %s

; floating-point store
; EG-CHECK: @store_f32
; EG-CHECK: RAT_WRITE_CACHELESS_32_eg T{{[0-9]+\.X, T[0-9]+\.X}}, 1
; CM-CHECK: @store_f32
; CM-CHECK: EXPORT_RAT_INST_STORE_DWORD T{{[0-9]+\.X, T[0-9]+\.X}}
; SI-CHECK: @store_f32
; SI-CHECK: BUFFER_STORE_DWORD

define void @store_f32(float addrspace(1)* %out, float %in) {
  store float %in, float addrspace(1)* %out
  ret void
}

; vec2 floating-point stores
; EG-CHECK: @store_v2f32
; EG-CHECK: RAT_WRITE_CACHELESS_64_eg
; CM-CHECK: @store_v2f32
; CM-CHECK: EXPORT_RAT_INST_STORE_DWORD
; SI-CHECK: @store_v2f32
; SI-CHECK: BUFFER_STORE_DWORDX2

define void @store_v2f32(<2 x float> addrspace(1)* %out, float %a, float %b) {
entry:
  %0 = insertelement <2 x float> <float 0.0, float 0.0>, float %a, i32 0
  %1 = insertelement <2 x float> %0, float %b, i32 0
  store <2 x float> %1, <2 x float> addrspace(1)* %out
  ret void
}

; The stores in this function are combined by the optimizer to create a
; 64-bit store with 32-bit alignment.  This is legal for SI and the legalizer
; should not try to split the 64-bit store back into 2 32-bit stores.
;
; Evergreen / Northern Islands don't support 64-bit stores yet, so there should
; be two 32-bit stores.

; EG-CHECK: @vecload2
; EG-CHECK: RAT_WRITE_CACHELESS_64_eg
; CM-CHECK: @vecload2
; CM-CHECK: EXPORT_RAT_INST_STORE_DWORD
; SI-CHECK: @vecload2
; SI-CHECK: BUFFER_STORE_DWORDX2
define void @vecload2(i32 addrspace(1)* nocapture %out, i32 addrspace(2)* nocapture %mem) #0 {
entry:
  %0 = load i32 addrspace(2)* %mem, align 4, !tbaa !5
  %arrayidx1.i = getelementptr inbounds i32 addrspace(2)* %mem, i64 1
  %1 = load i32 addrspace(2)* %arrayidx1.i, align 4, !tbaa !5
  store i32 %0, i32 addrspace(1)* %out, align 4, !tbaa !5
  %arrayidx1 = getelementptr inbounds i32 addrspace(1)* %out, i64 1
  store i32 %1, i32 addrspace(1)* %arrayidx1, align 4, !tbaa !5
  ret void
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }

!5 = metadata !{metadata !"int", metadata !6}
!6 = metadata !{metadata !"omnipotent char", metadata !7}
!7 = metadata !{metadata !"Simple C/C++ TBAA"}
