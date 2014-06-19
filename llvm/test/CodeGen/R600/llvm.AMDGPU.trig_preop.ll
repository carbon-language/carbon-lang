; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

declare double @llvm.AMDGPU.trig.preop.f64(double, i32) nounwind readnone

; SI-LABEL: @test_trig_preop_f64:
; SI-DAG: BUFFER_LOAD_DWORD [[SEG:v[0-9]+]]
; SI-DAG: BUFFER_LOAD_DWORDX2 [[SRC:v\[[0-9]+:[0-9]+\]]],
; SI: V_TRIG_PREOP_F64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[SRC]], [[SEG]]
; SI: BUFFER_STORE_DWORDX2 [[RESULT]],
; SI: S_ENDPGM
define void @test_trig_preop_f64(double addrspace(1)* %out, double addrspace(1)* %aptr, i32 addrspace(1)* %bptr) nounwind {
  %a = load double addrspace(1)* %aptr, align 8
  %b = load i32 addrspace(1)* %bptr, align 4
  %result = call double @llvm.AMDGPU.trig.preop.f64(double %a, i32 %b) nounwind readnone
  store double %result, double addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: @test_trig_preop_f64_imm_segment:
; SI: BUFFER_LOAD_DWORDX2 [[SRC:v\[[0-9]+:[0-9]+\]]],
; SI: V_TRIG_PREOP_F64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[SRC]], 7
; SI: BUFFER_STORE_DWORDX2 [[RESULT]],
; SI: S_ENDPGM
define void @test_trig_preop_f64_imm_segment(double addrspace(1)* %out, double addrspace(1)* %aptr) nounwind {
  %a = load double addrspace(1)* %aptr, align 8
  %result = call double @llvm.AMDGPU.trig.preop.f64(double %a, i32 7) nounwind readnone
  store double %result, double addrspace(1)* %out, align 8
  ret void
}
