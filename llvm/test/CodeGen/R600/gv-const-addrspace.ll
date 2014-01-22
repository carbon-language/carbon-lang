; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s --check-prefix=R600 --check-prefix=FUNC

; XXX: Test on SI once 64-bit adds are supportes.

@float_gv = internal addrspace(2) unnamed_addr constant [5 x float] [float 0.0, float 1.0, float 2.0, float 3.0, float 4.0], align 4

; FUNC-LABEL: @float

; R600-DAG: MOV {{\** *}}T2.X
; R600-DAG: MOV {{\** *}}T3.X
; R600-DAG: MOV {{\** *}}T4.X
; R600-DAG: MOV {{\** *}}T5.X
; R600-DAG: MOV {{\** *}}T6.X
; R600: MOVA_INT

define void @float(float addrspace(1)* %out, i32 %index) {
entry:
  %0 = getelementptr inbounds [5 x float] addrspace(2)* @float_gv, i32 0, i32 %index
  %1 = load float addrspace(2)* %0
  store float %1, float addrspace(1)* %out
  ret void
}

@i32_gv = internal addrspace(2) unnamed_addr constant [5 x i32] [i32 0, i32 1, i32 2, i32 3, i32 4], align 4

; FUNC-LABEL: @i32

; R600-DAG: MOV {{\** *}}T2.X
; R600-DAG: MOV {{\** *}}T3.X
; R600-DAG: MOV {{\** *}}T4.X
; R600-DAG: MOV {{\** *}}T5.X
; R600-DAG: MOV {{\** *}}T6.X
; R600: MOVA_INT

define void @i32(i32 addrspace(1)* %out, i32 %index) {
entry:
  %0 = getelementptr inbounds [5 x i32] addrspace(2)* @i32_gv, i32 0, i32 %index
  %1 = load i32 addrspace(2)* %0
  store i32 %1, i32 addrspace(1)* %out
  ret void
}
