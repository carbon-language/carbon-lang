; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s --check-prefix=R600-CHECK
; RUN: llc < %s -march=r600 -mcpu=SI | FileCheck %s --check-prefix=SI-CHECK

; R600-CHECK: @v4i32_kernel_arg
; R600-CHECK: VTX_READ_128 T{{[0-9]+}}.XYZW, T{{[0-9]+}}.X, 40
; SI-CHECK: @v4i32_kernel_arg
; SI-CHECK: BUFFER_STORE_DWORDX4
define void @v4i32_kernel_arg(<4 x i32> addrspace(1)* %out, <4 x i32>  %in) {
entry:
  store <4 x i32> %in, <4 x i32> addrspace(1)* %out
  ret void
}

; R600-CHECK: @v4f32_kernel_arg
; R600-CHECK: VTX_READ_128 T{{[0-9]+}}.XYZW, T{{[0-9]+}}.X, 40
; SI-CHECK: @v4f32_kernel_arg
; SI-CHECK: BUFFER_STORE_DWORDX4
define void @v4f32_kernel_args(<4 x float> addrspace(1)* %out, <4 x float>  %in) {
entry:
  store <4 x float> %in, <4 x float> addrspace(1)* %out
  ret void
}
