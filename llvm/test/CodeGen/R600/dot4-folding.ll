;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; Exactly one constant vector can be folded into dot4, which means exactly
; 4 MOV instructions
; CHECK: {{^}}main:
; CHECK: MOV
; CHECK: MOV
; CHECK: MOV
; CHECK: MOV
; CHECK-NOT: MOV
; CHECK-NOT: MOV
; CHECK-NOT: MOV
; CHECK-NOT: MOV

define void @main(float addrspace(1)* %out) {
main_body:
  %0 = load <4 x float>, <4 x float> addrspace(8)* null
  %1 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>] addrspace(8)* null, i64 0, i32 1)
  %2 = call float @llvm.AMDGPU.dp4(<4 x float> %0,<4 x float> %1)
  %3 = insertelement <4 x float> undef, float %2, i32 0
  call void @llvm.R600.store.swizzle(<4 x float> %3, i32 0, i32 0)
  ret void
}

declare float @llvm.AMDGPU.dp4(<4 x float>, <4 x float>) #1
declare void @llvm.R600.store.swizzle(<4 x float>, i32, i32)
attributes #1 = { readnone }
