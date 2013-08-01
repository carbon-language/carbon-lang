; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; CHECK: @fsub_f32
; CHECK: ADD * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], -T[0-9]+\.[XYZW]}}

define void @fsub_f32() {
   %r0 = call float @llvm.R600.load.input(i32 0)
   %r1 = call float @llvm.R600.load.input(i32 1)
   %r2 = fsub float %r0, %r1
   call void @llvm.AMDGPU.store.output(float %r2, i32 0)
   ret void
}

declare float @llvm.R600.load.input(i32) readnone

declare void @llvm.AMDGPU.store.output(float, i32)

; CHECK: @fsub_v2f32
; CHECK-DAG: ADD * T{{[0-9]+\.[XYZW]}}, KC0[3].X, -KC0[3].Z
; CHECK-DAG: ADD * T{{[0-9]+\.[XYZW]}}, KC0[2].W, -KC0[3].Y
define void @fsub_v2f32(<2 x float> addrspace(1)* %out, <2 x float> %a, <2 x float> %b) {
entry:
  %0 = fsub <2 x float> %a, %b
  store <2 x float> %0, <2 x float> addrspace(1)* %out
  ret void
}

; CHECK: @fsub_v4f32
; CHECK: ADD T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], -T[0-9]+\.[XYZW]}}
; CHECK: ADD * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], -T[0-9]+\.[XYZW]}}
; CHECK: ADD * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], -T[0-9]+\.[XYZW]}}
; CHECK: ADD * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], -T[0-9]+\.[XYZW]}}
define void @fsub_v4f32(<4 x float> addrspace(1)* %out, <4 x float> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x float> addrspace(1)* %in, i32 1
  %a = load <4 x float> addrspace(1) * %in
  %b = load <4 x float> addrspace(1) * %b_ptr
  %result = fsub <4 x float> %a, %b
  store <4 x float> %result, <4 x float> addrspace(1)* %out
  ret void
}
