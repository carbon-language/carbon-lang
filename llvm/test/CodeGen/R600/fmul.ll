; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; CHECK: @fmul_f32
; CHECK: MUL_IEEE T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

define void @fmul_f32() {
   %r0 = call float @llvm.R600.load.input(i32 0)
   %r1 = call float @llvm.R600.load.input(i32 1)
   %r2 = fmul float %r0, %r1
   call void @llvm.AMDGPU.store.output(float %r2, i32 0)
   ret void
}

declare float @llvm.R600.load.input(i32) readnone

declare void @llvm.AMDGPU.store.output(float, i32)

; CHECK: @fmul_v4f32
; CHECK: MUL_IEEE T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; CHECK: MUL_IEEE T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; CHECK: MUL_IEEE T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; CHECK: MUL_IEEE T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

define void @fmul_v4f32(<4 x float> addrspace(1)* %out, <4 x float> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x float> addrspace(1)* %in, i32 1
  %a = load <4 x float> addrspace(1) * %in
  %b = load <4 x float> addrspace(1) * %b_ptr
  %result = fmul <4 x float> %a, %b
  store <4 x float> %result, <4 x float> addrspace(1)* %out
  ret void
}
