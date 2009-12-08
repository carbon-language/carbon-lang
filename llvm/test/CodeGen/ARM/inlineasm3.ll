; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

%struct.int32x4_t = type { <4 x i32> }

define arm_apcscc void @t() nounwind {
entry:
; CHECK: vmov.I64 q15, #0
; CHECK: vmov.32 d30[0], r0
; CHECK: vmov q0, q15
  %tmp = alloca %struct.int32x4_t, align 16
  call void asm sideeffect "vmov.I64 q15, #0\0Avmov.32 d30[0], $1\0Avmov ${0:q}, q15\0A", "=*w,r,~{d31},~{d30}"(%struct.int32x4_t* %tmp, i32 8192) nounwind
  ret void
}
