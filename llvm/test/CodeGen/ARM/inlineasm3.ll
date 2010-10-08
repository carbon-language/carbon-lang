; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

; Radar 7449043
%struct.int32x4_t = type { <4 x i32> }

define void @t() nounwind {
entry:
; CHECK: vmov.I64 q15, #0
; CHECK: vmov.32 d30[0], r0
; CHECK: vmov q8, q15
  %tmp = alloca %struct.int32x4_t, align 16
  call void asm sideeffect "vmov.I64 q15, #0\0Avmov.32 d30[0], $1\0Avmov ${0:q}, q15\0A", "=*w,r,~{d31},~{d30}"(%struct.int32x4_t* %tmp, i32 8192) nounwind
  ret void
}

; Radar 7457110
%struct.int32x2_t = type { <4 x i32> }

define void @t2() nounwind {
entry:
; CHECK: vmov d30, d16
; CHECK: vmov.32 r0, d30[0]
  %asmtmp2 = tail call i32 asm sideeffect "vmov d30, $1\0Avmov.32 $0, d30[0]\0A", "=r,w,~{d30}"(<2 x i32> undef) nounwind
  ret void
}
