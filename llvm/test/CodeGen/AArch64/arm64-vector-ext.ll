; RUN: llc < %s -march=arm64 -aarch64-neon-syntax=apple | FileCheck %s

;CHECK: @func30
;CHECK: ushll.4s  v0, v0, #0
;CHECK: movi.4s v1, #0x1
;CHECK: and.16b v0, v0, v1
;CHECK: str  q0, [x0]
;CHECK: ret

%T0_30 = type <4 x i1>
%T1_30 = type <4 x i32>
define void @func30(%T0_30 %v0, %T1_30* %p1) {
  %r = zext %T0_30 %v0 to %T1_30
  store %T1_30 %r, %T1_30* %p1
  ret void
}
