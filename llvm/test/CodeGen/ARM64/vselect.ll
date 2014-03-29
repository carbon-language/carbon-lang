; RUN: llc < %s -march=arm64 -arm64-neon-syntax=apple | FileCheck %s

;CHECK: @func63
;CHECK: cmeq.4h v0, v0, v1
;CHECK: sshll.4s  v0, v0, #0
;CHECK: bsl.16b v0, v2, v3
;CHECK: str  q0, [x0]
;CHECK: ret

%T0_63 = type <4 x i16>
%T1_63 = type <4 x i32>
%T2_63 = type <4 x i1>
define void @func63(%T1_63* %out, %T0_63 %v0, %T0_63 %v1, %T1_63 %v2, %T1_63 %v3) {
  %cond = icmp eq %T0_63 %v0, %v1
  %r = select %T2_63 %cond, %T1_63 %v2, %T1_63 %v3
  store %T1_63 %r, %T1_63* %out
  ret void
}
