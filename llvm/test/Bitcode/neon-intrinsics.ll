; RUN: llvm-dis < %s.bc | FileCheck %s

; vmovls should be auto-upgraded to sext

; CHECK: vmovls8
; CHECK-NOT: arm.neon.vmovls.v8i16
; CHECK: sext <8 x i8>

; CHECK: vmovls16
; CHECK-NOT: arm.neon.vmovls.v4i32
; CHECK: sext <4 x i16>

; CHECK: vmovls32
; CHECK-NOT: arm.neon.vmovls.v2i64
; CHECK: sext <2 x i32>

; vmovlu should be auto-upgraded to zext

; CHECK: vmovlu8
; CHECK-NOT: arm.neon.vmovlu.v8i16
; CHECK: zext <8 x i8>

; CHECK: vmovlu16
; CHECK-NOT: arm.neon.vmovlu.v4i32
; CHECK: zext <4 x i16>

; CHECK: vmovlu32
; CHECK-NOT: arm.neon.vmovlu.v2i64
; CHECK: zext <2 x i32>

; vld* and vst* intrinsic calls need an alignment argument (defaulted to 1)

; CHECK: vld1i8
; CHECK: i32 1
; CHECK: vld2Qi16
; CHECK: i32 1
; CHECK: vld3i32
; CHECK: i32 1
; CHECK: vld4Qf
; CHECK: i32 1

; CHECK: vst1i8
; CHECK: i32 1
; CHECK: vst2Qi16
; CHECK: i32 1
; CHECK: vst3i32
; CHECK: i32 1
; CHECK: vst4Qf
; CHECK: i32 1

; CHECK: vld2laneQi16
; CHECK: i32 1
; CHECK: vld3lanei32
; CHECK: i32 1
; CHECK: vld4laneQf
; CHECK: i32 1

; CHECK: vst2laneQi16
; CHECK: i32 1
; CHECK: vst3lanei32
; CHECK: i32 1
; CHECK: vst4laneQf
; CHECK: i32 1
