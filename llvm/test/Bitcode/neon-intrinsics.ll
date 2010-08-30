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

; vaddl/vaddw should be auto-upgraded to add with sext/zext

; CHECK: vaddls16
; CHECK-NOT: arm.neon.vaddls.v4i32
; CHECK: sext <4 x i16>
; CHECK-NEXT: sext <4 x i16>
; CHECK-NEXT: add <4 x i32>

; CHECK: vaddlu32
; CHECK-NOT: arm.neon.vaddlu.v2i64
; CHECK: zext <2 x i32>
; CHECK-NEXT: zext <2 x i32>
; CHECK-NEXT: add <2 x i64>

; CHECK: vaddws8
; CHECK-NOT: arm.neon.vaddws.v8i16
; CHECK: sext <8 x i8>
; CHECK-NEXT: add <8 x i16>

; CHECK: vaddwu16
; CHECK-NOT: arm.neon.vaddwu.v4i32
; CHECK: zext <4 x i16>
; CHECK-NEXT: add <4 x i32>

; vsubl/vsubw should be auto-upgraded to sub with sext/zext

; CHECK: vsubls16
; CHECK-NOT: arm.neon.vsubls.v4i32
; CHECK: sext <4 x i16>
; CHECK-NEXT: sext <4 x i16>
; CHECK-NEXT: sub <4 x i32>

; CHECK: vsublu32
; CHECK-NOT: arm.neon.vsublu.v2i64
; CHECK: zext <2 x i32>
; CHECK-NEXT: zext <2 x i32>
; CHECK-NEXT: sub <2 x i64>

; CHECK: vsubws8
; CHECK-NOT: arm.neon.vsubws.v8i16
; CHECK: sext <8 x i8>
; CHECK-NEXT: sub <8 x i16>

; CHECK: vsubwu16
; CHECK-NOT: arm.neon.vsubwu.v4i32
; CHECK: zext <4 x i16>
; CHECK-NEXT: sub <4 x i32>

; vmovn should be auto-upgraded to trunc

; CHECK: vmovni16
; CHECK-NOT: arm.neon.vmovn.v8i8
; CHECK: trunc <8 x i16>

; CHECK: vmovni32
; CHECK-NOT: arm.neon.vmovn.v4i16
; CHECK: trunc <4 x i32>

; CHECK: vmovni64
; CHECK-NOT: arm.neon.vmovn.v2i32
; CHECK: trunc <2 x i64>

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
