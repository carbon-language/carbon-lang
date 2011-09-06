; DISABLED: llc -march=mipsel -mcpu=4ke < %s | FileCheck %s
; RUN: false

; byval is currently unsupported.
; XFAIL: *

%struct.S1 = type { [65536 x i8] }

@s1 = external global %struct.S1

define void @f() nounwind {
entry:
; CHECK:  lui $at, 65534
; CHECK:  addu  $at, $sp, $at
; CHECK:  addiu $sp, $at, -24
; CHECK:  .cprestore  65536

  %agg.tmp = alloca %struct.S1, align 1
  %tmp = getelementptr inbounds %struct.S1* %agg.tmp, i32 0, i32 0, i32 0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %tmp, i8* getelementptr inbounds (%struct.S1* @s1, i32 0, i32 0, i32 0), i32 65536, i32 1, i1 false)
  call void @f2(%struct.S1* byval %agg.tmp) nounwind
  ret void
}

declare void @f2(%struct.S1* byval)

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
