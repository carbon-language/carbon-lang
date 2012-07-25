; RUN: llc -march=mipsel -mcpu=mips32r2 < %s | FileCheck %s

%struct.S1 = type { [65536 x i8] }

@s1 = external global %struct.S1

define void @f() nounwind {
entry:
; CHECK:  lui $at, 65534
; CHECK:  addiu $at, $at, -16
; CHECK:  addu  $sp, $sp, $at

  %agg.tmp = alloca %struct.S1, align 1
  %tmp = getelementptr inbounds %struct.S1* %agg.tmp, i32 0, i32 0, i32 0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %tmp, i8* getelementptr inbounds (%struct.S1* @s1, i32 0, i32 0, i32 0), i32 65536, i32 1, i1 false)
  call void @f2(%struct.S1* byval %agg.tmp) nounwind
  ret void
}

declare void @f2(%struct.S1* byval)

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
