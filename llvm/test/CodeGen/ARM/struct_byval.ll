; RUN: llc < %s -mtriple=armv7-apple-ios6.0 | FileCheck %s

; rdar://9877866
%struct.SmallStruct = type { i32, [8 x i32], [37 x i8] }
%struct.LargeStruct = type { i32, [1001 x i8], [300 x i32] }

define i32 @f() nounwind ssp {
entry:
; CHECK: f:
; CHECK: ldr
; CHECK: str
; CHECK-NOT:bne
  %st = alloca %struct.SmallStruct, align 4
  %call = call i32 @e1(%struct.SmallStruct* byval %st)
  ret i32 0
}

; Generate a loop for large struct byval
define i32 @g() nounwind ssp {
entry:
; CHECK: g:
; CHECK: ldr
; CHECK: sub
; CHECK: str
; CHECK: bne
  %st = alloca %struct.LargeStruct, align 4
  %call = call i32 @e2(%struct.LargeStruct* byval %st)
  ret i32 0
}

declare i32 @e1(%struct.SmallStruct* nocapture byval %in) nounwind
declare i32 @e2(%struct.LargeStruct* nocapture byval %in) nounwind
