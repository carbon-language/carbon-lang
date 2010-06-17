; RUN: llc < %s -mcpu=cortex-a8 | FileCheck %s -check-prefix=CHECK-A8
; RUN: llc < %s -mcpu=cortex-m3 | FileCheck %s -check-prefix=CHECK-M3

target triple = "thumbv7-apple-darwin10"

define i32 @f1(i16* %ptr) nounwind {
; CHECK-A8: f1
; CHECK-A8: sxth
; CHECK-M3: f1
; CHECK-M3-NOT: sxth
; CHECK-M3: bx lr
  %1 = load i16* %ptr
  %2 = icmp eq i16 %1, 1
  %3 = sext i16 %1 to i32
  br i1 %2, label %.next, label %.exit

.next:
  br label %.exit

.exit:
  ret i32 %3
}
