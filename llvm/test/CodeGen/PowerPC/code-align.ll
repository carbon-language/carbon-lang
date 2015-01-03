; RUN: llc -mcpu=ppc64 < %s | FileCheck %s -check-prefix=GENERIC
; RUN: llc -mcpu=970 < %s | FileCheck %s -check-prefix=BASIC
; RUN: llc -mcpu=a2 < %s | FileCheck %s -check-prefix=BASIC
; RUN: llc -mcpu=e500mc < %s | FileCheck %s -check-prefix=BASIC
; RUN: llc -mcpu=e5500 < %s | FileCheck %s -check-prefix=BASIC
; RUN: llc -mcpu=pwr4 < %s | FileCheck %s -check-prefix=BASIC
; RUN: llc -mcpu=pwr5 < %s | FileCheck %s -check-prefix=BASIC
; RUN: llc -mcpu=pwr5x < %s | FileCheck %s -check-prefix=BASIC
; RUN: llc -mcpu=pwr6 < %s | FileCheck %s -check-prefix=BASIC
; RUN: llc -mcpu=pwr6x < %s | FileCheck %s -check-prefix=BASIC
; RUN: llc -mcpu=pwr7 < %s | FileCheck %s -check-prefix=BASIC
; RUN: llc -mcpu=pwr8 < %s | FileCheck %s -check-prefix=BASIC
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind readnone
define signext i32 @foo(i32 signext %x) #0 {
entry:
  %mul = shl nsw i32 %x, 1
  ret i32 %mul

; GENERIC-LABEL: .globl  foo
; BASIC-LABEL: .globl  foo
; GENERIC: .align  2
; BASIC: .align  4
; GENERIC: @foo
; BASIC: @foo
}

; Function Attrs: nounwind
define void @loop(i32 signext %x, i32* nocapture %a) #1 {
entry:
  br label %vector.body

; GENERIC-LABEL: @loop
; BASIC-LABEL: @loop
; GENERIC: mtctr
; BASIC: mtctr
; GENERIC-NOT: .align
; BASIC: .align  4
; GENERIC: bdnz
; BASIC: bdnz

vector.body:                                      ; preds = %vector.body, %entry
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %induction45 = or i64 %index, 1
  %0 = getelementptr inbounds i32* %a, i64 %index
  %1 = getelementptr inbounds i32* %a, i64 %induction45
  %2 = load i32* %0, align 4
  %3 = load i32* %1, align 4
  %4 = add nsw i32 %2, 4
  %5 = add nsw i32 %3, 4
  store i32 %4, i32* %0, align 4
  store i32 %5, i32* %1, align 4
  %index.next = add i64 %index, 2
  %6 = icmp eq i64 %index.next, 2048
  br i1 %6, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }

