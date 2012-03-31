; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s

define zeroext i1 @f1(i8* %x) {
entry:
  %0 = load i8* %x, align 1, !range !0
  %tobool = trunc i8 %0 to i1
  ret i1 %tobool
}

; CHECK: f1:
; CHECK: movb	(%rdi), %al
; CHECK-NEXT: ret

!0 = metadata !{i8 0, i8 2}
