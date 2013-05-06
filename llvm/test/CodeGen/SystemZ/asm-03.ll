; Test the "S" asm constraint, which accepts addresses that have a base
; and a 20-bit displacement.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define void @f1(i64 %base) {
; CHECK: f1:
; CHECK: blah 0(%r2)
; CHECK: br %r14
  %addr = inttoptr i64 %base to i64 *
  call void asm "blah $0", "=*S" (i64 *%addr)
  ret void
}

; FIXME: at the moment the precise constraint is not passed down to
; target code, so we must conservatively treat "S" as "Q".
