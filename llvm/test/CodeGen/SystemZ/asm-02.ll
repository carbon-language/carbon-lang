; Test the "R" asm constraint, which accepts addresses that have a base,
; an index and a 12-bit displacement.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check the lowest range.
define void @f1(i64 %base) {
; CHECK: f1:
; CHECK: blah 0(%r2)
; CHECK: br %r14
  %addr = inttoptr i64 %base to i64 *
  call void asm "blah $0", "=*R" (i64 *%addr)
  ret void
}

; Check the next lowest byte.
define void @f2(i64 %base) {
; CHECK: f2:
; CHECK: aghi %r2, -1
; CHECK: blah 0(%r2)
; CHECK: br %r14
  %add = add i64 %base, -1
  %addr = inttoptr i64 %add to i64 *
  call void asm "blah $0", "=*R" (i64 *%addr)
  ret void
}

; Check the highest range.
define void @f3(i64 %base) {
; CHECK: f3:
; CHECK: blah 4095(%r2)
; CHECK: br %r14
  %add = add i64 %base, 4095
  %addr = inttoptr i64 %add to i64 *
  call void asm "blah $0", "=*R" (i64 *%addr)
  ret void
}

; Check the next highest byte.
define void @f4(i64 %base) {
; CHECK: f4:
; CHECK: aghi %r2, 4096
; CHECK: blah 0(%r2)
; CHECK: br %r14
  %add = add i64 %base, 4096
  %addr = inttoptr i64 %add to i64 *
  call void asm "blah $0", "=*R" (i64 *%addr)
  ret void
}

; FIXME: at the moment the precise constraint is not passed down to
; target code, so we must conservatively treat "R" as "Q".
