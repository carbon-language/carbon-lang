; RUN: llc < %s -mtriple=i386-apple-darwin | FileCheck %s
; rdar://r7512579

; PHI defs in the atomic loop should be used by the add / adc
; instructions. They should not be dead.

define void @t(i64* nocapture %p) nounwind ssp {
entry:
; CHECK: t:
; CHECK: movl $1
; CHECK: movl (%ebp), %eax
; CHECK: movl 4(%ebp), %edx
; CHECK: LBB0_1:
; CHECK-NOT: movl $1
; CHECK-NOT: movl $0
; CHECK: addl
; CHECK: adcl
; CHECK: lock
; CHECK: cmpxchg8b
; CHECK: jne
  %0 = atomicrmw add i64* %p, i64 1 seq_cst
  ret void
}
