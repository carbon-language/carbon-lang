; RUN: llc -march=x86 -mattr=+cmov -mtriple=i386-pc-linux < %s | FileCheck %s
@sc64 = external global i64

define void @atomic_maxmin_i6432() {
; CHECK: atomic_maxmin_i6432
  %1 = atomicrmw max  i64* @sc64, i64 5 acquire
; CHECK: [[LABEL:.LBB[0-9]+_[0-9]+]]
; CHECK: cmpl
; CHECK: setl
; CHECK: cmpl
; CHECK: setl
; CHECK: cmovne
; CHECK: cmovne
; CHECK: lock
; CHECK-NEXT: cmpxchg8b
; CHECK: jne [[LABEL]]
  %2 = atomicrmw min  i64* @sc64, i64 6 acquire
; CHECK: [[LABEL:.LBB[0-9]+_[0-9]+]]
; CHECK: cmpl
; CHECK: setg
; CHECK: cmpl
; CHECK: setg
; CHECK: cmovne
; CHECK: cmovne
; CHECK: lock
; CHECK-NEXT: cmpxchg8b
; CHECK: jne [[LABEL]]
  %3 = atomicrmw umax i64* @sc64, i64 7 acquire
; CHECK: [[LABEL:.LBB[0-9]+_[0-9]+]]
; CHECK: cmpl
; CHECK: setb
; CHECK: cmpl
; CHECK: setb
; CHECK: cmovne
; CHECK: cmovne
; CHECK: lock
; CHECK-NEXT: cmpxchg8b
; CHECK: jne [[LABEL]]
  %4 = atomicrmw umin i64* @sc64, i64 8 acquire
; CHECK: [[LABEL:.LBB[0-9]+_[0-9]+]]
; CHECK: cmpl
; CHECK: seta
; CHECK: cmpl
; CHECK: seta
; CHECK: cmovne
; CHECK: cmovne
; CHECK: lock
; CHECK-NEXT: cmpxchg8b
; CHECK: jne [[LABEL]]
  ret void
}
