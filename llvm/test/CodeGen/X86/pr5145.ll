; RUN: llc -march=x86-64 < %s | FileCheck %s
@sc8 = external global i8

define void @atomic_maxmin_i8() {
; CHECK: atomic_maxmin_i8
  %1 = atomicrmw max  i8* @sc8, i8 5 acquire
; CHECK: [[LABEL:.LBB[0-9]+_[0-9]+]]
; CHECK: cmpb
; CHECK: cmovl
; CHECK: lock
; CHECK-NEXT: cmpxchgb
; CHECK: jne [[LABEL]]
  %2 = atomicrmw min  i8* @sc8, i8 6 acquire
; CHECK: [[LABEL:.LBB[0-9]+_[0-9]+]]
; CHECK: cmpb
; CHECK: cmovg
; CHECK: lock
; CHECK-NEXT: cmpxchgb
; CHECK: jne [[LABEL]]
  %3 = atomicrmw umax i8* @sc8, i8 7 acquire
; CHECK: [[LABEL:.LBB[0-9]+_[0-9]+]]
; CHECK: cmpb
; CHECK: cmovb
; CHECK: lock
; CHECK-NEXT: cmpxchgb
; CHECK: jne [[LABEL]]
  %4 = atomicrmw umin i8* @sc8, i8 8 acquire
; CHECK: [[LABEL:.LBB[0-9]+_[0-9]+]]
; CHECK: cmpb
; CHECK: cmova
; CHECK: lock
; CHECK-NEXT: cmpxchgb
; CHECK: jne [[LABEL]]
  ret void
}
