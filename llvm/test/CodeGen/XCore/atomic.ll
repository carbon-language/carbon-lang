; RUN: llc < %s -march=xcore | FileCheck %s

; CHECK-LABEL: atomic_fence
; CHECK: #MEMBARRIER
; CHECK: #MEMBARRIER
; CHECK: #MEMBARRIER
; CHECK: #MEMBARRIER
; CHECK: retsp 0
define void @atomic_fence() nounwind {
entry:
  fence acquire
  fence release
  fence acq_rel
  fence seq_cst
  ret void
}

@pool = external global i64

define void @atomicloadstore() nounwind {
entry:
; CHECK-LABEL: atomicloadstore

; CHECK: ldw r[[R0:[0-9]+]], dp[pool]
; CHECK-NEXT: ldaw r[[R1:[0-9]+]], dp[pool]
; CHECK-NEXT: #MEMBARRIER
; CHECK-NEXT: ldc r[[R2:[0-9]+]], 0
  %0 = load atomic i32* bitcast (i64* @pool to i32*) acquire, align 4

; CHECK-NEXT: ld16s r3, r[[R1]][r[[R2]]]
; CHECK-NEXT: #MEMBARRIER
  %1 = load atomic i16* bitcast (i64* @pool to i16*) acquire, align 2

; CHECK-NEXT: ld8u r11, r[[R1]][r[[R2]]]
; CHECK-NEXT: #MEMBARRIER
  %2 = load atomic i8* bitcast (i64* @pool to i8*) acquire, align 1

; CHECK-NEXT: ldw r4, dp[pool]
; CHECK-NEXT: #MEMBARRIER
  %3 = load atomic i32* bitcast (i64* @pool to i32*) seq_cst, align 4

; CHECK-NEXT: ld16s r5, r[[R1]][r[[R2]]]
; CHECK-NEXT: #MEMBARRIER
  %4 = load atomic i16* bitcast (i64* @pool to i16*) seq_cst, align 2

; CHECK-NEXT: ld8u r6, r[[R1]][r[[R2]]]
; CHECK-NEXT: #MEMBARRIER
  %5 = load atomic i8* bitcast (i64* @pool to i8*) seq_cst, align 1

; CHECK-NEXT: #MEMBARRIER
; CHECK-NEXT: stw r[[R0]], dp[pool]
  store atomic i32 %0, i32* bitcast (i64* @pool to i32*) release, align 4

; CHECK-NEXT: #MEMBARRIER
; CHECK-NEXT: st16 r3, r[[R1]][r[[R2]]]
  store atomic i16 %1, i16* bitcast (i64* @pool to i16*) release, align 2

; CHECK-NEXT: #MEMBARRIER
; CHECK-NEXT: st8 r11, r[[R1]][r[[R2]]]
  store atomic i8 %2, i8* bitcast (i64* @pool to i8*) release, align 1

; CHECK-NEXT: #MEMBARRIER
; CHECK-NEXT: stw r4, dp[pool]
; CHECK-NEXT: #MEMBARRIER
  store atomic i32 %3, i32* bitcast (i64* @pool to i32*) seq_cst, align 4

; CHECK-NEXT: #MEMBARRIER
; CHECK-NEXT: st16 r5, r[[R1]][r[[R2]]]
; CHECK-NEXT: #MEMBARRIER
  store atomic i16 %4, i16* bitcast (i64* @pool to i16*) seq_cst, align 2

; CHECK-NEXT: #MEMBARRIER
; CHECK-NEXT: st8 r6, r[[R1]][r[[R2]]]
; CHECK-NEXT: #MEMBARRIER
  store atomic i8 %5, i8* bitcast (i64* @pool to i8*) seq_cst, align 1

; CHECK-NEXT: ldw r[[R0]], dp[pool]
; CHECK-NEXT: stw r[[R0]], dp[pool]
; CHECK-NEXT: ld16s r[[R0]], r[[R1]][r[[R2]]]
; CHECK-NEXT: st16 r[[R0]], r[[R1]][r[[R2]]]
; CHECK-NEXT: ld8u r[[R0]], r[[R1]][r[[R2]]]
; CHECK-NEXT: st8 r[[R0]], r[[R1]][r[[R2]]]
  %6 = load atomic i32* bitcast (i64* @pool to i32*) monotonic, align 4
  store atomic i32 %6, i32* bitcast (i64* @pool to i32*) monotonic, align 4
  %7 = load atomic i16* bitcast (i64* @pool to i16*) monotonic, align 2
  store atomic i16 %7, i16* bitcast (i64* @pool to i16*) monotonic, align 2
  %8 = load atomic i8* bitcast (i64* @pool to i8*) monotonic, align 1
  store atomic i8 %8, i8* bitcast (i64* @pool to i8*) monotonic, align 1

  ret void
}
