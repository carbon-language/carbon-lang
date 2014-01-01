; RUN: llc < %s -march=sparcv9 | FileCheck %s

; CHECK-LABEL: test_atomic_i32
; CHECK:       ld [%o0]
; CHECK:       membar
; CHECK:       ld [%o1]
; CHECK:       membar
; CHECK:       membar
; CHECK:       st {{.+}}, [%o2]
define i32 @test_atomic_i32(i32* %ptr1, i32* %ptr2, i32* %ptr3) {
entry:
  %0 = load atomic i32* %ptr1 acquire, align 8
  %1 = load atomic i32* %ptr2 acquire, align 8
  %2 = add i32 %0, %1
  store atomic i32 %2, i32* %ptr3 release, align 8
  ret i32 %2
}

; CHECK-LABEL: test_atomic_i64
; CHECK:       ldx [%o0]
; CHECK:       membar
; CHECK:       ldx [%o1]
; CHECK:       membar
; CHECK:       membar
; CHECK:       stx {{.+}}, [%o2]
define i64 @test_atomic_i64(i64* %ptr1, i64* %ptr2, i64* %ptr3) {
entry:
  %0 = load atomic i64* %ptr1 acquire, align 8
  %1 = load atomic i64* %ptr2 acquire, align 8
  %2 = add i64 %0, %1
  store atomic i64 %2, i64* %ptr3 release, align 8
  ret i64 %2
}

; CHECK-LABEL: test_cmpxchg_i32
; CHECK:       or  %g0, 123, [[R:%[gilo][0-7]]]
; CHECK:       cas [%o1], %o0, [[R]]

define i32 @test_cmpxchg_i32(i32 %a, i32* %ptr) {
entry:
  %b = cmpxchg i32* %ptr, i32 %a, i32 123 monotonic
  ret i32 %b
}

; CHECK-LABEL: test_cmpxchg_i64
; CHECK:       or  %g0, 123, [[R:%[gilo][0-7]]]
; CHECK:       casx [%o1], %o0, [[R]]

define i64 @test_cmpxchg_i64(i64 %a, i64* %ptr) {
entry:
  %b = cmpxchg i64* %ptr, i64 %a, i64 123 monotonic
  ret i64 %b
}

; CHECK-LABEL: test_swap_i32
; CHECK:       or  %g0, 42, [[R:%[gilo][0-7]]]
; CHECK:       swap [%o1], [[R]]

define i32 @test_swap_i32(i32 %a, i32* %ptr) {
entry:
  %b = atomicrmw xchg i32* %ptr, i32 42 monotonic
  ret i32 %b
}
