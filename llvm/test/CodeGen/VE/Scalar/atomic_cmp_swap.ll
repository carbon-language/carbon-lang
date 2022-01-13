; RUN: llc < %s -mtriple=ve | FileCheck %s

;;; Test atomic compare and exchange weak for all types and all memory order
;;;
;;; Note:
;;;   - We test i1/i8/i16/i32/i64/i128/u8/u16/u32/u64/u128.
;;;   - We test relaxed, acquire, and seq_cst.
;;;   - We test only exchange with variables since VE doesn't have exchange
;;;     instructions with immediate values.
;;;   - We test against an object, a stack object, and a global variable.

%"struct.std::__1::atomic" = type { %"struct.std::__1::__atomic_base" }
%"struct.std::__1::__atomic_base" = type { %"struct.std::__1::__cxx_atomic_impl" }
%"struct.std::__1::__cxx_atomic_impl" = type { %"struct.std::__1::__cxx_atomic_base_impl" }
%"struct.std::__1::__cxx_atomic_base_impl" = type { i8 }
%"struct.std::__1::atomic.0" = type { %"struct.std::__1::__atomic_base.1" }
%"struct.std::__1::__atomic_base.1" = type { %"struct.std::__1::__atomic_base.2" }
%"struct.std::__1::__atomic_base.2" = type { %"struct.std::__1::__cxx_atomic_impl.3" }
%"struct.std::__1::__cxx_atomic_impl.3" = type { %"struct.std::__1::__cxx_atomic_base_impl.4" }
%"struct.std::__1::__cxx_atomic_base_impl.4" = type { i8 }
%"struct.std::__1::atomic.5" = type { %"struct.std::__1::__atomic_base.6" }
%"struct.std::__1::__atomic_base.6" = type { %"struct.std::__1::__atomic_base.7" }
%"struct.std::__1::__atomic_base.7" = type { %"struct.std::__1::__cxx_atomic_impl.8" }
%"struct.std::__1::__cxx_atomic_impl.8" = type { %"struct.std::__1::__cxx_atomic_base_impl.9" }
%"struct.std::__1::__cxx_atomic_base_impl.9" = type { i8 }
%"struct.std::__1::atomic.10" = type { %"struct.std::__1::__atomic_base.11" }
%"struct.std::__1::__atomic_base.11" = type { %"struct.std::__1::__atomic_base.12" }
%"struct.std::__1::__atomic_base.12" = type { %"struct.std::__1::__cxx_atomic_impl.13" }
%"struct.std::__1::__cxx_atomic_impl.13" = type { %"struct.std::__1::__cxx_atomic_base_impl.14" }
%"struct.std::__1::__cxx_atomic_base_impl.14" = type { i16 }
%"struct.std::__1::atomic.15" = type { %"struct.std::__1::__atomic_base.16" }
%"struct.std::__1::__atomic_base.16" = type { %"struct.std::__1::__atomic_base.17" }
%"struct.std::__1::__atomic_base.17" = type { %"struct.std::__1::__cxx_atomic_impl.18" }
%"struct.std::__1::__cxx_atomic_impl.18" = type { %"struct.std::__1::__cxx_atomic_base_impl.19" }
%"struct.std::__1::__cxx_atomic_base_impl.19" = type { i16 }
%"struct.std::__1::atomic.20" = type { %"struct.std::__1::__atomic_base.21" }
%"struct.std::__1::__atomic_base.21" = type { %"struct.std::__1::__atomic_base.22" }
%"struct.std::__1::__atomic_base.22" = type { %"struct.std::__1::__cxx_atomic_impl.23" }
%"struct.std::__1::__cxx_atomic_impl.23" = type { %"struct.std::__1::__cxx_atomic_base_impl.24" }
%"struct.std::__1::__cxx_atomic_base_impl.24" = type { i32 }
%"struct.std::__1::atomic.25" = type { %"struct.std::__1::__atomic_base.26" }
%"struct.std::__1::__atomic_base.26" = type { %"struct.std::__1::__atomic_base.27" }
%"struct.std::__1::__atomic_base.27" = type { %"struct.std::__1::__cxx_atomic_impl.28" }
%"struct.std::__1::__cxx_atomic_impl.28" = type { %"struct.std::__1::__cxx_atomic_base_impl.29" }
%"struct.std::__1::__cxx_atomic_base_impl.29" = type { i32 }
%"struct.std::__1::atomic.30" = type { %"struct.std::__1::__atomic_base.31" }
%"struct.std::__1::__atomic_base.31" = type { %"struct.std::__1::__atomic_base.32" }
%"struct.std::__1::__atomic_base.32" = type { %"struct.std::__1::__cxx_atomic_impl.33" }
%"struct.std::__1::__cxx_atomic_impl.33" = type { %"struct.std::__1::__cxx_atomic_base_impl.34" }
%"struct.std::__1::__cxx_atomic_base_impl.34" = type { i64 }
%"struct.std::__1::atomic.35" = type { %"struct.std::__1::__atomic_base.36" }
%"struct.std::__1::__atomic_base.36" = type { %"struct.std::__1::__atomic_base.37" }
%"struct.std::__1::__atomic_base.37" = type { %"struct.std::__1::__cxx_atomic_impl.38" }
%"struct.std::__1::__cxx_atomic_impl.38" = type { %"struct.std::__1::__cxx_atomic_base_impl.39" }
%"struct.std::__1::__cxx_atomic_base_impl.39" = type { i64 }
%"struct.std::__1::atomic.40" = type { %"struct.std::__1::__atomic_base.41" }
%"struct.std::__1::__atomic_base.41" = type { %"struct.std::__1::__atomic_base.42" }
%"struct.std::__1::__atomic_base.42" = type { %"struct.std::__1::__cxx_atomic_impl.43" }
%"struct.std::__1::__cxx_atomic_impl.43" = type { %"struct.std::__1::__cxx_atomic_base_impl.44" }
%"struct.std::__1::__cxx_atomic_base_impl.44" = type { i128 }
%"struct.std::__1::atomic.45" = type { %"struct.std::__1::__atomic_base.46" }
%"struct.std::__1::__atomic_base.46" = type { %"struct.std::__1::__atomic_base.47" }
%"struct.std::__1::__atomic_base.47" = type { %"struct.std::__1::__cxx_atomic_impl.48" }
%"struct.std::__1::__cxx_atomic_impl.48" = type { %"struct.std::__1::__cxx_atomic_base_impl.49" }
%"struct.std::__1::__cxx_atomic_base_impl.49" = type { i128 }

@gv_i1 = global %"struct.std::__1::atomic" zeroinitializer, align 4
@gv_i8 = global %"struct.std::__1::atomic.0" zeroinitializer, align 4
@gv_u8 = global %"struct.std::__1::atomic.5" zeroinitializer, align 4
@gv_i16 = global %"struct.std::__1::atomic.10" zeroinitializer, align 4
@gv_u16 = global %"struct.std::__1::atomic.15" zeroinitializer, align 4
@gv_i32 = global %"struct.std::__1::atomic.20" zeroinitializer, align 4
@gv_u32 = global %"struct.std::__1::atomic.25" zeroinitializer, align 4
@gv_i64 = global %"struct.std::__1::atomic.30" zeroinitializer, align 8
@gv_u64 = global %"struct.std::__1::atomic.35" zeroinitializer, align 8
@gv_i128 = global %"struct.std::__1::atomic.40" zeroinitializer, align 16
@gv_u128 = global %"struct.std::__1::atomic.45" zeroinitializer, align 16

; Function Attrs: nofree norecurse nounwind mustprogress
define zeroext i1 @_Z26atomic_cmp_swap_relaxed_i1RNSt3__16atomicIbEERbb(%"struct.std::__1::atomic"* nocapture nonnull align 1 dereferenceable(1) %0, i8* nocapture nonnull align 1 dereferenceable(1) %1, i1 zeroext %2) {
; CHECK-LABEL: _Z26atomic_cmp_swap_relaxed_i1RNSt3__16atomicIbEERbb:
; CHECK:       # %bb.0: # %partword.cmpxchg.loop
; CHECK-NEXT:    ld1b.zx %s3, (, %s1)
; CHECK-NEXT:    and %s4, -4, %s0
; CHECK-NEXT:    and %s0, 3, %s0
; CHECK-NEXT:    sla.w.sx %s0, %s0, 3
; CHECK-NEXT:    ldl.sx %s5, (, %s4)
; CHECK-NEXT:    sla.w.sx %s6, (56)0, %s0
; CHECK-NEXT:    sla.w.sx %s2, %s2, %s0
; CHECK-NEXT:    sla.w.sx %s3, %s3, %s0
; CHECK-NEXT:    nnd %s5, %s6, %s5
; CHECK-NEXT:    or %s2, %s5, %s2
; CHECK-NEXT:    or %s5, %s5, %s3
; CHECK-NEXT:    cas.w %s2, (%s4), %s5
; CHECK-NEXT:    cmps.w.sx %s4, %s2, %s5
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s3, (63)0, %s4
; CHECK-NEXT:    breq.w %s2, %s5, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    srl %s0, %s2, %s0
; CHECK-NEXT:    st1b %s0, (, %s1)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = zext i1 %2 to i8
  %5 = getelementptr inbounds %"struct.std::__1::atomic", %"struct.std::__1::atomic"* %0, i64 0, i32 0, i32 0, i32 0, i32 0
  %6 = load i8, i8* %1, align 1
  %7 = cmpxchg weak i8* %5, i8 %6, i8 %4 monotonic monotonic
  %8 = extractvalue { i8, i1 } %7, 1
  br i1 %8, label %11, label %9

9:                                                ; preds = %3
  %10 = extractvalue { i8, i1 } %7, 0
  store i8 %10, i8* %1, align 1
  br label %11

11:                                               ; preds = %3, %9
  ret i1 %8
}

; Function Attrs: nofree norecurse nounwind mustprogress
define signext i8 @_Z26atomic_cmp_swap_relaxed_i8RNSt3__16atomicIcEERcc(%"struct.std::__1::atomic.0"* nocapture nonnull align 1 dereferenceable(1) %0, i8* nocapture nonnull align 1 dereferenceable(1) %1, i8 signext %2) {
; CHECK-LABEL: _Z26atomic_cmp_swap_relaxed_i8RNSt3__16atomicIcEERcc:
; CHECK:       # %bb.0: # %partword.cmpxchg.loop
; CHECK-NEXT:    ld1b.zx %s3, (, %s1)
; CHECK-NEXT:    and %s4, -4, %s0
; CHECK-NEXT:    and %s0, 3, %s0
; CHECK-NEXT:    sla.w.sx %s0, %s0, 3
; CHECK-NEXT:    sla.w.sx %s5, (56)0, %s0
; CHECK-NEXT:    ldl.sx %s6, (, %s4)
; CHECK-NEXT:    and %s2, %s2, (56)0
; CHECK-NEXT:    sla.w.sx %s2, %s2, %s0
; CHECK-NEXT:    sla.w.sx %s3, %s3, %s0
; CHECK-NEXT:    nnd %s5, %s5, %s6
; CHECK-NEXT:    or %s2, %s5, %s2
; CHECK-NEXT:    or %s5, %s5, %s3
; CHECK-NEXT:    cas.w %s2, (%s4), %s5
; CHECK-NEXT:    cmps.w.sx %s4, %s2, %s5
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s3, (63)0, %s4
; CHECK-NEXT:    breq.w %s2, %s5, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    srl %s0, %s2, %s0
; CHECK-NEXT:    st1b %s0, (, %s1)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = getelementptr inbounds %"struct.std::__1::atomic.0", %"struct.std::__1::atomic.0"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %5 = load i8, i8* %1, align 1
  %6 = cmpxchg weak i8* %4, i8 %5, i8 %2 monotonic monotonic
  %7 = extractvalue { i8, i1 } %6, 1
  br i1 %7, label %10, label %8

8:                                                ; preds = %3
  %9 = extractvalue { i8, i1 } %6, 0
  store i8 %9, i8* %1, align 1
  br label %10

10:                                               ; preds = %3, %8
  %11 = zext i1 %7 to i8
  ret i8 %11
}

; Function Attrs: nofree norecurse nounwind mustprogress
define zeroext i8 @_Z26atomic_cmp_swap_relaxed_u8RNSt3__16atomicIhEERhh(%"struct.std::__1::atomic.5"* nocapture nonnull align 1 dereferenceable(1) %0, i8* nocapture nonnull align 1 dereferenceable(1) %1, i8 zeroext %2) {
; CHECK-LABEL: _Z26atomic_cmp_swap_relaxed_u8RNSt3__16atomicIhEERhh:
; CHECK:       # %bb.0: # %partword.cmpxchg.loop
; CHECK-NEXT:    ld1b.zx %s3, (, %s1)
; CHECK-NEXT:    and %s4, -4, %s0
; CHECK-NEXT:    and %s0, 3, %s0
; CHECK-NEXT:    sla.w.sx %s0, %s0, 3
; CHECK-NEXT:    ldl.sx %s5, (, %s4)
; CHECK-NEXT:    sla.w.sx %s6, (56)0, %s0
; CHECK-NEXT:    sla.w.sx %s2, %s2, %s0
; CHECK-NEXT:    sla.w.sx %s3, %s3, %s0
; CHECK-NEXT:    nnd %s5, %s6, %s5
; CHECK-NEXT:    or %s2, %s5, %s2
; CHECK-NEXT:    or %s5, %s5, %s3
; CHECK-NEXT:    cas.w %s2, (%s4), %s5
; CHECK-NEXT:    cmps.w.sx %s4, %s2, %s5
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s3, (63)0, %s4
; CHECK-NEXT:    breq.w %s2, %s5, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    srl %s0, %s2, %s0
; CHECK-NEXT:    st1b %s0, (, %s1)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = getelementptr inbounds %"struct.std::__1::atomic.5", %"struct.std::__1::atomic.5"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %5 = load i8, i8* %1, align 1
  %6 = cmpxchg weak i8* %4, i8 %5, i8 %2 monotonic monotonic
  %7 = extractvalue { i8, i1 } %6, 1
  br i1 %7, label %10, label %8

8:                                                ; preds = %3
  %9 = extractvalue { i8, i1 } %6, 0
  store i8 %9, i8* %1, align 1
  br label %10

10:                                               ; preds = %3, %8
  %11 = zext i1 %7 to i8
  ret i8 %11
}

; Function Attrs: nofree norecurse nounwind mustprogress
define signext i16 @_Z27atomic_cmp_swap_relaxed_i16RNSt3__16atomicIsEERss(%"struct.std::__1::atomic.10"* nocapture nonnull align 2 dereferenceable(2) %0, i16* nocapture nonnull align 2 dereferenceable(2) %1, i16 signext %2) {
; CHECK-LABEL: _Z27atomic_cmp_swap_relaxed_i16RNSt3__16atomicIsEERss:
; CHECK:       # %bb.0: # %partword.cmpxchg.loop
; CHECK-NEXT:    ld2b.zx %s3, (, %s1)
; CHECK-NEXT:    and %s4, -4, %s0
; CHECK-NEXT:    and %s0, 3, %s0
; CHECK-NEXT:    sla.w.sx %s0, %s0, 3
; CHECK-NEXT:    sla.w.sx %s5, (48)0, %s0
; CHECK-NEXT:    ldl.sx %s6, (, %s4)
; CHECK-NEXT:    and %s2, %s2, (48)0
; CHECK-NEXT:    sla.w.sx %s2, %s2, %s0
; CHECK-NEXT:    sla.w.sx %s3, %s3, %s0
; CHECK-NEXT:    nnd %s5, %s5, %s6
; CHECK-NEXT:    or %s2, %s5, %s2
; CHECK-NEXT:    or %s5, %s5, %s3
; CHECK-NEXT:    cas.w %s2, (%s4), %s5
; CHECK-NEXT:    cmps.w.sx %s4, %s2, %s5
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s3, (63)0, %s4
; CHECK-NEXT:    breq.w %s2, %s5, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    srl %s0, %s2, %s0
; CHECK-NEXT:    st2b %s0, (, %s1)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = getelementptr inbounds %"struct.std::__1::atomic.10", %"struct.std::__1::atomic.10"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %5 = load i16, i16* %1, align 2
  %6 = cmpxchg weak i16* %4, i16 %5, i16 %2 monotonic monotonic
  %7 = extractvalue { i16, i1 } %6, 1
  br i1 %7, label %10, label %8

8:                                                ; preds = %3
  %9 = extractvalue { i16, i1 } %6, 0
  store i16 %9, i16* %1, align 2
  br label %10

10:                                               ; preds = %3, %8
  %11 = zext i1 %7 to i16
  ret i16 %11
}

; Function Attrs: nofree norecurse nounwind mustprogress
define zeroext i16 @_Z27atomic_cmp_swap_relaxed_u16RNSt3__16atomicItEERtt(%"struct.std::__1::atomic.15"* nocapture nonnull align 2 dereferenceable(2) %0, i16* nocapture nonnull align 2 dereferenceable(2) %1, i16 zeroext %2) {
; CHECK-LABEL: _Z27atomic_cmp_swap_relaxed_u16RNSt3__16atomicItEERtt:
; CHECK:       # %bb.0: # %partword.cmpxchg.loop
; CHECK-NEXT:    ld2b.zx %s3, (, %s1)
; CHECK-NEXT:    and %s4, -4, %s0
; CHECK-NEXT:    and %s0, 3, %s0
; CHECK-NEXT:    sla.w.sx %s0, %s0, 3
; CHECK-NEXT:    ldl.sx %s5, (, %s4)
; CHECK-NEXT:    sla.w.sx %s6, (48)0, %s0
; CHECK-NEXT:    sla.w.sx %s2, %s2, %s0
; CHECK-NEXT:    sla.w.sx %s3, %s3, %s0
; CHECK-NEXT:    nnd %s5, %s6, %s5
; CHECK-NEXT:    or %s2, %s5, %s2
; CHECK-NEXT:    or %s5, %s5, %s3
; CHECK-NEXT:    cas.w %s2, (%s4), %s5
; CHECK-NEXT:    cmps.w.sx %s4, %s2, %s5
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s3, (63)0, %s4
; CHECK-NEXT:    breq.w %s2, %s5, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    srl %s0, %s2, %s0
; CHECK-NEXT:    st2b %s0, (, %s1)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = getelementptr inbounds %"struct.std::__1::atomic.15", %"struct.std::__1::atomic.15"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %5 = load i16, i16* %1, align 2
  %6 = cmpxchg weak i16* %4, i16 %5, i16 %2 monotonic monotonic
  %7 = extractvalue { i16, i1 } %6, 1
  br i1 %7, label %10, label %8

8:                                                ; preds = %3
  %9 = extractvalue { i16, i1 } %6, 0
  store i16 %9, i16* %1, align 2
  br label %10

10:                                               ; preds = %3, %8
  %11 = zext i1 %7 to i16
  ret i16 %11
}

; Function Attrs: nofree norecurse nounwind mustprogress
define signext i32 @_Z27atomic_cmp_swap_relaxed_i32RNSt3__16atomicIiEERii(%"struct.std::__1::atomic.20"* nocapture nonnull align 4 dereferenceable(4) %0, i32* nocapture nonnull align 4 dereferenceable(4) %1, i32 signext %2) {
; CHECK-LABEL: _Z27atomic_cmp_swap_relaxed_i32RNSt3__16atomicIiEERii:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ldl.sx %s3, (, %s1)
; CHECK-NEXT:    cas.w %s2, (%s0), %s3
; CHECK-NEXT:    cmps.w.sx %s4, %s2, %s3
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s0, (63)0, %s4
; CHECK-NEXT:    breq.w %s2, %s3, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    stl %s2, (, %s1)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = getelementptr inbounds %"struct.std::__1::atomic.20", %"struct.std::__1::atomic.20"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %5 = load i32, i32* %1, align 4
  %6 = cmpxchg weak i32* %4, i32 %5, i32 %2 monotonic monotonic
  %7 = extractvalue { i32, i1 } %6, 1
  br i1 %7, label %10, label %8

8:                                                ; preds = %3
  %9 = extractvalue { i32, i1 } %6, 0
  store i32 %9, i32* %1, align 4
  br label %10

10:                                               ; preds = %3, %8
  %11 = zext i1 %7 to i32
  ret i32 %11
}

; Function Attrs: nofree norecurse nounwind mustprogress
define zeroext i32 @_Z27atomic_cmp_swap_relaxed_u32RNSt3__16atomicIjEERjj(%"struct.std::__1::atomic.25"* nocapture nonnull align 4 dereferenceable(4) %0, i32* nocapture nonnull align 4 dereferenceable(4) %1, i32 zeroext %2) {
; CHECK-LABEL: _Z27atomic_cmp_swap_relaxed_u32RNSt3__16atomicIjEERjj:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ldl.sx %s3, (, %s1)
; CHECK-NEXT:    cas.w %s2, (%s0), %s3
; CHECK-NEXT:    cmps.w.sx %s4, %s2, %s3
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s0, (63)0, %s4
; CHECK-NEXT:    breq.w %s2, %s3, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    stl %s2, (, %s1)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = getelementptr inbounds %"struct.std::__1::atomic.25", %"struct.std::__1::atomic.25"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %5 = load i32, i32* %1, align 4
  %6 = cmpxchg weak i32* %4, i32 %5, i32 %2 monotonic monotonic
  %7 = extractvalue { i32, i1 } %6, 1
  br i1 %7, label %10, label %8

8:                                                ; preds = %3
  %9 = extractvalue { i32, i1 } %6, 0
  store i32 %9, i32* %1, align 4
  br label %10

10:                                               ; preds = %3, %8
  %11 = zext i1 %7 to i32
  ret i32 %11
}

; Function Attrs: nofree norecurse nounwind mustprogress
define i64 @_Z27atomic_cmp_swap_relaxed_i64RNSt3__16atomicIlEERll(%"struct.std::__1::atomic.30"* nocapture nonnull align 8 dereferenceable(8) %0, i64* nocapture nonnull align 8 dereferenceable(8) %1, i64 %2) {
; CHECK-LABEL: _Z27atomic_cmp_swap_relaxed_i64RNSt3__16atomicIlEERll:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld %s3, (, %s1)
; CHECK-NEXT:    cas.l %s2, (%s0), %s3
; CHECK-NEXT:    cmps.l %s4, %s2, %s3
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    cmov.l.eq %s0, (63)0, %s4
; CHECK-NEXT:    breq.l %s2, %s3, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    st %s2, (, %s1)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = getelementptr inbounds %"struct.std::__1::atomic.30", %"struct.std::__1::atomic.30"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %5 = load i64, i64* %1, align 8
  %6 = cmpxchg weak i64* %4, i64 %5, i64 %2 monotonic monotonic
  %7 = extractvalue { i64, i1 } %6, 1
  br i1 %7, label %10, label %8

8:                                                ; preds = %3
  %9 = extractvalue { i64, i1 } %6, 0
  store i64 %9, i64* %1, align 8
  br label %10

10:                                               ; preds = %3, %8
  %11 = zext i1 %7 to i64
  ret i64 %11
}

; Function Attrs: nofree norecurse nounwind mustprogress
define i64 @_Z27atomic_cmp_swap_relaxed_u64RNSt3__16atomicImEERmm(%"struct.std::__1::atomic.35"* nocapture nonnull align 8 dereferenceable(8) %0, i64* nocapture nonnull align 8 dereferenceable(8) %1, i64 %2) {
; CHECK-LABEL: _Z27atomic_cmp_swap_relaxed_u64RNSt3__16atomicImEERmm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld %s3, (, %s1)
; CHECK-NEXT:    cas.l %s2, (%s0), %s3
; CHECK-NEXT:    cmps.l %s4, %s2, %s3
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    cmov.l.eq %s0, (63)0, %s4
; CHECK-NEXT:    breq.l %s2, %s3, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    st %s2, (, %s1)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = getelementptr inbounds %"struct.std::__1::atomic.35", %"struct.std::__1::atomic.35"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %5 = load i64, i64* %1, align 8
  %6 = cmpxchg weak i64* %4, i64 %5, i64 %2 monotonic monotonic
  %7 = extractvalue { i64, i1 } %6, 1
  br i1 %7, label %10, label %8

8:                                                ; preds = %3
  %9 = extractvalue { i64, i1 } %6, 0
  store i64 %9, i64* %1, align 8
  br label %10

10:                                               ; preds = %3, %8
  %11 = zext i1 %7 to i64
  ret i64 %11
}

; Function Attrs: nounwind mustprogress
define i128 @_Z28atomic_cmp_swap_relaxed_i128RNSt3__16atomicInEERnn(%"struct.std::__1::atomic.40"* nonnull align 16 dereferenceable(16) %0, i128* nonnull align 16 dereferenceable(16) %1, i128 %2) {
; CHECK-LABEL: _Z28atomic_cmp_swap_relaxed_i128RNSt3__16atomicInEERnn:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s6, 0, %s1
; CHECK-NEXT:    or %s1, 0, %s0
; CHECK-NEXT:    st %s3, 248(, %s11)
; CHECK-NEXT:    st %s2, 240(, %s11)
; CHECK-NEXT:    lea %s0, __atomic_compare_exchange@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, __atomic_compare_exchange@hi(, %s0)
; CHECK-NEXT:    lea %s3, 240(, %s11)
; CHECK-NEXT:    or %s0, 16, (0)1
; CHECK-NEXT:    or %s4, 0, (0)1
; CHECK-NEXT:    or %s5, 0, (0)1
; CHECK-NEXT:    or %s2, 0, %s6
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %4 = alloca i128, align 16
  %5 = bitcast i128* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5)
  store i128 %2, i128* %4, align 16, !tbaa !2
  %6 = bitcast %"struct.std::__1::atomic.40"* %0 to i8*
  %7 = bitcast i128* %1 to i8*
  %8 = call zeroext i1 @__atomic_compare_exchange(i64 16, i8* nonnull %6, i8* nonnull %7, i8* nonnull %5, i32 signext 0, i32 signext 0)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5)
  %9 = zext i1 %8 to i128
  ret i128 %9
}

; Function Attrs: nounwind mustprogress
define i128 @_Z28atomic_cmp_swap_relaxed_u128RNSt3__16atomicIoEERoo(%"struct.std::__1::atomic.45"* nonnull align 16 dereferenceable(16) %0, i128* nonnull align 16 dereferenceable(16) %1, i128 %2) {
; CHECK-LABEL: _Z28atomic_cmp_swap_relaxed_u128RNSt3__16atomicIoEERoo:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s6, 0, %s1
; CHECK-NEXT:    or %s1, 0, %s0
; CHECK-NEXT:    st %s3, 248(, %s11)
; CHECK-NEXT:    st %s2, 240(, %s11)
; CHECK-NEXT:    lea %s0, __atomic_compare_exchange@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, __atomic_compare_exchange@hi(, %s0)
; CHECK-NEXT:    lea %s3, 240(, %s11)
; CHECK-NEXT:    or %s0, 16, (0)1
; CHECK-NEXT:    or %s4, 0, (0)1
; CHECK-NEXT:    or %s5, 0, (0)1
; CHECK-NEXT:    or %s2, 0, %s6
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %4 = alloca i128, align 16
  %5 = bitcast i128* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5)
  store i128 %2, i128* %4, align 16, !tbaa !2
  %6 = bitcast %"struct.std::__1::atomic.45"* %0 to i8*
  %7 = bitcast i128* %1 to i8*
  %8 = call zeroext i1 @__atomic_compare_exchange(i64 16, i8* nonnull %6, i8* nonnull %7, i8* nonnull %5, i32 signext 0, i32 signext 0)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5)
  %9 = zext i1 %8 to i128
  ret i128 %9
}

; Function Attrs: nofree norecurse nounwind mustprogress
define zeroext i1 @_Z26atomic_cmp_swap_acquire_i1RNSt3__16atomicIbEERbb(%"struct.std::__1::atomic"* nocapture nonnull align 1 dereferenceable(1) %0, i8* nocapture nonnull align 1 dereferenceable(1) %1, i1 zeroext %2) {
; CHECK-LABEL: _Z26atomic_cmp_swap_acquire_i1RNSt3__16atomicIbEERbb:
; CHECK:       # %bb.0: # %partword.cmpxchg.loop
; CHECK-NEXT:    ld1b.zx %s3, (, %s1)
; CHECK-NEXT:    and %s4, -4, %s0
; CHECK-NEXT:    and %s0, 3, %s0
; CHECK-NEXT:    sla.w.sx %s0, %s0, 3
; CHECK-NEXT:    ldl.sx %s5, (, %s4)
; CHECK-NEXT:    sla.w.sx %s6, (56)0, %s0
; CHECK-NEXT:    sla.w.sx %s2, %s2, %s0
; CHECK-NEXT:    sla.w.sx %s3, %s3, %s0
; CHECK-NEXT:    nnd %s5, %s6, %s5
; CHECK-NEXT:    or %s2, %s5, %s2
; CHECK-NEXT:    or %s5, %s5, %s3
; CHECK-NEXT:    cas.w %s2, (%s4), %s5
; CHECK-NEXT:    cmps.w.sx %s4, %s2, %s5
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s3, (63)0, %s4
; CHECK-NEXT:    fencem 2
; CHECK-NEXT:    breq.w %s2, %s5, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    srl %s0, %s2, %s0
; CHECK-NEXT:    st1b %s0, (, %s1)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = zext i1 %2 to i8
  %5 = getelementptr inbounds %"struct.std::__1::atomic", %"struct.std::__1::atomic"* %0, i64 0, i32 0, i32 0, i32 0, i32 0
  %6 = load i8, i8* %1, align 1
  %7 = cmpxchg weak i8* %5, i8 %6, i8 %4 acquire acquire
  %8 = extractvalue { i8, i1 } %7, 1
  br i1 %8, label %11, label %9

9:                                                ; preds = %3
  %10 = extractvalue { i8, i1 } %7, 0
  store i8 %10, i8* %1, align 1
  br label %11

11:                                               ; preds = %3, %9
  ret i1 %8
}

; Function Attrs: nofree norecurse nounwind mustprogress
define signext i8 @_Z26atomic_cmp_swap_acquire_i8RNSt3__16atomicIcEERcc(%"struct.std::__1::atomic.0"* nocapture nonnull align 1 dereferenceable(1) %0, i8* nocapture nonnull align 1 dereferenceable(1) %1, i8 signext %2) {
; CHECK-LABEL: _Z26atomic_cmp_swap_acquire_i8RNSt3__16atomicIcEERcc:
; CHECK:       # %bb.0: # %partword.cmpxchg.loop
; CHECK-NEXT:    ld1b.zx %s3, (, %s1)
; CHECK-NEXT:    and %s4, -4, %s0
; CHECK-NEXT:    and %s0, 3, %s0
; CHECK-NEXT:    sla.w.sx %s0, %s0, 3
; CHECK-NEXT:    sla.w.sx %s5, (56)0, %s0
; CHECK-NEXT:    ldl.sx %s6, (, %s4)
; CHECK-NEXT:    and %s2, %s2, (56)0
; CHECK-NEXT:    sla.w.sx %s2, %s2, %s0
; CHECK-NEXT:    sla.w.sx %s3, %s3, %s0
; CHECK-NEXT:    nnd %s5, %s5, %s6
; CHECK-NEXT:    or %s2, %s5, %s2
; CHECK-NEXT:    or %s5, %s5, %s3
; CHECK-NEXT:    cas.w %s2, (%s4), %s5
; CHECK-NEXT:    cmps.w.sx %s4, %s2, %s5
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s3, (63)0, %s4
; CHECK-NEXT:    fencem 2
; CHECK-NEXT:    breq.w %s2, %s5, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    srl %s0, %s2, %s0
; CHECK-NEXT:    st1b %s0, (, %s1)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = getelementptr inbounds %"struct.std::__1::atomic.0", %"struct.std::__1::atomic.0"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %5 = load i8, i8* %1, align 1
  %6 = cmpxchg weak i8* %4, i8 %5, i8 %2 acquire acquire
  %7 = extractvalue { i8, i1 } %6, 1
  br i1 %7, label %10, label %8

8:                                                ; preds = %3
  %9 = extractvalue { i8, i1 } %6, 0
  store i8 %9, i8* %1, align 1
  br label %10

10:                                               ; preds = %3, %8
  %11 = zext i1 %7 to i8
  ret i8 %11
}

; Function Attrs: nofree norecurse nounwind mustprogress
define zeroext i8 @_Z26atomic_cmp_swap_acquire_u8RNSt3__16atomicIhEERhh(%"struct.std::__1::atomic.5"* nocapture nonnull align 1 dereferenceable(1) %0, i8* nocapture nonnull align 1 dereferenceable(1) %1, i8 zeroext %2) {
; CHECK-LABEL: _Z26atomic_cmp_swap_acquire_u8RNSt3__16atomicIhEERhh:
; CHECK:       # %bb.0: # %partword.cmpxchg.loop
; CHECK-NEXT:    ld1b.zx %s3, (, %s1)
; CHECK-NEXT:    and %s4, -4, %s0
; CHECK-NEXT:    and %s0, 3, %s0
; CHECK-NEXT:    sla.w.sx %s0, %s0, 3
; CHECK-NEXT:    ldl.sx %s5, (, %s4)
; CHECK-NEXT:    sla.w.sx %s6, (56)0, %s0
; CHECK-NEXT:    sla.w.sx %s2, %s2, %s0
; CHECK-NEXT:    sla.w.sx %s3, %s3, %s0
; CHECK-NEXT:    nnd %s5, %s6, %s5
; CHECK-NEXT:    or %s2, %s5, %s2
; CHECK-NEXT:    or %s5, %s5, %s3
; CHECK-NEXT:    cas.w %s2, (%s4), %s5
; CHECK-NEXT:    cmps.w.sx %s4, %s2, %s5
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s3, (63)0, %s4
; CHECK-NEXT:    fencem 2
; CHECK-NEXT:    breq.w %s2, %s5, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    srl %s0, %s2, %s0
; CHECK-NEXT:    st1b %s0, (, %s1)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = getelementptr inbounds %"struct.std::__1::atomic.5", %"struct.std::__1::atomic.5"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %5 = load i8, i8* %1, align 1
  %6 = cmpxchg weak i8* %4, i8 %5, i8 %2 acquire acquire
  %7 = extractvalue { i8, i1 } %6, 1
  br i1 %7, label %10, label %8

8:                                                ; preds = %3
  %9 = extractvalue { i8, i1 } %6, 0
  store i8 %9, i8* %1, align 1
  br label %10

10:                                               ; preds = %3, %8
  %11 = zext i1 %7 to i8
  ret i8 %11
}

; Function Attrs: nofree norecurse nounwind mustprogress
define signext i16 @_Z27atomic_cmp_swap_acquire_i16RNSt3__16atomicIsEERss(%"struct.std::__1::atomic.10"* nocapture nonnull align 2 dereferenceable(2) %0, i16* nocapture nonnull align 2 dereferenceable(2) %1, i16 signext %2) {
; CHECK-LABEL: _Z27atomic_cmp_swap_acquire_i16RNSt3__16atomicIsEERss:
; CHECK:       # %bb.0: # %partword.cmpxchg.loop
; CHECK-NEXT:    ld2b.zx %s3, (, %s1)
; CHECK-NEXT:    and %s4, -4, %s0
; CHECK-NEXT:    and %s0, 3, %s0
; CHECK-NEXT:    sla.w.sx %s0, %s0, 3
; CHECK-NEXT:    sla.w.sx %s5, (48)0, %s0
; CHECK-NEXT:    ldl.sx %s6, (, %s4)
; CHECK-NEXT:    and %s2, %s2, (48)0
; CHECK-NEXT:    sla.w.sx %s2, %s2, %s0
; CHECK-NEXT:    sla.w.sx %s3, %s3, %s0
; CHECK-NEXT:    nnd %s5, %s5, %s6
; CHECK-NEXT:    or %s2, %s5, %s2
; CHECK-NEXT:    or %s5, %s5, %s3
; CHECK-NEXT:    cas.w %s2, (%s4), %s5
; CHECK-NEXT:    cmps.w.sx %s4, %s2, %s5
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s3, (63)0, %s4
; CHECK-NEXT:    fencem 2
; CHECK-NEXT:    breq.w %s2, %s5, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    srl %s0, %s2, %s0
; CHECK-NEXT:    st2b %s0, (, %s1)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = getelementptr inbounds %"struct.std::__1::atomic.10", %"struct.std::__1::atomic.10"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %5 = load i16, i16* %1, align 2
  %6 = cmpxchg weak i16* %4, i16 %5, i16 %2 acquire acquire
  %7 = extractvalue { i16, i1 } %6, 1
  br i1 %7, label %10, label %8

8:                                                ; preds = %3
  %9 = extractvalue { i16, i1 } %6, 0
  store i16 %9, i16* %1, align 2
  br label %10

10:                                               ; preds = %3, %8
  %11 = zext i1 %7 to i16
  ret i16 %11
}

; Function Attrs: nofree norecurse nounwind mustprogress
define zeroext i16 @_Z27atomic_cmp_swap_acquire_u16RNSt3__16atomicItEERtt(%"struct.std::__1::atomic.15"* nocapture nonnull align 2 dereferenceable(2) %0, i16* nocapture nonnull align 2 dereferenceable(2) %1, i16 zeroext %2) {
; CHECK-LABEL: _Z27atomic_cmp_swap_acquire_u16RNSt3__16atomicItEERtt:
; CHECK:       # %bb.0: # %partword.cmpxchg.loop
; CHECK-NEXT:    ld2b.zx %s3, (, %s1)
; CHECK-NEXT:    and %s4, -4, %s0
; CHECK-NEXT:    and %s0, 3, %s0
; CHECK-NEXT:    sla.w.sx %s0, %s0, 3
; CHECK-NEXT:    ldl.sx %s5, (, %s4)
; CHECK-NEXT:    sla.w.sx %s6, (48)0, %s0
; CHECK-NEXT:    sla.w.sx %s2, %s2, %s0
; CHECK-NEXT:    sla.w.sx %s3, %s3, %s0
; CHECK-NEXT:    nnd %s5, %s6, %s5
; CHECK-NEXT:    or %s2, %s5, %s2
; CHECK-NEXT:    or %s5, %s5, %s3
; CHECK-NEXT:    cas.w %s2, (%s4), %s5
; CHECK-NEXT:    cmps.w.sx %s4, %s2, %s5
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s3, (63)0, %s4
; CHECK-NEXT:    fencem 2
; CHECK-NEXT:    breq.w %s2, %s5, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    srl %s0, %s2, %s0
; CHECK-NEXT:    st2b %s0, (, %s1)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = getelementptr inbounds %"struct.std::__1::atomic.15", %"struct.std::__1::atomic.15"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %5 = load i16, i16* %1, align 2
  %6 = cmpxchg weak i16* %4, i16 %5, i16 %2 acquire acquire
  %7 = extractvalue { i16, i1 } %6, 1
  br i1 %7, label %10, label %8

8:                                                ; preds = %3
  %9 = extractvalue { i16, i1 } %6, 0
  store i16 %9, i16* %1, align 2
  br label %10

10:                                               ; preds = %3, %8
  %11 = zext i1 %7 to i16
  ret i16 %11
}

; Function Attrs: nofree norecurse nounwind mustprogress
define signext i32 @_Z27atomic_cmp_swap_acquire_i32RNSt3__16atomicIiEERii(%"struct.std::__1::atomic.20"* nocapture nonnull align 4 dereferenceable(4) %0, i32* nocapture nonnull align 4 dereferenceable(4) %1, i32 signext %2) {
; CHECK-LABEL: _Z27atomic_cmp_swap_acquire_i32RNSt3__16atomicIiEERii:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ldl.sx %s3, (, %s1)
; CHECK-NEXT:    cas.w %s2, (%s0), %s3
; CHECK-NEXT:    cmps.w.sx %s4, %s2, %s3
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s0, (63)0, %s4
; CHECK-NEXT:    fencem 2
; CHECK-NEXT:    breq.w %s2, %s3, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    stl %s2, (, %s1)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = getelementptr inbounds %"struct.std::__1::atomic.20", %"struct.std::__1::atomic.20"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %5 = load i32, i32* %1, align 4
  %6 = cmpxchg weak i32* %4, i32 %5, i32 %2 acquire acquire
  %7 = extractvalue { i32, i1 } %6, 1
  br i1 %7, label %10, label %8

8:                                                ; preds = %3
  %9 = extractvalue { i32, i1 } %6, 0
  store i32 %9, i32* %1, align 4
  br label %10

10:                                               ; preds = %3, %8
  %11 = zext i1 %7 to i32
  ret i32 %11
}

; Function Attrs: nofree norecurse nounwind mustprogress
define zeroext i32 @_Z27atomic_cmp_swap_acquire_u32RNSt3__16atomicIjEERjj(%"struct.std::__1::atomic.25"* nocapture nonnull align 4 dereferenceable(4) %0, i32* nocapture nonnull align 4 dereferenceable(4) %1, i32 zeroext %2) {
; CHECK-LABEL: _Z27atomic_cmp_swap_acquire_u32RNSt3__16atomicIjEERjj:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ldl.sx %s3, (, %s1)
; CHECK-NEXT:    cas.w %s2, (%s0), %s3
; CHECK-NEXT:    cmps.w.sx %s4, %s2, %s3
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s0, (63)0, %s4
; CHECK-NEXT:    fencem 2
; CHECK-NEXT:    breq.w %s2, %s3, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    stl %s2, (, %s1)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = getelementptr inbounds %"struct.std::__1::atomic.25", %"struct.std::__1::atomic.25"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %5 = load i32, i32* %1, align 4
  %6 = cmpxchg weak i32* %4, i32 %5, i32 %2 acquire acquire
  %7 = extractvalue { i32, i1 } %6, 1
  br i1 %7, label %10, label %8

8:                                                ; preds = %3
  %9 = extractvalue { i32, i1 } %6, 0
  store i32 %9, i32* %1, align 4
  br label %10

10:                                               ; preds = %3, %8
  %11 = zext i1 %7 to i32
  ret i32 %11
}

; Function Attrs: nofree norecurse nounwind mustprogress
define i64 @_Z27atomic_cmp_swap_acquire_i64RNSt3__16atomicIlEERll(%"struct.std::__1::atomic.30"* nocapture nonnull align 8 dereferenceable(8) %0, i64* nocapture nonnull align 8 dereferenceable(8) %1, i64 %2) {
; CHECK-LABEL: _Z27atomic_cmp_swap_acquire_i64RNSt3__16atomicIlEERll:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld %s3, (, %s1)
; CHECK-NEXT:    cas.l %s2, (%s0), %s3
; CHECK-NEXT:    cmps.l %s4, %s2, %s3
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    cmov.l.eq %s0, (63)0, %s4
; CHECK-NEXT:    fencem 2
; CHECK-NEXT:    breq.l %s2, %s3, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    st %s2, (, %s1)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = getelementptr inbounds %"struct.std::__1::atomic.30", %"struct.std::__1::atomic.30"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %5 = load i64, i64* %1, align 8
  %6 = cmpxchg weak i64* %4, i64 %5, i64 %2 acquire acquire
  %7 = extractvalue { i64, i1 } %6, 1
  br i1 %7, label %10, label %8

8:                                                ; preds = %3
  %9 = extractvalue { i64, i1 } %6, 0
  store i64 %9, i64* %1, align 8
  br label %10

10:                                               ; preds = %3, %8
  %11 = zext i1 %7 to i64
  ret i64 %11
}

; Function Attrs: nofree norecurse nounwind mustprogress
define i64 @_Z27atomic_cmp_swap_acquire_u64RNSt3__16atomicImEERmm(%"struct.std::__1::atomic.35"* nocapture nonnull align 8 dereferenceable(8) %0, i64* nocapture nonnull align 8 dereferenceable(8) %1, i64 %2) {
; CHECK-LABEL: _Z27atomic_cmp_swap_acquire_u64RNSt3__16atomicImEERmm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld %s3, (, %s1)
; CHECK-NEXT:    cas.l %s2, (%s0), %s3
; CHECK-NEXT:    cmps.l %s4, %s2, %s3
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    cmov.l.eq %s0, (63)0, %s4
; CHECK-NEXT:    fencem 2
; CHECK-NEXT:    breq.l %s2, %s3, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    st %s2, (, %s1)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = getelementptr inbounds %"struct.std::__1::atomic.35", %"struct.std::__1::atomic.35"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %5 = load i64, i64* %1, align 8
  %6 = cmpxchg weak i64* %4, i64 %5, i64 %2 acquire acquire
  %7 = extractvalue { i64, i1 } %6, 1
  br i1 %7, label %10, label %8

8:                                                ; preds = %3
  %9 = extractvalue { i64, i1 } %6, 0
  store i64 %9, i64* %1, align 8
  br label %10

10:                                               ; preds = %3, %8
  %11 = zext i1 %7 to i64
  ret i64 %11
}

; Function Attrs: nounwind mustprogress
define i128 @_Z28atomic_cmp_swap_acquire_i128RNSt3__16atomicInEERnn(%"struct.std::__1::atomic.40"* nonnull align 16 dereferenceable(16) %0, i128* nonnull align 16 dereferenceable(16) %1, i128 %2) {
; CHECK-LABEL: _Z28atomic_cmp_swap_acquire_i128RNSt3__16atomicInEERnn:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s6, 0, %s1
; CHECK-NEXT:    or %s1, 0, %s0
; CHECK-NEXT:    st %s3, 248(, %s11)
; CHECK-NEXT:    st %s2, 240(, %s11)
; CHECK-NEXT:    lea %s0, __atomic_compare_exchange@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, __atomic_compare_exchange@hi(, %s0)
; CHECK-NEXT:    lea %s3, 240(, %s11)
; CHECK-NEXT:    or %s0, 16, (0)1
; CHECK-NEXT:    or %s4, 2, (0)1
; CHECK-NEXT:    or %s5, 2, (0)1
; CHECK-NEXT:    or %s2, 0, %s6
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %4 = alloca i128, align 16
  %5 = bitcast i128* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5)
  store i128 %2, i128* %4, align 16, !tbaa !2
  %6 = bitcast %"struct.std::__1::atomic.40"* %0 to i8*
  %7 = bitcast i128* %1 to i8*
  %8 = call zeroext i1 @__atomic_compare_exchange(i64 16, i8* nonnull %6, i8* nonnull %7, i8* nonnull %5, i32 signext 2, i32 signext 2)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5)
  %9 = zext i1 %8 to i128
  ret i128 %9
}

; Function Attrs: nounwind mustprogress
define i128 @_Z28atomic_cmp_swap_acquire_u128RNSt3__16atomicIoEERoo(%"struct.std::__1::atomic.45"* nonnull align 16 dereferenceable(16) %0, i128* nonnull align 16 dereferenceable(16) %1, i128 %2) {
; CHECK-LABEL: _Z28atomic_cmp_swap_acquire_u128RNSt3__16atomicIoEERoo:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s6, 0, %s1
; CHECK-NEXT:    or %s1, 0, %s0
; CHECK-NEXT:    st %s3, 248(, %s11)
; CHECK-NEXT:    st %s2, 240(, %s11)
; CHECK-NEXT:    lea %s0, __atomic_compare_exchange@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, __atomic_compare_exchange@hi(, %s0)
; CHECK-NEXT:    lea %s3, 240(, %s11)
; CHECK-NEXT:    or %s0, 16, (0)1
; CHECK-NEXT:    or %s4, 2, (0)1
; CHECK-NEXT:    or %s5, 2, (0)1
; CHECK-NEXT:    or %s2, 0, %s6
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %4 = alloca i128, align 16
  %5 = bitcast i128* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5)
  store i128 %2, i128* %4, align 16, !tbaa !2
  %6 = bitcast %"struct.std::__1::atomic.45"* %0 to i8*
  %7 = bitcast i128* %1 to i8*
  %8 = call zeroext i1 @__atomic_compare_exchange(i64 16, i8* nonnull %6, i8* nonnull %7, i8* nonnull %5, i32 signext 2, i32 signext 2)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5)
  %9 = zext i1 %8 to i128
  ret i128 %9
}

; Function Attrs: nofree norecurse nounwind mustprogress
define zeroext i1 @_Z26atomic_cmp_swap_seq_cst_i1RNSt3__16atomicIbEERbb(%"struct.std::__1::atomic"* nocapture nonnull align 1 dereferenceable(1) %0, i8* nocapture nonnull align 1 dereferenceable(1) %1, i1 zeroext %2) {
; CHECK-LABEL: _Z26atomic_cmp_swap_seq_cst_i1RNSt3__16atomicIbEERbb:
; CHECK:       # %bb.0: # %partword.cmpxchg.loop
; CHECK-NEXT:    ld1b.zx %s3, (, %s1)
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    and %s4, -4, %s0
; CHECK-NEXT:    and %s0, 3, %s0
; CHECK-NEXT:    sla.w.sx %s0, %s0, 3
; CHECK-NEXT:    ldl.sx %s5, (, %s4)
; CHECK-NEXT:    sla.w.sx %s6, (56)0, %s0
; CHECK-NEXT:    sla.w.sx %s2, %s2, %s0
; CHECK-NEXT:    sla.w.sx %s3, %s3, %s0
; CHECK-NEXT:    nnd %s5, %s6, %s5
; CHECK-NEXT:    or %s2, %s5, %s2
; CHECK-NEXT:    or %s5, %s5, %s3
; CHECK-NEXT:    cas.w %s2, (%s4), %s5
; CHECK-NEXT:    cmps.w.sx %s4, %s2, %s5
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s3, (63)0, %s4
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    breq.w %s2, %s5, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    srl %s0, %s2, %s0
; CHECK-NEXT:    st1b %s0, (, %s1)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = zext i1 %2 to i8
  %5 = getelementptr inbounds %"struct.std::__1::atomic", %"struct.std::__1::atomic"* %0, i64 0, i32 0, i32 0, i32 0, i32 0
  %6 = load i8, i8* %1, align 1
  %7 = cmpxchg weak i8* %5, i8 %6, i8 %4 seq_cst seq_cst
  %8 = extractvalue { i8, i1 } %7, 1
  br i1 %8, label %11, label %9

9:                                                ; preds = %3
  %10 = extractvalue { i8, i1 } %7, 0
  store i8 %10, i8* %1, align 1
  br label %11

11:                                               ; preds = %3, %9
  ret i1 %8
}

; Function Attrs: nofree norecurse nounwind mustprogress
define signext i8 @_Z26atomic_cmp_swap_seq_cst_i8RNSt3__16atomicIcEERcc(%"struct.std::__1::atomic.0"* nocapture nonnull align 1 dereferenceable(1) %0, i8* nocapture nonnull align 1 dereferenceable(1) %1, i8 signext %2) {
; CHECK-LABEL: _Z26atomic_cmp_swap_seq_cst_i8RNSt3__16atomicIcEERcc:
; CHECK:       # %bb.0: # %partword.cmpxchg.loop
; CHECK-NEXT:    ld1b.zx %s3, (, %s1)
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    and %s4, -4, %s0
; CHECK-NEXT:    and %s0, 3, %s0
; CHECK-NEXT:    sla.w.sx %s0, %s0, 3
; CHECK-NEXT:    sla.w.sx %s5, (56)0, %s0
; CHECK-NEXT:    ldl.sx %s6, (, %s4)
; CHECK-NEXT:    and %s2, %s2, (56)0
; CHECK-NEXT:    sla.w.sx %s2, %s2, %s0
; CHECK-NEXT:    sla.w.sx %s3, %s3, %s0
; CHECK-NEXT:    nnd %s5, %s5, %s6
; CHECK-NEXT:    or %s2, %s5, %s2
; CHECK-NEXT:    or %s5, %s5, %s3
; CHECK-NEXT:    cas.w %s2, (%s4), %s5
; CHECK-NEXT:    cmps.w.sx %s4, %s2, %s5
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s3, (63)0, %s4
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    breq.w %s2, %s5, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    srl %s0, %s2, %s0
; CHECK-NEXT:    st1b %s0, (, %s1)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = getelementptr inbounds %"struct.std::__1::atomic.0", %"struct.std::__1::atomic.0"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %5 = load i8, i8* %1, align 1
  %6 = cmpxchg weak i8* %4, i8 %5, i8 %2 seq_cst seq_cst
  %7 = extractvalue { i8, i1 } %6, 1
  br i1 %7, label %10, label %8

8:                                                ; preds = %3
  %9 = extractvalue { i8, i1 } %6, 0
  store i8 %9, i8* %1, align 1
  br label %10

10:                                               ; preds = %3, %8
  %11 = zext i1 %7 to i8
  ret i8 %11
}

; Function Attrs: nofree norecurse nounwind mustprogress
define zeroext i8 @_Z26atomic_cmp_swap_seq_cst_u8RNSt3__16atomicIhEERhh(%"struct.std::__1::atomic.5"* nocapture nonnull align 1 dereferenceable(1) %0, i8* nocapture nonnull align 1 dereferenceable(1) %1, i8 zeroext %2) {
; CHECK-LABEL: _Z26atomic_cmp_swap_seq_cst_u8RNSt3__16atomicIhEERhh:
; CHECK:       # %bb.0: # %partword.cmpxchg.loop
; CHECK-NEXT:    ld1b.zx %s3, (, %s1)
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    and %s4, -4, %s0
; CHECK-NEXT:    and %s0, 3, %s0
; CHECK-NEXT:    sla.w.sx %s0, %s0, 3
; CHECK-NEXT:    ldl.sx %s5, (, %s4)
; CHECK-NEXT:    sla.w.sx %s6, (56)0, %s0
; CHECK-NEXT:    sla.w.sx %s2, %s2, %s0
; CHECK-NEXT:    sla.w.sx %s3, %s3, %s0
; CHECK-NEXT:    nnd %s5, %s6, %s5
; CHECK-NEXT:    or %s2, %s5, %s2
; CHECK-NEXT:    or %s5, %s5, %s3
; CHECK-NEXT:    cas.w %s2, (%s4), %s5
; CHECK-NEXT:    cmps.w.sx %s4, %s2, %s5
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s3, (63)0, %s4
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    breq.w %s2, %s5, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    srl %s0, %s2, %s0
; CHECK-NEXT:    st1b %s0, (, %s1)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = getelementptr inbounds %"struct.std::__1::atomic.5", %"struct.std::__1::atomic.5"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %5 = load i8, i8* %1, align 1
  %6 = cmpxchg weak i8* %4, i8 %5, i8 %2 seq_cst seq_cst
  %7 = extractvalue { i8, i1 } %6, 1
  br i1 %7, label %10, label %8

8:                                                ; preds = %3
  %9 = extractvalue { i8, i1 } %6, 0
  store i8 %9, i8* %1, align 1
  br label %10

10:                                               ; preds = %3, %8
  %11 = zext i1 %7 to i8
  ret i8 %11
}

; Function Attrs: nofree norecurse nounwind mustprogress
define signext i16 @_Z27atomic_cmp_swap_seq_cst_i16RNSt3__16atomicIsEERss(%"struct.std::__1::atomic.10"* nocapture nonnull align 2 dereferenceable(2) %0, i16* nocapture nonnull align 2 dereferenceable(2) %1, i16 signext %2) {
; CHECK-LABEL: _Z27atomic_cmp_swap_seq_cst_i16RNSt3__16atomicIsEERss:
; CHECK:       # %bb.0: # %partword.cmpxchg.loop
; CHECK-NEXT:    ld2b.zx %s3, (, %s1)
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    and %s4, -4, %s0
; CHECK-NEXT:    and %s0, 3, %s0
; CHECK-NEXT:    sla.w.sx %s0, %s0, 3
; CHECK-NEXT:    sla.w.sx %s5, (48)0, %s0
; CHECK-NEXT:    ldl.sx %s6, (, %s4)
; CHECK-NEXT:    and %s2, %s2, (48)0
; CHECK-NEXT:    sla.w.sx %s2, %s2, %s0
; CHECK-NEXT:    sla.w.sx %s3, %s3, %s0
; CHECK-NEXT:    nnd %s5, %s5, %s6
; CHECK-NEXT:    or %s2, %s5, %s2
; CHECK-NEXT:    or %s5, %s5, %s3
; CHECK-NEXT:    cas.w %s2, (%s4), %s5
; CHECK-NEXT:    cmps.w.sx %s4, %s2, %s5
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s3, (63)0, %s4
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    breq.w %s2, %s5, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    srl %s0, %s2, %s0
; CHECK-NEXT:    st2b %s0, (, %s1)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = getelementptr inbounds %"struct.std::__1::atomic.10", %"struct.std::__1::atomic.10"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %5 = load i16, i16* %1, align 2
  %6 = cmpxchg weak i16* %4, i16 %5, i16 %2 seq_cst seq_cst
  %7 = extractvalue { i16, i1 } %6, 1
  br i1 %7, label %10, label %8

8:                                                ; preds = %3
  %9 = extractvalue { i16, i1 } %6, 0
  store i16 %9, i16* %1, align 2
  br label %10

10:                                               ; preds = %3, %8
  %11 = zext i1 %7 to i16
  ret i16 %11
}

; Function Attrs: nofree norecurse nounwind mustprogress
define zeroext i16 @_Z27atomic_cmp_swap_seq_cst_u16RNSt3__16atomicItEERtt(%"struct.std::__1::atomic.15"* nocapture nonnull align 2 dereferenceable(2) %0, i16* nocapture nonnull align 2 dereferenceable(2) %1, i16 zeroext %2) {
; CHECK-LABEL: _Z27atomic_cmp_swap_seq_cst_u16RNSt3__16atomicItEERtt:
; CHECK:       # %bb.0: # %partword.cmpxchg.loop
; CHECK-NEXT:    ld2b.zx %s3, (, %s1)
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    and %s4, -4, %s0
; CHECK-NEXT:    and %s0, 3, %s0
; CHECK-NEXT:    sla.w.sx %s0, %s0, 3
; CHECK-NEXT:    ldl.sx %s5, (, %s4)
; CHECK-NEXT:    sla.w.sx %s6, (48)0, %s0
; CHECK-NEXT:    sla.w.sx %s2, %s2, %s0
; CHECK-NEXT:    sla.w.sx %s3, %s3, %s0
; CHECK-NEXT:    nnd %s5, %s6, %s5
; CHECK-NEXT:    or %s2, %s5, %s2
; CHECK-NEXT:    or %s5, %s5, %s3
; CHECK-NEXT:    cas.w %s2, (%s4), %s5
; CHECK-NEXT:    cmps.w.sx %s4, %s2, %s5
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s3, (63)0, %s4
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    breq.w %s2, %s5, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    srl %s0, %s2, %s0
; CHECK-NEXT:    st2b %s0, (, %s1)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = getelementptr inbounds %"struct.std::__1::atomic.15", %"struct.std::__1::atomic.15"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %5 = load i16, i16* %1, align 2
  %6 = cmpxchg weak i16* %4, i16 %5, i16 %2 seq_cst seq_cst
  %7 = extractvalue { i16, i1 } %6, 1
  br i1 %7, label %10, label %8

8:                                                ; preds = %3
  %9 = extractvalue { i16, i1 } %6, 0
  store i16 %9, i16* %1, align 2
  br label %10

10:                                               ; preds = %3, %8
  %11 = zext i1 %7 to i16
  ret i16 %11
}

; Function Attrs: nofree norecurse nounwind mustprogress
define signext i32 @_Z27atomic_cmp_swap_seq_cst_i32RNSt3__16atomicIiEERii(%"struct.std::__1::atomic.20"* nocapture nonnull align 4 dereferenceable(4) %0, i32* nocapture nonnull align 4 dereferenceable(4) %1, i32 signext %2) {
; CHECK-LABEL: _Z27atomic_cmp_swap_seq_cst_i32RNSt3__16atomicIiEERii:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ldl.sx %s3, (, %s1)
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    cas.w %s2, (%s0), %s3
; CHECK-NEXT:    cmps.w.sx %s4, %s2, %s3
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s0, (63)0, %s4
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    breq.w %s2, %s3, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    stl %s2, (, %s1)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = getelementptr inbounds %"struct.std::__1::atomic.20", %"struct.std::__1::atomic.20"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %5 = load i32, i32* %1, align 4
  %6 = cmpxchg weak i32* %4, i32 %5, i32 %2 seq_cst seq_cst
  %7 = extractvalue { i32, i1 } %6, 1
  br i1 %7, label %10, label %8

8:                                                ; preds = %3
  %9 = extractvalue { i32, i1 } %6, 0
  store i32 %9, i32* %1, align 4
  br label %10

10:                                               ; preds = %3, %8
  %11 = zext i1 %7 to i32
  ret i32 %11
}

; Function Attrs: nofree norecurse nounwind mustprogress
define zeroext i32 @_Z27atomic_cmp_swap_seq_cst_u32RNSt3__16atomicIjEERjj(%"struct.std::__1::atomic.25"* nocapture nonnull align 4 dereferenceable(4) %0, i32* nocapture nonnull align 4 dereferenceable(4) %1, i32 zeroext %2) {
; CHECK-LABEL: _Z27atomic_cmp_swap_seq_cst_u32RNSt3__16atomicIjEERjj:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ldl.sx %s3, (, %s1)
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    cas.w %s2, (%s0), %s3
; CHECK-NEXT:    cmps.w.sx %s4, %s2, %s3
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s0, (63)0, %s4
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    breq.w %s2, %s3, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    stl %s2, (, %s1)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = getelementptr inbounds %"struct.std::__1::atomic.25", %"struct.std::__1::atomic.25"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %5 = load i32, i32* %1, align 4
  %6 = cmpxchg weak i32* %4, i32 %5, i32 %2 seq_cst seq_cst
  %7 = extractvalue { i32, i1 } %6, 1
  br i1 %7, label %10, label %8

8:                                                ; preds = %3
  %9 = extractvalue { i32, i1 } %6, 0
  store i32 %9, i32* %1, align 4
  br label %10

10:                                               ; preds = %3, %8
  %11 = zext i1 %7 to i32
  ret i32 %11
}

; Function Attrs: nofree norecurse nounwind mustprogress
define i64 @_Z27atomic_cmp_swap_seq_cst_i64RNSt3__16atomicIlEERll(%"struct.std::__1::atomic.30"* nocapture nonnull align 8 dereferenceable(8) %0, i64* nocapture nonnull align 8 dereferenceable(8) %1, i64 %2) {
; CHECK-LABEL: _Z27atomic_cmp_swap_seq_cst_i64RNSt3__16atomicIlEERll:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld %s3, (, %s1)
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    cas.l %s2, (%s0), %s3
; CHECK-NEXT:    cmps.l %s4, %s2, %s3
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    cmov.l.eq %s0, (63)0, %s4
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    breq.l %s2, %s3, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    st %s2, (, %s1)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = getelementptr inbounds %"struct.std::__1::atomic.30", %"struct.std::__1::atomic.30"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %5 = load i64, i64* %1, align 8
  %6 = cmpxchg weak i64* %4, i64 %5, i64 %2 seq_cst seq_cst
  %7 = extractvalue { i64, i1 } %6, 1
  br i1 %7, label %10, label %8

8:                                                ; preds = %3
  %9 = extractvalue { i64, i1 } %6, 0
  store i64 %9, i64* %1, align 8
  br label %10

10:                                               ; preds = %3, %8
  %11 = zext i1 %7 to i64
  ret i64 %11
}

; Function Attrs: nofree norecurse nounwind mustprogress
define i64 @_Z27atomic_cmp_swap_seq_cst_u64RNSt3__16atomicImEERmm(%"struct.std::__1::atomic.35"* nocapture nonnull align 8 dereferenceable(8) %0, i64* nocapture nonnull align 8 dereferenceable(8) %1, i64 %2) {
; CHECK-LABEL: _Z27atomic_cmp_swap_seq_cst_u64RNSt3__16atomicImEERmm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld %s3, (, %s1)
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    cas.l %s2, (%s0), %s3
; CHECK-NEXT:    cmps.l %s4, %s2, %s3
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    cmov.l.eq %s0, (63)0, %s4
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    breq.l %s2, %s3, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    st %s2, (, %s1)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = getelementptr inbounds %"struct.std::__1::atomic.35", %"struct.std::__1::atomic.35"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %5 = load i64, i64* %1, align 8
  %6 = cmpxchg weak i64* %4, i64 %5, i64 %2 seq_cst seq_cst
  %7 = extractvalue { i64, i1 } %6, 1
  br i1 %7, label %10, label %8

8:                                                ; preds = %3
  %9 = extractvalue { i64, i1 } %6, 0
  store i64 %9, i64* %1, align 8
  br label %10

10:                                               ; preds = %3, %8
  %11 = zext i1 %7 to i64
  ret i64 %11
}

; Function Attrs: nounwind mustprogress
define i128 @_Z28atomic_cmp_swap_seq_cst_i128RNSt3__16atomicInEERnn(%"struct.std::__1::atomic.40"* nonnull align 16 dereferenceable(16) %0, i128* nonnull align 16 dereferenceable(16) %1, i128 %2) {
; CHECK-LABEL: _Z28atomic_cmp_swap_seq_cst_i128RNSt3__16atomicInEERnn:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s6, 0, %s1
; CHECK-NEXT:    or %s1, 0, %s0
; CHECK-NEXT:    st %s3, 248(, %s11)
; CHECK-NEXT:    st %s2, 240(, %s11)
; CHECK-NEXT:    lea %s0, __atomic_compare_exchange@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, __atomic_compare_exchange@hi(, %s0)
; CHECK-NEXT:    lea %s3, 240(, %s11)
; CHECK-NEXT:    or %s0, 16, (0)1
; CHECK-NEXT:    or %s4, 5, (0)1
; CHECK-NEXT:    or %s5, 5, (0)1
; CHECK-NEXT:    or %s2, 0, %s6
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %4 = alloca i128, align 16
  %5 = bitcast i128* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5)
  store i128 %2, i128* %4, align 16, !tbaa !2
  %6 = bitcast %"struct.std::__1::atomic.40"* %0 to i8*
  %7 = bitcast i128* %1 to i8*
  %8 = call zeroext i1 @__atomic_compare_exchange(i64 16, i8* nonnull %6, i8* nonnull %7, i8* nonnull %5, i32 signext 5, i32 signext 5)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5)
  %9 = zext i1 %8 to i128
  ret i128 %9
}

; Function Attrs: nounwind mustprogress
define i128 @_Z28atomic_cmp_swap_seq_cst_u128RNSt3__16atomicIoEERoo(%"struct.std::__1::atomic.45"* nonnull align 16 dereferenceable(16) %0, i128* nonnull align 16 dereferenceable(16) %1, i128 %2) {
; CHECK-LABEL: _Z28atomic_cmp_swap_seq_cst_u128RNSt3__16atomicIoEERoo:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s6, 0, %s1
; CHECK-NEXT:    or %s1, 0, %s0
; CHECK-NEXT:    st %s3, 248(, %s11)
; CHECK-NEXT:    st %s2, 240(, %s11)
; CHECK-NEXT:    lea %s0, __atomic_compare_exchange@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, __atomic_compare_exchange@hi(, %s0)
; CHECK-NEXT:    lea %s3, 240(, %s11)
; CHECK-NEXT:    or %s0, 16, (0)1
; CHECK-NEXT:    or %s4, 5, (0)1
; CHECK-NEXT:    or %s5, 5, (0)1
; CHECK-NEXT:    or %s2, 0, %s6
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %4 = alloca i128, align 16
  %5 = bitcast i128* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5)
  store i128 %2, i128* %4, align 16, !tbaa !2
  %6 = bitcast %"struct.std::__1::atomic.45"* %0 to i8*
  %7 = bitcast i128* %1 to i8*
  %8 = call zeroext i1 @__atomic_compare_exchange(i64 16, i8* nonnull %6, i8* nonnull %7, i8* nonnull %5, i32 signext 5, i32 signext 5)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5)
  %9 = zext i1 %8 to i128
  ret i128 %9
}

; Function Attrs: nofree nounwind mustprogress
define zeroext i1 @_Z30atomic_cmp_swap_relaxed_stk_i1Rbb(i8* nocapture nonnull align 1 dereferenceable(1) %0, i1 zeroext %1) {
; CHECK-LABEL: _Z30atomic_cmp_swap_relaxed_stk_i1Rbb:
; CHECK:       .LBB{{[0-9]+}}_4: # %partword.cmpxchg.loop
; CHECK-NEXT:    ld1b.zx %s2, (, %s0)
; CHECK-NEXT:    ldl.zx %s3, 8(, %s11)
; CHECK-NEXT:    lea %s4, -256
; CHECK-NEXT:    and %s4, %s4, (32)0
; CHECK-NEXT:    and %s3, %s3, %s4
; CHECK-NEXT:    or %s1, %s3, %s1
; CHECK-NEXT:    or %s3, %s3, %s2
; CHECK-NEXT:    cas.w %s1, 8(%s11), %s3
; CHECK-NEXT:    cmps.w.sx %s4, %s1, %s3
; CHECK-NEXT:    or %s2, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s2, (63)0, %s4
; CHECK-NEXT:    breq.w %s1, %s3, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    st1b %s1, (, %s0)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s2, (0)1
; CHECK-NEXT:    adds.l %s11, 16, %s11
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = alloca %"struct.std::__1::atomic", align 1
  %4 = getelementptr inbounds %"struct.std::__1::atomic", %"struct.std::__1::atomic"* %3, i64 0, i32 0, i32 0, i32 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %4)
  %5 = zext i1 %1 to i8
  %6 = load i8, i8* %0, align 1
  %7 = cmpxchg weak volatile i8* %4, i8 %6, i8 %5 monotonic monotonic
  %8 = extractvalue { i8, i1 } %7, 1
  br i1 %8, label %11, label %9

9:                                                ; preds = %2
  %10 = extractvalue { i8, i1 } %7, 0
  store i8 %10, i8* %0, align 1
  br label %11

11:                                               ; preds = %2, %9
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %4)
  ret i1 %8
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)

; Function Attrs: nofree nounwind mustprogress
define signext i8 @_Z30atomic_cmp_swap_relaxed_stk_i8Rcc(i8* nocapture nonnull align 1 dereferenceable(1) %0, i8 signext %1) {
; CHECK-LABEL: _Z30atomic_cmp_swap_relaxed_stk_i8Rcc:
; CHECK:       .LBB{{[0-9]+}}_4: # %partword.cmpxchg.loop
; CHECK-NEXT:    ld1b.zx %s2, (, %s0)
; CHECK-NEXT:    ldl.zx %s3, 8(, %s11)
; CHECK-NEXT:    and %s1, %s1, (56)0
; CHECK-NEXT:    lea %s4, -256
; CHECK-NEXT:    and %s4, %s4, (32)0
; CHECK-NEXT:    and %s3, %s3, %s4
; CHECK-NEXT:    or %s1, %s3, %s1
; CHECK-NEXT:    or %s3, %s3, %s2
; CHECK-NEXT:    cas.w %s1, 8(%s11), %s3
; CHECK-NEXT:    cmps.w.sx %s4, %s1, %s3
; CHECK-NEXT:    or %s2, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s2, (63)0, %s4
; CHECK-NEXT:    breq.w %s1, %s3, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    st1b %s1, (, %s0)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s2, (0)1
; CHECK-NEXT:    adds.l %s11, 16, %s11
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = alloca %"struct.std::__1::atomic.0", align 1
  %4 = getelementptr inbounds %"struct.std::__1::atomic.0", %"struct.std::__1::atomic.0"* %3, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %4)
  %5 = load i8, i8* %0, align 1
  %6 = cmpxchg weak volatile i8* %4, i8 %5, i8 %1 monotonic monotonic
  %7 = extractvalue { i8, i1 } %6, 1
  br i1 %7, label %10, label %8

8:                                                ; preds = %2
  %9 = extractvalue { i8, i1 } %6, 0
  store i8 %9, i8* %0, align 1
  br label %10

10:                                               ; preds = %2, %8
  %11 = zext i1 %7 to i8
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %4)
  ret i8 %11
}

; Function Attrs: nofree nounwind mustprogress
define zeroext i8 @_Z30atomic_cmp_swap_relaxed_stk_u8Rhh(i8* nocapture nonnull align 1 dereferenceable(1) %0, i8 zeroext %1) {
; CHECK-LABEL: _Z30atomic_cmp_swap_relaxed_stk_u8Rhh:
; CHECK:       .LBB{{[0-9]+}}_4: # %partword.cmpxchg.loop
; CHECK-NEXT:    ld1b.zx %s2, (, %s0)
; CHECK-NEXT:    ldl.zx %s3, 8(, %s11)
; CHECK-NEXT:    lea %s4, -256
; CHECK-NEXT:    and %s4, %s4, (32)0
; CHECK-NEXT:    and %s3, %s3, %s4
; CHECK-NEXT:    or %s1, %s3, %s1
; CHECK-NEXT:    or %s3, %s3, %s2
; CHECK-NEXT:    cas.w %s1, 8(%s11), %s3
; CHECK-NEXT:    cmps.w.sx %s4, %s1, %s3
; CHECK-NEXT:    or %s2, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s2, (63)0, %s4
; CHECK-NEXT:    breq.w %s1, %s3, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    st1b %s1, (, %s0)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s2, (0)1
; CHECK-NEXT:    adds.l %s11, 16, %s11
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = alloca %"struct.std::__1::atomic.5", align 1
  %4 = getelementptr inbounds %"struct.std::__1::atomic.5", %"struct.std::__1::atomic.5"* %3, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %4)
  %5 = load i8, i8* %0, align 1
  %6 = cmpxchg weak volatile i8* %4, i8 %5, i8 %1 monotonic monotonic
  %7 = extractvalue { i8, i1 } %6, 1
  br i1 %7, label %10, label %8

8:                                                ; preds = %2
  %9 = extractvalue { i8, i1 } %6, 0
  store i8 %9, i8* %0, align 1
  br label %10

10:                                               ; preds = %2, %8
  %11 = zext i1 %7 to i8
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %4)
  ret i8 %11
}

; Function Attrs: nofree nounwind mustprogress
define signext i16 @_Z31atomic_cmp_swap_relaxed_stk_i16Rss(i16* nocapture nonnull align 2 dereferenceable(2) %0, i16 signext %1) {
; CHECK-LABEL: _Z31atomic_cmp_swap_relaxed_stk_i16Rss:
; CHECK:       .LBB{{[0-9]+}}_4: # %partword.cmpxchg.loop
; CHECK-NEXT:    ld2b.zx %s2, (, %s0)
; CHECK-NEXT:    ldl.zx %s3, 8(, %s11)
; CHECK-NEXT:    and %s1, %s1, (48)0
; CHECK-NEXT:    lea %s4, -65536
; CHECK-NEXT:    and %s4, %s4, (32)0
; CHECK-NEXT:    and %s3, %s3, %s4
; CHECK-NEXT:    or %s1, %s3, %s1
; CHECK-NEXT:    or %s3, %s3, %s2
; CHECK-NEXT:    cas.w %s1, 8(%s11), %s3
; CHECK-NEXT:    cmps.w.sx %s4, %s1, %s3
; CHECK-NEXT:    or %s2, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s2, (63)0, %s4
; CHECK-NEXT:    breq.w %s1, %s3, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    st2b %s1, (, %s0)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s2, (0)1
; CHECK-NEXT:    adds.l %s11, 16, %s11
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = alloca %"struct.std::__1::atomic.10", align 2
  %4 = bitcast %"struct.std::__1::atomic.10"* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %4)
  %5 = getelementptr inbounds %"struct.std::__1::atomic.10", %"struct.std::__1::atomic.10"* %3, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %6 = load i16, i16* %0, align 2
  %7 = cmpxchg weak volatile i16* %5, i16 %6, i16 %1 monotonic monotonic
  %8 = extractvalue { i16, i1 } %7, 1
  br i1 %8, label %11, label %9

9:                                                ; preds = %2
  %10 = extractvalue { i16, i1 } %7, 0
  store i16 %10, i16* %0, align 2
  br label %11

11:                                               ; preds = %2, %9
  %12 = zext i1 %8 to i16
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %4)
  ret i16 %12
}

; Function Attrs: nofree nounwind mustprogress
define zeroext i16 @_Z31atomic_cmp_swap_relaxed_stk_u16Rtt(i16* nocapture nonnull align 2 dereferenceable(2) %0, i16 zeroext %1) {
; CHECK-LABEL: _Z31atomic_cmp_swap_relaxed_stk_u16Rtt:
; CHECK:       .LBB{{[0-9]+}}_4: # %partword.cmpxchg.loop
; CHECK-NEXT:    ld2b.zx %s2, (, %s0)
; CHECK-NEXT:    ldl.zx %s3, 8(, %s11)
; CHECK-NEXT:    lea %s4, -65536
; CHECK-NEXT:    and %s4, %s4, (32)0
; CHECK-NEXT:    and %s3, %s3, %s4
; CHECK-NEXT:    or %s1, %s3, %s1
; CHECK-NEXT:    or %s3, %s3, %s2
; CHECK-NEXT:    cas.w %s1, 8(%s11), %s3
; CHECK-NEXT:    cmps.w.sx %s4, %s1, %s3
; CHECK-NEXT:    or %s2, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s2, (63)0, %s4
; CHECK-NEXT:    breq.w %s1, %s3, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    st2b %s1, (, %s0)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s2, (0)1
; CHECK-NEXT:    adds.l %s11, 16, %s11
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = alloca %"struct.std::__1::atomic.15", align 2
  %4 = bitcast %"struct.std::__1::atomic.15"* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %4)
  %5 = getelementptr inbounds %"struct.std::__1::atomic.15", %"struct.std::__1::atomic.15"* %3, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %6 = load i16, i16* %0, align 2
  %7 = cmpxchg weak volatile i16* %5, i16 %6, i16 %1 monotonic monotonic
  %8 = extractvalue { i16, i1 } %7, 1
  br i1 %8, label %11, label %9

9:                                                ; preds = %2
  %10 = extractvalue { i16, i1 } %7, 0
  store i16 %10, i16* %0, align 2
  br label %11

11:                                               ; preds = %2, %9
  %12 = zext i1 %8 to i16
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %4)
  ret i16 %12
}

; Function Attrs: nofree nounwind mustprogress
define signext i32 @_Z31atomic_cmp_swap_relaxed_stk_i32Rii(i32* nocapture nonnull align 4 dereferenceable(4) %0, i32 signext %1) {
; CHECK-LABEL: _Z31atomic_cmp_swap_relaxed_stk_i32Rii:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    ldl.sx %s3, (, %s0)
; CHECK-NEXT:    cas.w %s1, 8(%s11), %s3
; CHECK-NEXT:    cmps.w.sx %s4, %s1, %s3
; CHECK-NEXT:    or %s2, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s2, (63)0, %s4
; CHECK-NEXT:    breq.w %s1, %s3, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    stl %s1, (, %s0)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s2, (0)1
; CHECK-NEXT:    adds.l %s11, 16, %s11
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = alloca %"struct.std::__1::atomic.20", align 4
  %4 = bitcast %"struct.std::__1::atomic.20"* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %4)
  %5 = getelementptr inbounds %"struct.std::__1::atomic.20", %"struct.std::__1::atomic.20"* %3, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %6 = load i32, i32* %0, align 4
  %7 = cmpxchg weak volatile i32* %5, i32 %6, i32 %1 monotonic monotonic
  %8 = extractvalue { i32, i1 } %7, 1
  br i1 %8, label %11, label %9

9:                                                ; preds = %2
  %10 = extractvalue { i32, i1 } %7, 0
  store i32 %10, i32* %0, align 4
  br label %11

11:                                               ; preds = %2, %9
  %12 = zext i1 %8 to i32
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %4)
  ret i32 %12
}

; Function Attrs: nofree nounwind mustprogress
define zeroext i32 @_Z31atomic_cmp_swap_relaxed_stk_u32Rjj(i32* nocapture nonnull align 4 dereferenceable(4) %0, i32 zeroext %1) {
; CHECK-LABEL: _Z31atomic_cmp_swap_relaxed_stk_u32Rjj:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    ldl.sx %s3, (, %s0)
; CHECK-NEXT:    cas.w %s1, 8(%s11), %s3
; CHECK-NEXT:    cmps.w.sx %s4, %s1, %s3
; CHECK-NEXT:    or %s2, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s2, (63)0, %s4
; CHECK-NEXT:    breq.w %s1, %s3, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    stl %s1, (, %s0)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s2, (0)1
; CHECK-NEXT:    adds.l %s11, 16, %s11
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = alloca %"struct.std::__1::atomic.25", align 4
  %4 = bitcast %"struct.std::__1::atomic.25"* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %4)
  %5 = getelementptr inbounds %"struct.std::__1::atomic.25", %"struct.std::__1::atomic.25"* %3, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %6 = load i32, i32* %0, align 4
  %7 = cmpxchg weak volatile i32* %5, i32 %6, i32 %1 monotonic monotonic
  %8 = extractvalue { i32, i1 } %7, 1
  br i1 %8, label %11, label %9

9:                                                ; preds = %2
  %10 = extractvalue { i32, i1 } %7, 0
  store i32 %10, i32* %0, align 4
  br label %11

11:                                               ; preds = %2, %9
  %12 = zext i1 %8 to i32
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %4)
  ret i32 %12
}

; Function Attrs: nofree nounwind mustprogress
define i64 @_Z31atomic_cmp_swap_relaxed_stk_i64Rll(i64* nocapture nonnull align 8 dereferenceable(8) %0, i64 %1) {
; CHECK-LABEL: _Z31atomic_cmp_swap_relaxed_stk_i64Rll:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    ld %s3, (, %s0)
; CHECK-NEXT:    cas.l %s1, 8(%s11), %s3
; CHECK-NEXT:    cmps.l %s4, %s1, %s3
; CHECK-NEXT:    or %s2, 0, (0)1
; CHECK-NEXT:    cmov.l.eq %s2, (63)0, %s4
; CHECK-NEXT:    breq.l %s1, %s3, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    st %s1, (, %s0)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s2, (0)1
; CHECK-NEXT:    adds.l %s11, 16, %s11
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = alloca %"struct.std::__1::atomic.30", align 8
  %4 = bitcast %"struct.std::__1::atomic.30"* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %4)
  %5 = getelementptr inbounds %"struct.std::__1::atomic.30", %"struct.std::__1::atomic.30"* %3, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %6 = load i64, i64* %0, align 8
  %7 = cmpxchg weak volatile i64* %5, i64 %6, i64 %1 monotonic monotonic
  %8 = extractvalue { i64, i1 } %7, 1
  br i1 %8, label %11, label %9

9:                                                ; preds = %2
  %10 = extractvalue { i64, i1 } %7, 0
  store i64 %10, i64* %0, align 8
  br label %11

11:                                               ; preds = %2, %9
  %12 = zext i1 %8 to i64
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %4)
  ret i64 %12
}

; Function Attrs: nofree nounwind mustprogress
define i64 @_Z31atomic_cmp_swap_relaxed_stk_u64Rmm(i64* nocapture nonnull align 8 dereferenceable(8) %0, i64 %1) {
; CHECK-LABEL: _Z31atomic_cmp_swap_relaxed_stk_u64Rmm:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    ld %s3, (, %s0)
; CHECK-NEXT:    cas.l %s1, 8(%s11), %s3
; CHECK-NEXT:    cmps.l %s4, %s1, %s3
; CHECK-NEXT:    or %s2, 0, (0)1
; CHECK-NEXT:    cmov.l.eq %s2, (63)0, %s4
; CHECK-NEXT:    breq.l %s1, %s3, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    st %s1, (, %s0)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s2, (0)1
; CHECK-NEXT:    adds.l %s11, 16, %s11
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = alloca %"struct.std::__1::atomic.35", align 8
  %4 = bitcast %"struct.std::__1::atomic.35"* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %4)
  %5 = getelementptr inbounds %"struct.std::__1::atomic.35", %"struct.std::__1::atomic.35"* %3, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %6 = load i64, i64* %0, align 8
  %7 = cmpxchg weak volatile i64* %5, i64 %6, i64 %1 monotonic monotonic
  %8 = extractvalue { i64, i1 } %7, 1
  br i1 %8, label %11, label %9

9:                                                ; preds = %2
  %10 = extractvalue { i64, i1 } %7, 0
  store i64 %10, i64* %0, align 8
  br label %11

11:                                               ; preds = %2, %9
  %12 = zext i1 %8 to i64
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %4)
  ret i64 %12
}

; Function Attrs: nounwind mustprogress
define i128 @_Z32atomic_cmp_swap_relaxed_stk_i128Rnn(i128* nonnull align 16 dereferenceable(16) %0, i128 %1) {
; CHECK-LABEL: _Z32atomic_cmp_swap_relaxed_stk_i128Rnn:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s6, 0, %s0
; CHECK-NEXT:    st %s2, 264(, %s11)
; CHECK-NEXT:    st %s1, 256(, %s11)
; CHECK-NEXT:    lea %s0, __atomic_compare_exchange@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, __atomic_compare_exchange@hi(, %s0)
; CHECK-NEXT:    lea %s1, 240(, %s11)
; CHECK-NEXT:    lea %s3, 256(, %s11)
; CHECK-NEXT:    or %s0, 16, (0)1
; CHECK-NEXT:    or %s4, 0, (0)1
; CHECK-NEXT:    or %s5, 0, (0)1
; CHECK-NEXT:    or %s2, 0, %s6
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = alloca i128, align 16
  %4 = alloca %"struct.std::__1::atomic.40", align 16
  %5 = bitcast %"struct.std::__1::atomic.40"* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5)
  %6 = bitcast i128* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %6)
  store i128 %1, i128* %3, align 16, !tbaa !2
  %7 = bitcast i128* %0 to i8*
  %8 = call zeroext i1 @__atomic_compare_exchange(i64 16, i8* nonnull %5, i8* nonnull %7, i8* nonnull %6, i32 signext 0, i32 signext 0)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %6)
  %9 = zext i1 %8 to i128
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5)
  ret i128 %9
}

; Function Attrs: nounwind mustprogress
define i128 @_Z32atomic_cmp_swap_relaxed_stk_u128Roo(i128* nonnull align 16 dereferenceable(16) %0, i128 %1) {
; CHECK-LABEL: _Z32atomic_cmp_swap_relaxed_stk_u128Roo:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s6, 0, %s0
; CHECK-NEXT:    st %s2, 264(, %s11)
; CHECK-NEXT:    st %s1, 256(, %s11)
; CHECK-NEXT:    lea %s0, __atomic_compare_exchange@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, __atomic_compare_exchange@hi(, %s0)
; CHECK-NEXT:    lea %s1, 240(, %s11)
; CHECK-NEXT:    lea %s3, 256(, %s11)
; CHECK-NEXT:    or %s0, 16, (0)1
; CHECK-NEXT:    or %s4, 0, (0)1
; CHECK-NEXT:    or %s5, 0, (0)1
; CHECK-NEXT:    or %s2, 0, %s6
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = alloca i128, align 16
  %4 = alloca %"struct.std::__1::atomic.45", align 16
  %5 = bitcast %"struct.std::__1::atomic.45"* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5)
  %6 = bitcast i128* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %6)
  store i128 %1, i128* %3, align 16, !tbaa !2
  %7 = bitcast i128* %0 to i8*
  %8 = call zeroext i1 @__atomic_compare_exchange(i64 16, i8* nonnull %5, i8* nonnull %7, i8* nonnull %6, i32 signext 0, i32 signext 0)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %6)
  %9 = zext i1 %8 to i128
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5)
  ret i128 %9
}

; Function Attrs: nofree norecurse nounwind mustprogress
define zeroext i1 @_Z29atomic_cmp_swap_relaxed_gv_i1Rbb(i8* nocapture nonnull align 1 dereferenceable(1) %0, i1 zeroext %1) {
; CHECK-LABEL: _Z29atomic_cmp_swap_relaxed_gv_i1Rbb:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s2, %s1, (32)0
; CHECK-NEXT:    lea %s1, gv_i1@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s1, gv_i1@hi(, %s1)
; CHECK-NEXT:    and %s1, -4, %s1
; CHECK-NEXT:    ldl.zx %s4, (, %s1)
; CHECK-NEXT:    ld1b.zx %s3, (, %s0)
; CHECK-NEXT:    lea %s5, -256
; CHECK-NEXT:    and %s5, %s5, (32)0
; CHECK-NEXT:    and %s4, %s4, %s5
; CHECK-NEXT:    and %s4, %s4, (32)0
; CHECK-NEXT:    or %s2, %s4, %s2
; CHECK-NEXT:    or %s3, %s4, %s3
; CHECK-NEXT:    cas.w %s2, (%s1), %s3
; CHECK-NEXT:    cmps.w.sx %s3, %s2, %s3
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s1, (63)0, %s3
; CHECK-NEXT:    brne.w 0, %s1, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    st1b %s2, (, %s0)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = zext i1 %1 to i8
  %4 = load i8, i8* %0, align 1
  %5 = cmpxchg weak i8* getelementptr inbounds (%"struct.std::__1::atomic", %"struct.std::__1::atomic"* @gv_i1, i64 0, i32 0, i32 0, i32 0, i32 0), i8 %4, i8 %3 monotonic monotonic
  %6 = extractvalue { i8, i1 } %5, 1
  br i1 %6, label %9, label %7

7:                                                ; preds = %2
  %8 = extractvalue { i8, i1 } %5, 0
  store i8 %8, i8* %0, align 1
  br label %9

9:                                                ; preds = %2, %7
  ret i1 %6
}

; Function Attrs: nofree norecurse nounwind mustprogress
define signext i8 @_Z29atomic_cmp_swap_relaxed_gv_i8Rcc(i8* nocapture nonnull align 1 dereferenceable(1) %0, i8 signext %1) {
; CHECK-LABEL: _Z29atomic_cmp_swap_relaxed_gv_i8Rcc:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld1b.zx %s2, (, %s0)
; CHECK-NEXT:    and %s3, %s1, (56)0
; CHECK-NEXT:    lea %s1, gv_i8@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s1, gv_i8@hi(, %s1)
; CHECK-NEXT:    and %s1, -4, %s1
; CHECK-NEXT:    ldl.zx %s4, (, %s1)
; CHECK-NEXT:    and %s3, %s3, (32)0
; CHECK-NEXT:    lea %s5, -256
; CHECK-NEXT:    and %s5, %s5, (32)0
; CHECK-NEXT:    and %s4, %s4, %s5
; CHECK-NEXT:    and %s4, %s4, (32)0
; CHECK-NEXT:    or %s3, %s4, %s3
; CHECK-NEXT:    or %s2, %s4, %s2
; CHECK-NEXT:    cas.w %s3, (%s1), %s2
; CHECK-NEXT:    cmps.w.sx %s2, %s3, %s2
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s1, (63)0, %s2
; CHECK-NEXT:    brne.w 0, %s1, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    st1b %s3, (, %s0)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = load i8, i8* %0, align 1
  %4 = cmpxchg weak i8* getelementptr inbounds (%"struct.std::__1::atomic.0", %"struct.std::__1::atomic.0"* @gv_i8, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0), i8 %3, i8 %1 monotonic monotonic
  %5 = extractvalue { i8, i1 } %4, 1
  br i1 %5, label %8, label %6

6:                                                ; preds = %2
  %7 = extractvalue { i8, i1 } %4, 0
  store i8 %7, i8* %0, align 1
  br label %8

8:                                                ; preds = %2, %6
  %9 = zext i1 %5 to i8
  ret i8 %9
}

; Function Attrs: nofree norecurse nounwind mustprogress
define zeroext i8 @_Z29atomic_cmp_swap_relaxed_gv_u8Rhh(i8* nocapture nonnull align 1 dereferenceable(1) %0, i8 zeroext %1) {
; CHECK-LABEL: _Z29atomic_cmp_swap_relaxed_gv_u8Rhh:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s2, %s1, (32)0
; CHECK-NEXT:    lea %s1, gv_u8@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s1, gv_u8@hi(, %s1)
; CHECK-NEXT:    and %s1, -4, %s1
; CHECK-NEXT:    ldl.zx %s4, (, %s1)
; CHECK-NEXT:    ld1b.zx %s3, (, %s0)
; CHECK-NEXT:    lea %s5, -256
; CHECK-NEXT:    and %s5, %s5, (32)0
; CHECK-NEXT:    and %s4, %s4, %s5
; CHECK-NEXT:    and %s4, %s4, (32)0
; CHECK-NEXT:    or %s2, %s4, %s2
; CHECK-NEXT:    or %s3, %s4, %s3
; CHECK-NEXT:    cas.w %s2, (%s1), %s3
; CHECK-NEXT:    cmps.w.sx %s3, %s2, %s3
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s1, (63)0, %s3
; CHECK-NEXT:    brne.w 0, %s1, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    st1b %s2, (, %s0)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = load i8, i8* %0, align 1
  %4 = cmpxchg weak i8* getelementptr inbounds (%"struct.std::__1::atomic.5", %"struct.std::__1::atomic.5"* @gv_u8, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0), i8 %3, i8 %1 monotonic monotonic
  %5 = extractvalue { i8, i1 } %4, 1
  br i1 %5, label %8, label %6

6:                                                ; preds = %2
  %7 = extractvalue { i8, i1 } %4, 0
  store i8 %7, i8* %0, align 1
  br label %8

8:                                                ; preds = %2, %6
  %9 = zext i1 %5 to i8
  ret i8 %9
}

; Function Attrs: nofree norecurse nounwind mustprogress
define signext i16 @_Z30atomic_cmp_swap_relaxed_gv_i16Rss(i16* nocapture nonnull align 2 dereferenceable(2) %0, i16 signext %1) {
; CHECK-LABEL: _Z30atomic_cmp_swap_relaxed_gv_i16Rss:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, gv_i16@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s2, gv_i16@hi(, %s2)
; CHECK-NEXT:    and %s2, -4, %s2
; CHECK-NEXT:    ld2b.zx %s4, 2(, %s2)
; CHECK-NEXT:    ld2b.zx %s3, (, %s0)
; CHECK-NEXT:    and %s1, %s1, (48)0
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    sla.w.sx %s4, %s4, 16
; CHECK-NEXT:    or %s1, %s4, %s1
; CHECK-NEXT:    or %s3, %s4, %s3
; CHECK-NEXT:    cas.w %s1, (%s2), %s3
; CHECK-NEXT:    cmps.w.sx %s3, %s1, %s3
; CHECK-NEXT:    or %s2, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s2, (63)0, %s3
; CHECK-NEXT:    brne.w 0, %s2, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    st2b %s1, (, %s0)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s2, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = load i16, i16* %0, align 2
  %4 = cmpxchg weak i16* getelementptr inbounds (%"struct.std::__1::atomic.10", %"struct.std::__1::atomic.10"* @gv_i16, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0), i16 %3, i16 %1 monotonic monotonic
  %5 = extractvalue { i16, i1 } %4, 1
  br i1 %5, label %8, label %6

6:                                                ; preds = %2
  %7 = extractvalue { i16, i1 } %4, 0
  store i16 %7, i16* %0, align 2
  br label %8

8:                                                ; preds = %2, %6
  %9 = zext i1 %5 to i16
  ret i16 %9
}

; Function Attrs: nofree norecurse nounwind mustprogress
define zeroext i16 @_Z30atomic_cmp_swap_relaxed_gv_u16Rtt(i16* nocapture nonnull align 2 dereferenceable(2) %0, i16 zeroext %1) {
; CHECK-LABEL: _Z30atomic_cmp_swap_relaxed_gv_u16Rtt:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, gv_u16@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s2, gv_u16@hi(, %s2)
; CHECK-NEXT:    and %s2, -4, %s2
; CHECK-NEXT:    ld2b.zx %s4, 2(, %s2)
; CHECK-NEXT:    ld2b.zx %s3, (, %s0)
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    sla.w.sx %s4, %s4, 16
; CHECK-NEXT:    or %s1, %s4, %s1
; CHECK-NEXT:    or %s3, %s4, %s3
; CHECK-NEXT:    cas.w %s1, (%s2), %s3
; CHECK-NEXT:    cmps.w.sx %s3, %s1, %s3
; CHECK-NEXT:    or %s2, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s2, (63)0, %s3
; CHECK-NEXT:    brne.w 0, %s2, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    st2b %s1, (, %s0)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s2, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = load i16, i16* %0, align 2
  %4 = cmpxchg weak i16* getelementptr inbounds (%"struct.std::__1::atomic.15", %"struct.std::__1::atomic.15"* @gv_u16, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0), i16 %3, i16 %1 monotonic monotonic
  %5 = extractvalue { i16, i1 } %4, 1
  br i1 %5, label %8, label %6

6:                                                ; preds = %2
  %7 = extractvalue { i16, i1 } %4, 0
  store i16 %7, i16* %0, align 2
  br label %8

8:                                                ; preds = %2, %6
  %9 = zext i1 %5 to i16
  ret i16 %9
}

; Function Attrs: nofree norecurse nounwind mustprogress
define signext i32 @_Z30atomic_cmp_swap_relaxed_gv_i32Rii(i32* nocapture nonnull align 4 dereferenceable(4) %0, i32 signext %1) {
; CHECK-LABEL: _Z30atomic_cmp_swap_relaxed_gv_i32Rii:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ldl.sx %s3, (, %s0)
; CHECK-NEXT:    lea %s2, gv_i32@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s2, gv_i32@hi(, %s2)
; CHECK-NEXT:    cas.w %s1, (%s2), %s3
; CHECK-NEXT:    cmps.w.sx %s4, %s1, %s3
; CHECK-NEXT:    or %s2, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s2, (63)0, %s4
; CHECK-NEXT:    breq.w %s1, %s3, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    stl %s1, (, %s0)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s2, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = load i32, i32* %0, align 4
  %4 = cmpxchg weak i32* getelementptr inbounds (%"struct.std::__1::atomic.20", %"struct.std::__1::atomic.20"* @gv_i32, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0), i32 %3, i32 %1 monotonic monotonic
  %5 = extractvalue { i32, i1 } %4, 1
  br i1 %5, label %8, label %6

6:                                                ; preds = %2
  %7 = extractvalue { i32, i1 } %4, 0
  store i32 %7, i32* %0, align 4
  br label %8

8:                                                ; preds = %2, %6
  %9 = zext i1 %5 to i32
  ret i32 %9
}

; Function Attrs: nofree norecurse nounwind mustprogress
define zeroext i32 @_Z30atomic_cmp_swap_relaxed_gv_u32Rjj(i32* nocapture nonnull align 4 dereferenceable(4) %0, i32 zeroext %1) {
; CHECK-LABEL: _Z30atomic_cmp_swap_relaxed_gv_u32Rjj:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ldl.sx %s3, (, %s0)
; CHECK-NEXT:    lea %s2, gv_u32@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s2, gv_u32@hi(, %s2)
; CHECK-NEXT:    cas.w %s1, (%s2), %s3
; CHECK-NEXT:    cmps.w.sx %s4, %s1, %s3
; CHECK-NEXT:    or %s2, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s2, (63)0, %s4
; CHECK-NEXT:    breq.w %s1, %s3, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    stl %s1, (, %s0)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s2, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = load i32, i32* %0, align 4
  %4 = cmpxchg weak i32* getelementptr inbounds (%"struct.std::__1::atomic.25", %"struct.std::__1::atomic.25"* @gv_u32, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0), i32 %3, i32 %1 monotonic monotonic
  %5 = extractvalue { i32, i1 } %4, 1
  br i1 %5, label %8, label %6

6:                                                ; preds = %2
  %7 = extractvalue { i32, i1 } %4, 0
  store i32 %7, i32* %0, align 4
  br label %8

8:                                                ; preds = %2, %6
  %9 = zext i1 %5 to i32
  ret i32 %9
}

; Function Attrs: nofree norecurse nounwind mustprogress
define i64 @_Z30atomic_cmp_swap_relaxed_gv_i64Rll(i64* nocapture nonnull align 8 dereferenceable(8) %0, i64 %1) {
; CHECK-LABEL: _Z30atomic_cmp_swap_relaxed_gv_i64Rll:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld %s3, (, %s0)
; CHECK-NEXT:    lea %s2, gv_i64@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s2, gv_i64@hi(, %s2)
; CHECK-NEXT:    cas.l %s1, (%s2), %s3
; CHECK-NEXT:    cmps.l %s4, %s1, %s3
; CHECK-NEXT:    or %s2, 0, (0)1
; CHECK-NEXT:    cmov.l.eq %s2, (63)0, %s4
; CHECK-NEXT:    breq.l %s1, %s3, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    st %s1, (, %s0)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s2, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = load i64, i64* %0, align 8
  %4 = cmpxchg weak i64* getelementptr inbounds (%"struct.std::__1::atomic.30", %"struct.std::__1::atomic.30"* @gv_i64, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0), i64 %3, i64 %1 monotonic monotonic
  %5 = extractvalue { i64, i1 } %4, 1
  br i1 %5, label %8, label %6

6:                                                ; preds = %2
  %7 = extractvalue { i64, i1 } %4, 0
  store i64 %7, i64* %0, align 8
  br label %8

8:                                                ; preds = %2, %6
  %9 = zext i1 %5 to i64
  ret i64 %9
}

; Function Attrs: nofree norecurse nounwind mustprogress
define i64 @_Z30atomic_cmp_swap_relaxed_gv_u64Rmm(i64* nocapture nonnull align 8 dereferenceable(8) %0, i64 %1) {
; CHECK-LABEL: _Z30atomic_cmp_swap_relaxed_gv_u64Rmm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld %s3, (, %s0)
; CHECK-NEXT:    lea %s2, gv_u64@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s2, gv_u64@hi(, %s2)
; CHECK-NEXT:    cas.l %s1, (%s2), %s3
; CHECK-NEXT:    cmps.l %s4, %s1, %s3
; CHECK-NEXT:    or %s2, 0, (0)1
; CHECK-NEXT:    cmov.l.eq %s2, (63)0, %s4
; CHECK-NEXT:    breq.l %s1, %s3, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    st %s1, (, %s0)
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s2, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = load i64, i64* %0, align 8
  %4 = cmpxchg weak i64* getelementptr inbounds (%"struct.std::__1::atomic.35", %"struct.std::__1::atomic.35"* @gv_u64, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0), i64 %3, i64 %1 monotonic monotonic
  %5 = extractvalue { i64, i1 } %4, 1
  br i1 %5, label %8, label %6

6:                                                ; preds = %2
  %7 = extractvalue { i64, i1 } %4, 0
  store i64 %7, i64* %0, align 8
  br label %8

8:                                                ; preds = %2, %6
  %9 = zext i1 %5 to i64
  ret i64 %9
}

; Function Attrs: nounwind mustprogress
define i128 @_Z31atomic_cmp_swap_relaxed_gv_i128Rnn(i128* nonnull align 16 dereferenceable(16) %0, i128 %1) {
; CHECK-LABEL: _Z31atomic_cmp_swap_relaxed_gv_i128Rnn:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s6, 0, %s0
; CHECK-NEXT:    st %s2, 248(, %s11)
; CHECK-NEXT:    st %s1, 240(, %s11)
; CHECK-NEXT:    lea %s0, __atomic_compare_exchange@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, __atomic_compare_exchange@hi(, %s0)
; CHECK-NEXT:    lea %s0, gv_i128@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s1, gv_i128@hi(, %s0)
; CHECK-NEXT:    lea %s3, 240(, %s11)
; CHECK-NEXT:    or %s0, 16, (0)1
; CHECK-NEXT:    or %s4, 0, (0)1
; CHECK-NEXT:    or %s5, 0, (0)1
; CHECK-NEXT:    or %s2, 0, %s6
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = alloca i128, align 16
  %4 = bitcast i128* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %4)
  store i128 %1, i128* %3, align 16, !tbaa !2
  %5 = bitcast i128* %0 to i8*
  %6 = call zeroext i1 @__atomic_compare_exchange(i64 16, i8* nonnull bitcast (%"struct.std::__1::atomic.40"* @gv_i128 to i8*), i8* nonnull %5, i8* nonnull %4, i32 signext 0, i32 signext 0)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %4)
  %7 = zext i1 %6 to i128
  ret i128 %7
}

; Function Attrs: nounwind mustprogress
define i128 @_Z31atomic_cmp_swap_relaxed_gv_u128Roo(i128* nonnull align 16 dereferenceable(16) %0, i128 %1) {
; CHECK-LABEL: _Z31atomic_cmp_swap_relaxed_gv_u128Roo:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s6, 0, %s0
; CHECK-NEXT:    st %s2, 248(, %s11)
; CHECK-NEXT:    st %s1, 240(, %s11)
; CHECK-NEXT:    lea %s0, __atomic_compare_exchange@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, __atomic_compare_exchange@hi(, %s0)
; CHECK-NEXT:    lea %s0, gv_u128@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s1, gv_u128@hi(, %s0)
; CHECK-NEXT:    lea %s3, 240(, %s11)
; CHECK-NEXT:    or %s0, 16, (0)1
; CHECK-NEXT:    or %s4, 0, (0)1
; CHECK-NEXT:    or %s5, 0, (0)1
; CHECK-NEXT:    or %s2, 0, %s6
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = alloca i128, align 16
  %4 = bitcast i128* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %4)
  store i128 %1, i128* %3, align 16, !tbaa !2
  %5 = bitcast i128* %0 to i8*
  %6 = call zeroext i1 @__atomic_compare_exchange(i64 16, i8* nonnull bitcast (%"struct.std::__1::atomic.45"* @gv_u128 to i8*), i8* nonnull %5, i8* nonnull %4, i32 signext 0, i32 signext 0)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %4)
  %7 = zext i1 %6 to i128
  ret i128 %7
}

; Function Attrs: nounwind willreturn
declare i1 @__atomic_compare_exchange(i64, i8*, i8*, i8*, i32, i32)

!2 = !{!3, !3, i64 0}
!3 = !{!"__int128", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
