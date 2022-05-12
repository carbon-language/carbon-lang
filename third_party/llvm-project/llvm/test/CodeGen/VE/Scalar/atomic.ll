; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

;;; Test atomicrmw operations

@c = common global i8 0, align 4
@s = common global i16 0, align 4
@i = common global i32 0, align 4
@l = common global i64 0, align 4

; Function Attrs: norecurse nounwind
define signext i8 @test_atomic_fetch_add_1() {
; CHECK-LABEL: test_atomic_fetch_add_1:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    lea %s0, c@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, c@hi(, %s0)
; CHECK-NEXT:    and %s0, -4, %s0
; CHECK-NEXT:    ldl.sx %s2, (, %s0)
; CHECK-NEXT:    lea %s1, -256
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:  .LBB{{[0-9]+}}_1: # %atomicrmw.start
; CHECK-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    or %s3, 0, %s2
; CHECK-NEXT:    adds.w.sx %s2, 1, %s2
; CHECK-NEXT:    and %s2, %s2, (56)0
; CHECK-NEXT:    and %s4, %s3, %s1
; CHECK-NEXT:    or %s2, %s4, %s2
; CHECK-NEXT:    cas.w %s2, (%s0), %s3
; CHECK-NEXT:    brne.w %s2, %s3, .LBB{{[0-9]+}}_1
; CHECK-NEXT:  # %bb.2: # %atomicrmw.end
; CHECK-NEXT:    sll %s0, %s2, 56
; CHECK-NEXT:    sra.l %s0, %s0, 56
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %0 = atomicrmw add i8* @c, i8 1 seq_cst
  ret i8 %0
}

; Function Attrs: norecurse nounwind
define signext i16 @test_atomic_fetch_sub_2() {
; CHECK-LABEL: test_atomic_fetch_sub_2:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    lea %s0, s@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, s@hi(, %s0)
; CHECK-NEXT:    and %s0, -4, %s0
; CHECK-NEXT:    ldl.sx %s2, (, %s0)
; CHECK-NEXT:    lea %s1, -65536
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:  .LBB{{[0-9]+}}_1: # %atomicrmw.start
; CHECK-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    or %s3, 0, %s2
; CHECK-NEXT:    adds.w.sx %s2, -1, %s2
; CHECK-NEXT:    and %s2, %s2, (48)0
; CHECK-NEXT:    and %s4, %s3, %s1
; CHECK-NEXT:    or %s2, %s4, %s2
; CHECK-NEXT:    cas.w %s2, (%s0), %s3
; CHECK-NEXT:    brne.w %s2, %s3, .LBB{{[0-9]+}}_1
; CHECK-NEXT:  # %bb.2: # %atomicrmw.end
; CHECK-NEXT:    sll %s0, %s2, 48
; CHECK-NEXT:    sra.l %s0, %s0, 48
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %0 = atomicrmw sub i16* @s, i16 1 seq_cst
  ret i16 %0
}

; Function Attrs: norecurse nounwind
define signext i32 @test_atomic_fetch_and_4() {
; CHECK-LABEL: test_atomic_fetch_and_4:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    lea %s0, i@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, i@hi(, %s0)
; CHECK-NEXT:    ldl.sx %s1, (, %s0)
; CHECK-NEXT:  .LBB{{[0-9]+}}_1: # %atomicrmw.start
; CHECK-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    or %s2, 0, %s1
; CHECK-NEXT:    and %s1, 1, %s2
; CHECK-NEXT:    cas.w %s1, (%s0), %s2
; CHECK-NEXT:    brne.w %s1, %s2, .LBB{{[0-9]+}}_1
; CHECK-NEXT:  # %bb.2: # %atomicrmw.end
; CHECK-NEXT:    adds.w.sx %s0, %s1, (0)1
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %0 = atomicrmw and i32* @i, i32 1 seq_cst
  ret i32 %0
}
; Function Attrs: norecurse nounwind
define i64 @test_atomic_fetch_or_8() {
; CHECK-LABEL: test_atomic_fetch_or_8:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    lea %s0, l@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s1, l@hi(, %s0)
; CHECK-NEXT:    ld %s0, (, %s1)
; CHECK-NEXT:  .LBB{{[0-9]+}}_1: # %atomicrmw.start
; CHECK-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    or %s2, 0, %s0
; CHECK-NEXT:    or %s0, 1, %s0
; CHECK-NEXT:    cas.l %s0, (%s1), %s2
; CHECK-NEXT:    brne.l %s0, %s2, .LBB{{[0-9]+}}_1
; CHECK-NEXT:  # %bb.2: # %atomicrmw.end
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %0 = atomicrmw or i64* @l, i64 1 seq_cst
  ret i64 %0
}

; Function Attrs: norecurse nounwind
define signext i8 @test_atomic_fetch_xor_1() {
; CHECK-LABEL: test_atomic_fetch_xor_1:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    lea %s0, c@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, c@hi(, %s0)
; CHECK-NEXT:    and %s1, -4, %s0
; CHECK-NEXT:    ldl.sx %s0, (, %s1)
; CHECK-NEXT:  .LBB{{[0-9]+}}_1: # %atomicrmw.start
; CHECK-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    or %s2, 0, %s0
; CHECK-NEXT:    xor %s0, 1, %s2
; CHECK-NEXT:    cas.w %s0, (%s1), %s2
; CHECK-NEXT:    brne.w %s0, %s2, .LBB{{[0-9]+}}_1
; CHECK-NEXT:  # %bb.2: # %atomicrmw.end
; CHECK-NEXT:    sll %s0, %s0, 56
; CHECK-NEXT:    sra.l %s0, %s0, 56
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %0 = atomicrmw xor i8* @c, i8 1 seq_cst
  ret i8 %0
}

; Function Attrs: norecurse nounwind
define signext i16 @test_atomic_fetch_nand_2() {
; CHECK-LABEL: test_atomic_fetch_nand_2:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    lea %s0, s@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, s@hi(, %s0)
; CHECK-NEXT:    and %s0, -4, %s0
; CHECK-NEXT:    ldl.sx %s2, (, %s0)
; CHECK-NEXT:    lea %s1, 65534
; CHECK-NEXT:    lea %s3, -65536
; CHECK-NEXT:    and %s3, %s3, (32)0
; CHECK-NEXT:  .LBB{{[0-9]+}}_1: # %atomicrmw.start
; CHECK-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    or %s4, 0, %s2
; CHECK-NEXT:    xor %s2, -1, %s4
; CHECK-NEXT:    or %s2, %s2, %s1
; CHECK-NEXT:    and %s2, %s2, (48)0
; CHECK-NEXT:    and %s5, %s4, %s3
; CHECK-NEXT:    or %s2, %s5, %s2
; CHECK-NEXT:    cas.w %s2, (%s0), %s4
; CHECK-NEXT:    brne.w %s2, %s4, .LBB{{[0-9]+}}_1
; CHECK-NEXT:  # %bb.2: # %atomicrmw.end
; CHECK-NEXT:    sll %s0, %s2, 48
; CHECK-NEXT:    sra.l %s0, %s0, 48
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %0 = atomicrmw nand i16* @s, i16 1 seq_cst
  ret i16 %0
}

; Function Attrs: norecurse nounwind
define signext i32 @test_atomic_fetch_max_4() {
; CHECK-LABEL: test_atomic_fetch_max_4:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    lea %s0, i@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s1, i@hi(, %s0)
; CHECK-NEXT:    ldl.sx %s0, (, %s1)
; CHECK-NEXT:    or %s2, 1, (0)1
; CHECK-NEXT:  .LBB{{[0-9]+}}_1: # %atomicrmw.start
; CHECK-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    or %s3, 0, %s0
; CHECK-NEXT:    maxs.w.sx %s0, %s0, %s2
; CHECK-NEXT:    cas.w %s0, (%s1), %s3
; CHECK-NEXT:    brne.w %s0, %s3, .LBB{{[0-9]+}}_1
; CHECK-NEXT:  # %bb.2: # %atomicrmw.end
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %0 = atomicrmw max i32* @i, i32 1 seq_cst
  ret i32 %0
}

; Function Attrs: norecurse nounwind
define signext i32 @test_atomic_fetch_min_4() {
; CHECK-LABEL: test_atomic_fetch_min_4:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    lea %s0, i@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, i@hi(, %s0)
; CHECK-NEXT:    ldl.sx %s1, (, %s0)
; CHECK-NEXT:    or %s2, 2, (0)1
; CHECK-NEXT:  .LBB{{[0-9]+}}_1: # %atomicrmw.start
; CHECK-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    or %s3, 0, %s1
; CHECK-NEXT:    cmps.w.sx %s4, %s1, %s2
; CHECK-NEXT:    or %s1, 1, (0)1
; CHECK-NEXT:    cmov.w.lt %s1, %s3, %s4
; CHECK-NEXT:    cas.w %s1, (%s0), %s3
; CHECK-NEXT:    brne.w %s1, %s3, .LBB{{[0-9]+}}_1
; CHECK-NEXT:  # %bb.2: # %atomicrmw.end
; CHECK-NEXT:    adds.w.sx %s0, %s1, (0)1
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %0 = atomicrmw min i32* @i, i32 1 seq_cst
  ret i32 %0
}

; Function Attrs: norecurse nounwind
define signext i32 @test_atomic_fetch_umax_4() {
; CHECK-LABEL: test_atomic_fetch_umax_4:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    lea %s0, i@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, i@hi(, %s0)
; CHECK-NEXT:    ldl.sx %s1, (, %s0)
; CHECK-NEXT:    or %s2, 1, (0)1
; CHECK-NEXT:  .LBB{{[0-9]+}}_1: # %atomicrmw.start
; CHECK-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    or %s3, 0, %s1
; CHECK-NEXT:    cmpu.w %s4, %s1, %s2
; CHECK-NEXT:    or %s1, 1, (0)1
; CHECK-NEXT:    cmov.w.gt %s1, %s3, %s4
; CHECK-NEXT:    cas.w %s1, (%s0), %s3
; CHECK-NEXT:    brne.w %s1, %s3, .LBB{{[0-9]+}}_1
; CHECK-NEXT:  # %bb.2: # %atomicrmw.end
; CHECK-NEXT:    adds.w.sx %s0, %s1, (0)1
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %0 = atomicrmw umax i32* @i, i32 1 seq_cst
  ret i32 %0
}

; Function Attrs: norecurse nounwind
define signext i32 @test_atomic_fetch_umin_4() {
; CHECK-LABEL: test_atomic_fetch_umin_4:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    lea %s0, i@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, i@hi(, %s0)
; CHECK-NEXT:    ldl.sx %s1, (, %s0)
; CHECK-NEXT:    or %s2, 2, (0)1
; CHECK-NEXT:  .LBB{{[0-9]+}}_1: # %atomicrmw.start
; CHECK-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    or %s3, 0, %s1
; CHECK-NEXT:    cmpu.w %s4, %s1, %s2
; CHECK-NEXT:    or %s1, 1, (0)1
; CHECK-NEXT:    cmov.w.lt %s1, %s3, %s4
; CHECK-NEXT:    cas.w %s1, (%s0), %s3
; CHECK-NEXT:    brne.w %s1, %s3, .LBB{{[0-9]+}}_1
; CHECK-NEXT:  # %bb.2: # %atomicrmw.end
; CHECK-NEXT:    adds.w.sx %s0, %s1, (0)1
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %0 = atomicrmw umin i32* @i, i32 1 seq_cst
  ret i32 %0
}
