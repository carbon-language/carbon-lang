; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define double @maxf64(double, double) {
; CHECK-LABEL: maxf64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fmax.d %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = fcmp ogt double %0, %1
  %4 = select i1 %3, double %0, double %1
  ret double %4
}

define double @max2f64(double, double) {
; CHECK-LABEL: max2f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fmax.d %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = fcmp oge double %0, %1
  %4 = select i1 %3, double %0, double %1
  ret double %4
}

; VE has no max for unordered comparison
define double @maxuf64(double, double) {
; CHECK-LABEL: maxuf64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fcmp.d %s2, %s0, %s1
; CHECK-NEXT:    cmov.d.gtnan %s1, %s0, %s2
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = fcmp ugt double %0, %1
  %4 = select i1 %3, double %0, double %1
  ret double %4
}

; VE has no max for unordered comparison
define double @max2uf64(double, double) {
; CHECK-LABEL: max2uf64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fcmp.d %s2, %s0, %s1
; CHECK-NEXT:    cmov.d.genan %s1, %s0, %s2
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = fcmp uge double %0, %1
  %4 = select i1 %3, double %0, double %1
  ret double %4
}

define float @maxf32(float, float) {
; CHECK-LABEL: maxf32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fmax.s %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = fcmp ogt float %0, %1
  %4 = select i1 %3, float %0, float %1
  ret float %4
}

define float @max2f32(float, float) {
; CHECK-LABEL: max2f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fmax.s %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = fcmp oge float %0, %1
  %4 = select i1 %3, float %0, float %1
  ret float %4
}

define float @maxuf32(float, float) {
; CHECK-LABEL: maxuf32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fcmp.s %s2, %s0, %s1
; CHECK-NEXT:    cmov.s.gtnan %s1, %s0, %s2
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = fcmp ugt float %0, %1
  %4 = select i1 %3, float %0, float %1
  ret float %4
}

define float @max2uf32(float, float) {
; CHECK-LABEL: max2uf32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fcmp.s %s2, %s0, %s1
; CHECK-NEXT:    cmov.s.genan %s1, %s0, %s2
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = fcmp uge float %0, %1
  %4 = select i1 %3, float %0, float %1
  ret float %4
}

define i64 @maxi64(i64, i64) {
; CHECK-LABEL: maxi64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    maxs.l %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp sgt i64 %0, %1
  %4 = select i1 %3, i64 %0, i64 %1
  ret i64 %4
}

define i64 @max2i64(i64, i64) {
; CHECK-LABEL: max2i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    maxs.l %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp sge i64 %0, %1
  %4 = select i1 %3, i64 %0, i64 %1
  ret i64 %4
}

define i64 @maxu64(i64, i64) {
; CHECK-LABEL: maxu64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmpu.l %s2, %s0, %s1
; CHECK-NEXT:    cmov.l.gt %s1, %s0, %s2
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp ugt i64 %0, %1
  %4 = select i1 %3, i64 %0, i64 %1
  ret i64 %4
}

define i64 @max2u64(i64, i64) {
; CHECK-LABEL: max2u64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmpu.l %s2, %s0, %s1
; CHECK-NEXT:    cmov.l.ge %s1, %s0, %s2
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp uge i64 %0, %1
  %4 = select i1 %3, i64 %0, i64 %1
  ret i64 %4
}

define i32 @maxi32(i32, i32) {
; CHECK-LABEL: maxi32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    maxs.w.sx %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp sgt i32 %0, %1
  %4 = select i1 %3, i32 %0, i32 %1
  ret i32 %4
}

define i32 @max2i32(i32, i32) {
; CHECK-LABEL: max2i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    maxs.w.sx %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp sge i32 %0, %1
  %4 = select i1 %3, i32 %0, i32 %1
  ret i32 %4
}

define i32 @maxu32(i32, i32) {
; CHECK-LABEL: maxu32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmpu.w %s2, %s0, %s1
; CHECK-NEXT:    cmov.w.gt %s1, %s0, %s2
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp ugt i32 %0, %1
  %4 = select i1 %3, i32 %0, i32 %1
  ret i32 %4
}

define i32 @max2u32(i32, i32) {
; CHECK-LABEL: max2u32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmpu.w %s2, %s0, %s1
; CHECK-NEXT:    cmov.w.ge %s1, %s0, %s2
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp uge i32 %0, %1
  %4 = select i1 %3, i32 %0, i32 %1
  ret i32 %4
}

define zeroext i1 @maxi1(i1 zeroext, i1 zeroext) {
; CHECK-LABEL: maxi1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    or %s0, %s1, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = xor i1 %1, true
  %4 = and i1 %3, %0
  %5 = select i1 %4, i1 %0, i1 %1
  ret i1 %5
}
