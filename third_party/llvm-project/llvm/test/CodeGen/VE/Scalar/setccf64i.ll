; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define zeroext i1 @setccaf(double, double) {
; CHECK-LABEL: setccaf:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = fcmp false double %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccat(double, double) {
; CHECK-LABEL: setccat:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s0, 1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = fcmp true double %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccoeq(double, double) {
; CHECK-LABEL: setccoeq:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea.sl %s1, 0
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.d.eq %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = fcmp oeq double %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccone(double, double) {
; CHECK-LABEL: setccone:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea.sl %s1, 0
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.d.ne %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = fcmp one double %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccogt(double, double) {
; CHECK-LABEL: setccogt:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea.sl %s1, 0
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.d.gt %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = fcmp ogt double %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccoge(double, double) {
; CHECK-LABEL: setccoge:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea.sl %s1, 0
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.d.ge %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = fcmp oge double %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccolt(double, double) {
; CHECK-LABEL: setccolt:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea.sl %s1, 0
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.d.lt %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = fcmp olt double %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccole(double, double) {
; CHECK-LABEL: setccole:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea.sl %s1, 0
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.d.le %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = fcmp ole double %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccord(double, double) {
; CHECK-LABEL: setccord:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fcmp.d %s0, %s0, %s0
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.d.num %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = fcmp ord double %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccuno(double, double) {
; CHECK-LABEL: setccuno:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fcmp.d %s0, %s0, %s0
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.d.nan %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = fcmp uno double %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccueq(double, double) {
; CHECK-LABEL: setccueq:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea.sl %s1, 0
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.d.eqnan %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = fcmp ueq double %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccune(double, double) {
; CHECK-LABEL: setccune:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea.sl %s1, 0
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.d.nenan %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = fcmp une double %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccugt(double, double) {
; CHECK-LABEL: setccugt:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea.sl %s1, 0
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.d.gtnan %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = fcmp ugt double %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccuge(double, double) {
; CHECK-LABEL: setccuge:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea.sl %s1, 0
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.d.genan %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = fcmp uge double %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccult(double, double) {
; CHECK-LABEL: setccult:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea.sl %s1, 0
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.d.ltnan %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = fcmp ult double %0, 0.0
  ret i1 %3
}

define zeroext i1 @setccule(double, double) {
; CHECK-LABEL: setccule:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea.sl %s1, 0
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.d.lenan %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = fcmp ule double %0, 0.0
  ret i1 %3
}
