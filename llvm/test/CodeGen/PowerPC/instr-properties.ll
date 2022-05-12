; RUN: llc < %s -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 -verify-misched -stop-after=machine-scheduler -o - | FileCheck %s --check-prefix=CHECK-P8

; Verify XFLOADf64 didn't implict def 'rm'.
define double @rm() {
; CHECK-P8-LABEL: bb.0.entry
; CHECK-P8: %{{[0-9]+}}:f8rc = LFD target-flags(ppc-toc-lo) %const.0, %{{[0-9]+}}, implicit $x2 ::
entry:
  ret double 2.300000e+00
}
