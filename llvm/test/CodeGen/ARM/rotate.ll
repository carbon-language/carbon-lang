; RUN: llc < %s -mtriple=thumbv8--linux-gnueabihf | FileCheck %s

;; This used to cause a backend crash about not being able to
;; select ROTL. Make sure if generates the basic VSHL/VSHR.
define <2 x i64> @testcase(<2 x i64>* %in) {
; CHECK-LABEL: testcase
; CHECK: vshl.i64
; CHECK: vshr.u64
  %1 = load <2 x i64>, <2 x i64>* %in
  %2 = lshr <2 x i64> %1, <i64 8, i64 8>
  %3 = shl <2 x i64> %1, <i64 56, i64 56>
  %4 = or <2 x i64> %2, %3
  ret <2 x i64> %4
}
