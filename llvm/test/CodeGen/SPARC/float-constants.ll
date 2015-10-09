; RUN: llc < %s -march=sparc | FileCheck %s

; TODO: actually fix the codegen to be optimal. At least we don't
; crash for now, though...

;; Bitcast should not do a runtime conversion, but rather emit a
;; constant into integer registers directly.

; CHECK-LABEL: bitcast:
; TODO-CHECK: sethi 1049856, %o0
; TODO-CHECK: sethi 0, %o1
define <2 x i32> @bitcast() {
  %1 = bitcast double 5.0 to <2 x i32>
  ret <2 x i32> %1
}

;; Same thing for a call using a double (which gets passed in integer
;; registers)

; CHECK-LABEL: test_call
; TODO-CHECK: sethi 1049856, %o0
; TODO-CHECK: sethi 0, %o1
declare void @a(double)
define void @test_call() {
  call void @a(double 5.0)
  ret void
}

;; And for a libcall emitted from the pow intrinsic.  (libcall
;; emission happens after SelectionDAG type legalization, so is a bit
;; different than a normal function call. This was crashing before,
;; due to an earlier broken workaround for this issue.)

; CHECK-LABEL: test_intrins_call
; TODO-CHECK: sethi 1049856, %o0
; TODO-CHECK: sethi 0, %o1
declare double @llvm.pow.f64(double, double)
define double @test_intrins_call() {
  %1 = call double @llvm.pow.f64(double 2.0, double 2.0)
  ret double %1
}
