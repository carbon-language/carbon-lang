; RUN: llc -mtriple=aarch64-apple-darwin -fast-isel -fast-isel-abort -code-model=small -verify-machineinstrs < %s | FileCheck %s --check-prefix=SMALL
; RUN: llc -mtriple=aarch64-apple-darwin -fast-isel -fast-isel-abort -code-model=large -verify-machineinstrs < %s | FileCheck %s --check-prefix=LARGE

define float @frem_f32(float %a, float %b) {
; SMALL-LABEL: frem_f32
; SMALL:       bl _fmodf
; LARGE-LABEL: frem_f32
; LARGE:       adrp  [[REG:x[0-9]+]], _fmodf@GOTPAGE
; LARGE:       ldr [[REG]], {{\[}}[[REG]], _fmodf@GOTPAGEOFF{{\]}}
; LARGE-NEXT:  blr [[REG]]
  %1 = frem float %a, %b
  ret float %1
}

define double @frem_f64(double %a, double %b) {
; SMALL-LABEL: frem_f64
; SMALL:       bl _fmod
; LARGE-LABEL: frem_f64
; LARGE:       adrp  [[REG:x[0-9]+]], _fmod@GOTPAGE
; LARGE:       ldr [[REG]], {{\[}}[[REG]], _fmod@GOTPAGEOFF{{\]}}
; LARGE-NEXT:  blr [[REG]]
  %1 = frem double %a, %b
  ret double %1
}
