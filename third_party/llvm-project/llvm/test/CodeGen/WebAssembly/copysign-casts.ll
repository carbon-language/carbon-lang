; RUN: llc < %s -asm-verbose=false -wasm-keep-registers | FileCheck %s

; DAGCombiner oddly folds casts into the rhs of copysign. Test that they get
; unfolded.

target triple = "wasm32-unknown-unknown"

declare double @copysign(double, double) nounwind readnone
declare float @copysignf(float, float) nounwind readnone

; CHECK-LABEL: fold_promote:
; CHECK: f64.promote_f32 $push0=, $pop{{[0-9]+}}{{$}}
; CHECK: f64.copysign    $push1=, $pop{{[0-9]+}}, $pop0{{$}}
define double @fold_promote(double %a, float %b) {
  %c = fpext float %b to double
  %t = call double @copysign(double %a, double %c)
  ret double %t
}

; CHECK-LABEL: fold_demote:{{$}}
; CHECK: f32.demote_f64  $push0=, $pop{{[0-9]+}}{{$}}
; CHECK: f32.copysign    $push1=, $pop{{[0-9]+}}, $pop0{{$}}
define float @fold_demote(float %a, double %b) {
  %c = fptrunc double %b to float
  %t = call float @copysignf(float %a, float %c)
  ret float %t
}
