; RUN: llc < %s -mtriple=aarch64-none-eabi | FileCheck %s

define half @add_h(half %a, half %b) {
entry:
; CHECK-LABEL: add_h:
; CHECK: fcvt
; CHECK: fcvt
; CHECK: fadd
; CHECK: fcvt
  %0 = fadd half %a, %b
  ret half %0
}


define half @sub_h(half %a, half %b) {
entry:
; CHECK-LABEL: sub_h:
; CHECK: fcvt
; CHECK: fcvt
; CHECK: fsub
; CHECK: fcvt
  %0 = fsub half %a, %b
  ret half %0
}


define half @mul_h(half %a, half %b) {
entry:
; CHECK-LABEL: mul_h:
; CHECK: fcvt
; CHECK: fcvt
; CHECK: fmul
; CHECK: fcvt
  %0 = fmul half %a, %b
  ret half %0
}


define half @div_h(half %a, half %b) {
entry:
; CHECK-LABEL: div_h:
; CHECK: fcvt
; CHECK: fcvt
; CHECK: fdiv
; CHECK: fcvt
  %0 = fdiv half %a, %b
  ret half %0
}


define half @load_h(half* %a) {
entry:
; CHECK-LABEL: load_h:
; CHECK: ldr h
  %0 = load half* %a, align 4
  ret half %0
}


define void @store_h(half* %a, half %b) {
entry:
; CHECK-LABEL: store_h:
; CHECK: str h
  store half %b, half* %a, align 4
  ret void
}

define half @s_to_h(float %a) {
; CHECK-LABEL: s_to_h:
; CHECK: fcvt
  %1 = fptrunc float %a to half
  ret half %1
}

define half @d_to_h(double %a) {
; CHECK-LABEL: d_to_h:
; CHECK: fcvt
  %1 = fptrunc double %a to half
  ret half %1
}

define float @h_to_s(half %a) {
; CHECK-LABEL: h_to_s:
; CHECK: fcvt
  %1 = fpext half %a to float
  ret float %1
}

define double @h_to_d(half %a) {
; CHECK-LABEL: h_to_d:
; CHECK: fcvt
  %1 = fpext half %a to double
  ret double %1
}

define half @bitcast_i_to_h(i16 %a) {
; CHECK-LABEL: bitcast_i_to_h:
; CHECK: fmov
  %1 = bitcast i16 %a to half
  ret half %1
}


define i16 @bitcast_h_to_i(half %a) {
; CHECK-LABEL: bitcast_h_to_i:
; CHECK: fmov
  %1 = bitcast half %a to i16
  ret i16 %1
}
