; RUN: llc < %s -mtriple=aarch64-none-eabi | FileCheck %s

define half @add_h(half %a, half %b) {
entry:
; CHECK-LABEL: add_h:
; CHECK-DAG: fcvt [[OP1:s[0-9]+]], h0
; CHECK-DAG: fcvt [[OP2:s[0-9]+]], h1
; CHECK: fadd [[RES:s[0-9]+]], [[OP1]], [[OP2]]
; CHECK: fcvt h0, [[RES]]
  %0 = fadd half %a, %b
  ret half %0
}


define half @sub_h(half %a, half %b) {
entry:
; CHECK-LABEL: sub_h:
; CHECK-DAG: fcvt [[OP1:s[0-9]+]], h0
; CHECK-DAG: fcvt [[OP2:s[0-9]+]], h1
; CHECK: fsub [[RES:s[0-9]+]], [[OP1]], [[OP2]]
; CHECK: fcvt h0, [[RES]]
  %0 = fsub half %a, %b
  ret half %0
}


define half @mul_h(half %a, half %b) {
entry:
; CHECK-LABEL: mul_h:
; CHECK-DAG: fcvt [[OP1:s[0-9]+]], h0
; CHECK-DAG: fcvt [[OP2:s[0-9]+]], h1
; CHECK: fmul [[RES:s[0-9]+]], [[OP1]], [[OP2]]
; CHECK: fcvt h0, [[RES]]
  %0 = fmul half %a, %b
  ret half %0
}


define half @div_h(half %a, half %b) {
entry:
; CHECK-LABEL: div_h:
; CHECK-DAG: fcvt [[OP1:s[0-9]+]], h0
; CHECK-DAG: fcvt [[OP2:s[0-9]+]], h1
; CHECK: fdiv [[RES:s[0-9]+]], [[OP1]], [[OP2]]
; CHECK: fcvt h0, [[RES]]
  %0 = fdiv half %a, %b
  ret half %0
}


define half @load_h(half* %a) {
entry:
; CHECK-LABEL: load_h:
; CHECK: ldr h0, [x0]
  %0 = load half, half* %a, align 4
  ret half %0
}


define void @store_h(half* %a, half %b) {
entry:
; CHECK-LABEL: store_h:
; CHECK: str h0, [x0]
  store half %b, half* %a, align 4
  ret void
}

define half @s_to_h(float %a) {
; CHECK-LABEL: s_to_h:
; CHECK: fcvt h0, s0
  %1 = fptrunc float %a to half
  ret half %1
}

define half @d_to_h(double %a) {
; CHECK-LABEL: d_to_h:
; CHECK: fcvt h0, d0
  %1 = fptrunc double %a to half
  ret half %1
}

define float @h_to_s(half %a) {
; CHECK-LABEL: h_to_s:
; CHECK: fcvt s0, h0
  %1 = fpext half %a to float
  ret float %1
}

define double @h_to_d(half %a) {
; CHECK-LABEL: h_to_d:
; CHECK: fcvt d0, h0
  %1 = fpext half %a to double
  ret double %1
}

define half @bitcast_i_to_h(i16 %a) {
; CHECK-LABEL: bitcast_i_to_h:
; CHECK: fmov s0, w0
  %1 = bitcast i16 %a to half
  ret half %1
}


define i16 @bitcast_h_to_i(half %a) {
; CHECK-LABEL: bitcast_h_to_i:
; CHECK: fmov w0, s0
  %1 = bitcast half %a to i16
  ret i16 %1
}
