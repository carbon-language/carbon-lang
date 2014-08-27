; RUN: llc < %s -mtriple=aarch64-none-eabi | FileCheck %s

define <8 x half> @add_h(<8 x half> %a, <8 x half> %b) {
entry:
; CHECK-LABEL: add_h:
; CHECK: fcvt
; CHECK: fcvt
; CHECK-DAG: fadd
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fadd
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fadd
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fadd
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fadd
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fadd
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fadd
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fadd
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK: fcvt
  %0 = fadd <8 x half> %a, %b
  ret <8 x half> %0
}


define <8 x half> @sub_h(<8 x half> %a, <8 x half> %b) {
entry:
; CHECK-LABEL: sub_h:
; CHECK: fcvt
; CHECK: fcvt
; CHECK-DAG: fsub
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fsub
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fsub
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fsub
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fsub
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fsub
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fsub
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fsub
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK: fcvt
  %0 = fsub <8 x half> %a, %b
  ret <8 x half> %0
}


define <8 x half> @mul_h(<8 x half> %a, <8 x half> %b) {
entry:
; CHECK-LABEL: mul_h:
; CHECK: fcvt
; CHECK: fcvt
; CHECK-DAG: fmul
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fmul
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fmul
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fmul
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fmul
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fmul
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fmul
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fmul
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK: fcvt
  %0 = fmul <8 x half> %a, %b
  ret <8 x half> %0
}


define <8 x half> @div_h(<8 x half> %a, <8 x half> %b) {
entry:
; CHECK-LABEL: div_h:
; CHECK: fcvt
; CHECK: fcvt
; CHECK-DAG: fdiv
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fdiv
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fdiv
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fdiv
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fdiv
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fdiv
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fdiv
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fdiv
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK: fcvt
  %0 = fdiv <8 x half> %a, %b
  ret <8 x half> %0
}


define <8 x half> @load_h(<8 x half>* %a) {
entry:
; CHECK-LABEL: load_h:
; CHECK: ldr q0, [x0]
  %0 = load <8 x half>* %a, align 4
  ret <8 x half> %0
}


define void @store_h(<8 x half>* %a, <8 x half> %b) {
entry:
; CHECK-LABEL: store_h:
; CHECK: str q0, [x0]
  store <8 x half> %b, <8 x half>* %a, align 4
  ret void
}

define <8 x half> @s_to_h(<8 x float> %a) {
; CHECK-LABEL: s_to_h:
; CHECK-DAG: fcvtn v0.4h, v0.4s
; CHECK-DAG: fcvtn [[REG:v[0-9+]]].4h, v1.4s
; CHECK: ins v0.d[1], [[REG]].d[0]
  %1 = fptrunc <8 x float> %a to <8 x half>
  ret <8 x half> %1
}

define <8 x half> @d_to_h(<8 x double> %a) {
; CHECK-LABEL: d_to_h:
; CHECK-DAG: ins v{{[0-9]+}}.d
; CHECK-DAG: ins v{{[0-9]+}}.d
; CHECK-DAG: ins v{{[0-9]+}}.d
; CHECK-DAG: ins v{{[0-9]+}}.d
; CHECK-DAG: fcvt h
; CHECK-DAG: fcvt h
; CHECK-DAG: fcvt h
; CHECK-DAG: fcvt h
; CHECK-DAG: fcvt h
; CHECK-DAG: fcvt h
; CHECK-DAG: fcvt h
; CHECK-DAG: fcvt h
; CHECK-DAG: ins v{{[0-9]+}}.h
; CHECK-DAG: ins v{{[0-9]+}}.h
; CHECK-DAG: ins v{{[0-9]+}}.h
; CHECK-DAG: ins v{{[0-9]+}}.h
; CHECK-DAG: ins v{{[0-9]+}}.h
; CHECK-DAG: ins v{{[0-9]+}}.h
; CHECK-DAG: ins v{{[0-9]+}}.h
; CHECK-DAG: ins v{{[0-9]+}}.h
  %1 = fptrunc <8 x double> %a to <8 x half>
  ret <8 x half> %1
}

define <8 x float> @h_to_s(<8 x half> %a) {
; CHECK-LABEL: h_to_s:
; CHECK: fcvtl2 v1.4s, v0.8h
; CHECK: fcvtl v0.4s, v0.4h
  %1 = fpext <8 x half> %a to <8 x float>
  ret <8 x float> %1
}

define <8 x double> @h_to_d(<8 x half> %a) {
; CHECK-LABEL: h_to_d:
; CHECK-DAG: fcvt d
; CHECK-DAG: fcvt d
; CHECK-DAG: fcvt d
; CHECK-DAG: fcvt d
; CHECK-DAG: fcvt d
; CHECK-DAG: fcvt d
; CHECK-DAG: fcvt d
; CHECK-DAG: fcvt d
; CHECK-DAG: ins
; CHECK-DAG: ins
; CHECK-DAG: ins
; CHECK-DAG: ins
  %1 = fpext <8 x half> %a to <8 x double>
  ret <8 x double> %1
}


define <8 x half> @bitcast_i_to_h(float, <8 x i16> %a) {
; CHECK-LABEL: bitcast_i_to_h:
; CHECK: mov v0.16b, v1.16b
  %2 = bitcast <8 x i16> %a to <8 x half>
  ret <8 x half> %2
}

define <8 x i16> @bitcast_h_to_i(float, <8 x half> %a) {
; CHECK-LABEL: bitcast_h_to_i:
; CHECK: mov v0.16b, v1.16b
  %2 = bitcast <8 x half> %a to <8 x i16>
  ret <8 x i16> %2
}

