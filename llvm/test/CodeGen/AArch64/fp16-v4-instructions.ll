; RUN: llc < %s -mtriple=aarch64-none-eabi | FileCheck %s

define <4 x half> @add_h(<4 x half> %a, <4 x half> %b) {
entry:
; CHECK-LABEL: add_h:
; CHECK-DAG: fcvtl [[OP1:v[0-9]+\.4s]], v0.4h
; CHECK-DAG: fcvtl [[OP2:v[0-9]+\.4s]], v1.4h
; CHECK: fadd [[RES:v[0-9]+.4s]], [[OP1]], [[OP2]]
; CHECK: fcvtn v0.4h, [[RES]]
  %0 = fadd <4 x half> %a, %b
  ret <4 x half> %0
}


define <4 x half> @build_h4(<4 x half> %a) {
entry:
; CHECK-LABEL: build_h4:
; CHECK: movz [[GPR:w[0-9]+]], #0x3ccd
; CHECK: dup v0.4h, [[GPR]]
  ret <4 x half> <half 0xH3CCD, half 0xH3CCD, half 0xH3CCD, half 0xH3CCD>
}


define <4 x half> @sub_h(<4 x half> %a, <4 x half> %b) {
entry:
; CHECK-LABEL: sub_h:
; CHECK-DAG: fcvtl [[OP1:v[0-9]+\.4s]], v0.4h
; CHECK-DAG: fcvtl [[OP2:v[0-9]+\.4s]], v1.4h
; CHECK: fsub [[RES:v[0-9]+.4s]], [[OP1]], [[OP2]]
; CHECK: fcvtn v0.4h, [[RES]]
  %0 = fsub <4 x half> %a, %b
  ret <4 x half> %0
}


define <4 x half> @mul_h(<4 x half> %a, <4 x half> %b) {
entry:
; CHECK-LABEL: mul_h:
; CHECK-DAG: fcvtl [[OP1:v[0-9]+\.4s]], v0.4h
; CHECK-DAG: fcvtl [[OP2:v[0-9]+\.4s]], v1.4h
; CHECK: fmul [[RES:v[0-9]+.4s]], [[OP1]], [[OP2]]
; CHECK: fcvtn v0.4h, [[RES]]
  %0 = fmul <4 x half> %a, %b
  ret <4 x half> %0
}


define <4 x half> @div_h(<4 x half> %a, <4 x half> %b) {
entry:
; CHECK-LABEL: div_h:
; CHECK-DAG: fcvtl [[OP1:v[0-9]+\.4s]], v0.4h
; CHECK-DAG: fcvtl [[OP2:v[0-9]+\.4s]], v1.4h
; CHECK: fdiv [[RES:v[0-9]+.4s]], [[OP1]], [[OP2]]
; CHECK: fcvtn v0.4h, [[RES]]
  %0 = fdiv <4 x half> %a, %b
  ret <4 x half> %0
}


define <4 x half> @load_h(<4 x half>* %a) {
entry:
; CHECK-LABEL: load_h:
; CHECK: ldr d0, [x0]
  %0 = load <4 x half>, <4 x half>* %a, align 4
  ret <4 x half> %0
}


define void @store_h(<4 x half>* %a, <4 x half> %b) {
entry:
; CHECK-LABEL: store_h:
; CHECK: str d0, [x0]
  store <4 x half> %b, <4 x half>* %a, align 4
  ret void
}

define <4 x half> @s_to_h(<4 x float> %a) {
; CHECK-LABEL: s_to_h:
; CHECK: fcvtn v0.4h, v0.4s
  %1 = fptrunc <4 x float> %a to <4 x half>
  ret <4 x half> %1
}

define <4 x half> @d_to_h(<4 x double> %a) {
; CHECK-LABEL: d_to_h:
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: ins
; CHECK-DAG: ins
; CHECK-DAG: ins
; CHECK-DAG: ins
  %1 = fptrunc <4 x double> %a to <4 x half>
  ret <4 x half> %1
}

define <4 x float> @h_to_s(<4 x half> %a) {
; CHECK-LABEL: h_to_s:
; CHECK: fcvtl v0.4s, v0.4h
  %1 = fpext <4 x half> %a to <4 x float>
  ret <4 x float> %1
}

define <4 x double> @h_to_d(<4 x half> %a) {
; CHECK-LABEL: h_to_d:
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: fcvt
; CHECK-DAG: ins
; CHECK-DAG: ins
; CHECK-DAG: ins
; CHECK-DAG: ins
  %1 = fpext <4 x half> %a to <4 x double>
  ret <4 x double> %1
}

define <4 x half> @bitcast_i_to_h(float, <4 x i16> %a) {
; CHECK-LABEL: bitcast_i_to_h:
; CHECK: mov v0.16b, v1.16b
  %2 = bitcast <4 x i16> %a to <4 x half>
  ret <4 x half> %2
}

define <4 x i16> @bitcast_h_to_i(float, <4 x half> %a) {
; CHECK-LABEL: bitcast_h_to_i:
; CHECK: mov v0.16b, v1.16b
  %2 = bitcast <4 x half> %a to <4 x i16>
  ret <4 x i16> %2
}
