; RUN: llc -mtriple=aarch64--linux-gnu -mattr=sve < %s | FileCheck %s

; NOTE: invalid, upper and lower bound immediate values of the regimm
; addressing mode are checked only for the byte version of each
; instruction (`ld<N>b`), as the code for detecting the immediate is
; common to all instructions, and varies only for the number of
; elements of the structure store, which is <N> = 2, 3, 4.

; ld2b
define <vscale x 32 x i8> @ld2.nxv32i8(<vscale x 16 x i1> %Pg, <vscale x 16 x i8> *%addr) {
; CHECK-LABEL: ld2.nxv32i8:
; CHECK: ld2b { z0.b, z1.b }, p0/z, [x0, #2, mul vl]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 2
%base_ptr = bitcast <vscale x 16 x i8>* %base to i8*
%res = call <vscale x 32 x i8> @llvm.aarch64.sve.ld2.nxv32i8.nxv16i1.p0i8(<vscale x 16 x i1> %Pg, i8 *%base_ptr)
ret <vscale x 32 x i8> %res
}

define <vscale x 32 x i8> @ld2.nxv32i8_lower_bound(<vscale x 16 x i1> %Pg, <vscale x 16 x i8> *%addr) {
; CHECK-LABEL: ld2.nxv32i8_lower_bound:
; CHECK: ld2b { z0.b, z1.b }, p0/z, [x0, #-16, mul vl]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 -16
%base_ptr = bitcast <vscale x 16 x i8>* %base to i8 *
%res = call <vscale x 32 x i8> @llvm.aarch64.sve.ld2.nxv32i8.nxv16i1.p0i8(<vscale x 16 x i1> %Pg, i8 *%base_ptr)
ret <vscale x 32 x i8> %res
}

define <vscale x 32 x i8> @ld2.nxv32i8_upper_bound(<vscale x 16 x i1> %Pg, <vscale x 16 x i8> *%addr) {
; CHECK-LABEL: ld2.nxv32i8_upper_bound:
; CHECK: ld2b { z0.b, z1.b }, p0/z, [x0, #14, mul vl]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 14
%base_ptr = bitcast <vscale x 16 x i8>* %base to i8 *
%res = call <vscale x 32 x i8> @llvm.aarch64.sve.ld2.nxv32i8.nxv16i1.p0i8(<vscale x 16 x i1> %Pg, i8 *%base_ptr)
ret <vscale x 32 x i8> %res
}

define <vscale x 32 x i8> @ld2.nxv32i8_not_multiple_of_2(<vscale x 16 x i1> %Pg, <vscale x 16 x i8> *%addr) {
; CHECK-LABEL: ld2.nxv32i8_not_multiple_of_2:
; CHECK:      rdvl x[[OFFSET:[0-9]]], #3
; CHECK-NEXT: ld2b { z0.b, z1.b }, p0/z, [x0, x[[OFFSET]]]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 3
%base_ptr = bitcast <vscale x 16 x i8>* %base to i8 *
%res = call <vscale x 32 x i8> @llvm.aarch64.sve.ld2.nxv32i8.nxv16i1.p0i8(<vscale x 16 x i1> %Pg, i8 *%base_ptr)
ret <vscale x 32 x i8> %res
}

define <vscale x 32 x i8> @ld2.nxv32i8_outside_lower_bound(<vscale x 16 x i1> %Pg, <vscale x 16 x i8> *%addr) {
; CHECK-LABEL: ld2.nxv32i8_outside_lower_bound:
; CHECK:      rdvl x[[OFFSET:[0-9]]], #-18
; CHECK-NEXT: ld2b { z0.b, z1.b }, p0/z, [x0, x[[OFFSET]]]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 -18
%base_ptr = bitcast <vscale x 16 x i8>* %base to i8 *
%res = call <vscale x 32 x i8> @llvm.aarch64.sve.ld2.nxv32i8.nxv16i1.p0i8(<vscale x 16 x i1> %Pg, i8 *%base_ptr)
ret <vscale x 32 x i8> %res
}

define <vscale x 32 x i8> @ld2.nxv32i8_outside_upper_bound(<vscale x 16 x i1> %Pg, <vscale x 16 x i8> *%addr) {
; CHECK-LABEL: ld2.nxv32i8_outside_upper_bound:
; CHECK:      rdvl x[[OFFSET:[0-9]]], #16
; CHECK-NEXT: ld2b { z0.b, z1.b }, p0/z, [x0, x[[OFFSET]]]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 16
%base_ptr = bitcast <vscale x 16 x i8>* %base to i8 *
%res = call <vscale x 32 x i8> @llvm.aarch64.sve.ld2.nxv32i8.nxv16i1.p0i8(<vscale x 16 x i1> %Pg, i8 *%base_ptr)
ret <vscale x 32 x i8> %res
}

; ld2h
define <vscale x 16 x i16> @ld2.nxv16i16(<vscale x 8 x i1> %Pg, <vscale x 8 x i16>* %addr) {
; CHECK-LABEL: ld2.nxv16i16:
; CHECK: ld2h { z0.h, z1.h }, p0/z, [x0, #14, mul vl]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 8 x i16>, <vscale x 8 x i16>* %addr, i64 14
%base_ptr = bitcast <vscale x 8 x i16>* %base to i16 *
%res = call <vscale x 16 x i16> @llvm.aarch64.sve.ld2.nxv16i16.nxv8i1.p0i16(<vscale x 8 x i1> %Pg, i16 *%base_ptr)
ret <vscale x 16 x i16> %res
}

define <vscale x 16 x half> @ld2.nxv16f16(<vscale x 8 x i1> %Pg, <vscale x 8 x half>* %addr) {
; CHECK-LABEL: ld2.nxv16f16:
; CHECK: ld2h { z0.h, z1.h }, p0/z, [x0, #-16, mul vl]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 8 x half>, <vscale x 8 x half>* %addr, i64 -16
%base_ptr = bitcast <vscale x 8 x half>* %base to half *
%res = call <vscale x 16 x half> @llvm.aarch64.sve.ld2.nxv16f16.nxv8i1.p0f16(<vscale x 8 x i1> %Pg, half *%base_ptr)
ret <vscale x 16 x half> %res
}

define <vscale x 16 x bfloat> @ld2.nxv16bf16(<vscale x 8 x i1> %Pg, <vscale x 8 x bfloat>* %addr) #0 {
; CHECK-LABEL: ld2.nxv16bf16:
; CHECK: ld2h { z0.h, z1.h }, p0/z, [x0, #12, mul vl]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 8 x bfloat>, <vscale x 8 x bfloat>* %addr, i64 12
%base_ptr = bitcast <vscale x 8 x bfloat>* %base to bfloat *
%res = call <vscale x 16 x bfloat> @llvm.aarch64.sve.ld2.nxv16bf16.nxv8i1.p0bf16(<vscale x 8 x i1> %Pg, bfloat *%base_ptr)
ret <vscale x 16 x bfloat> %res
}

; ld2w
define <vscale x 8 x i32> @ld2.nxv8i32(<vscale x 4 x i1> %Pg, <vscale x 4 x i32>* %addr) {
; CHECK-LABEL: ld2.nxv8i32:
; CHECK: ld2w { z0.s, z1.s }, p0/z, [x0, #14, mul vl]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %addr, i64 14
%base_ptr = bitcast <vscale x 4 x i32>* %base to i32 *
%res = call <vscale x 8 x i32> @llvm.aarch64.sve.ld2.nxv8i32.nxv4i1.p0i32(<vscale x 4 x i1> %Pg, i32 *%base_ptr)
ret <vscale x 8 x i32> %res
}

define <vscale x 8 x float> @ld2.nxv8f32(<vscale x 4 x i1> %Pg, <vscale x 4 x float>* %addr) {
; CHECK-LABEL: ld2.nxv8f32:
; CHECK: ld2w { z0.s, z1.s }, p0/z, [x0, #-16, mul vl]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 4 x float>, <vscale x 4 x float>* %addr, i64 -16
%base_ptr = bitcast <vscale x 4 x float>* %base to float *
%res = call <vscale x 8 x float> @llvm.aarch64.sve.ld2.nxv8f32.nxv4i1.p0f32(<vscale x 4 x i1> %Pg, float *%base_ptr)
ret <vscale x 8 x float> %res
}

; ld2d
define <vscale x 4 x i64> @ld2.nxv4i64(<vscale x 2 x i1> %Pg, <vscale x 2 x i64>* %addr) {
; CHECK-LABEL: ld2.nxv4i64:
; CHECK: ld2d { z0.d, z1.d }, p0/z, [x0, #14, mul vl]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 2 x i64>, <vscale x 2 x i64>* %addr, i64 14
%base_ptr = bitcast <vscale x 2 x i64>* %base to i64 *
%res = call <vscale x 4 x i64> @llvm.aarch64.sve.ld2.nxv4i64.nxv2i1.p0i64(<vscale x 2 x i1> %Pg, i64 *%base_ptr)
ret <vscale x 4 x i64> %res
}

define <vscale x 4 x double> @ld2.nxv4f64(<vscale x 2 x i1> %Pg, <vscale x 2 x double>* %addr) {
; CHECK-LABEL: ld2.nxv4f64:
; CHECK: ld2d { z0.d, z1.d }, p0/z, [x0, #-16, mul vl]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 2 x double>, <vscale x 2 x double>* %addr, i64 -16
%base_ptr = bitcast <vscale x 2 x double>* %base to double *
%res = call <vscale x 4 x double> @llvm.aarch64.sve.ld2.nxv4f64.nxv2i1.p0f64(<vscale x 2 x i1> %Pg, double *%base_ptr)
ret <vscale x 4 x double> %res
}

; ld3b
define <vscale x 48 x i8> @ld3.nxv48i8(<vscale x 16 x i1> %Pg, <vscale x 16 x i8> *%addr) {
; CHECK-LABEL: ld3.nxv48i8:
; CHECK: ld3b { z0.b, z1.b, z2.b }, p0/z, [x0, #3, mul vl]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 3
%base_ptr = bitcast <vscale x 16 x i8>* %base to i8 *
%res = call <vscale x 48 x i8> @llvm.aarch64.sve.ld3.nxv48i8.nxv16i1.p0i8(<vscale x 16 x i1> %Pg, i8 *%base_ptr)
ret <vscale x 48 x i8> %res
}

define <vscale x 48 x i8> @ld3.nxv48i8_lower_bound(<vscale x 16 x i1> %Pg, <vscale x 16 x i8> *%addr) {
; CHECK-LABEL: ld3.nxv48i8_lower_bound:
; CHECK: ld3b { z0.b, z1.b, z2.b }, p0/z, [x0, #-24, mul vl]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 -24
%base_ptr = bitcast <vscale x 16 x i8>* %base to i8 *
%res = call <vscale x 48 x i8> @llvm.aarch64.sve.ld3.nxv48i8.nxv16i1.p0i8(<vscale x 16 x i1> %Pg, i8 *%base_ptr)
ret <vscale x 48 x i8> %res
}

define <vscale x 48 x i8> @ld3.nxv48i8_upper_bound(<vscale x 16 x i1> %Pg, <vscale x 16 x i8> *%addr) {
; CHECK-LABEL: ld3.nxv48i8_upper_bound:
; CHECK: ld3b { z0.b, z1.b, z2.b }, p0/z, [x0, #21, mul vl]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 21
%base_ptr = bitcast <vscale x 16 x i8>* %base to i8 *
%res = call <vscale x 48 x i8> @llvm.aarch64.sve.ld3.nxv48i8.nxv16i1.p0i8(<vscale x 16 x i1> %Pg, i8 *%base_ptr)
ret <vscale x 48 x i8> %res
}

define <vscale x 48 x i8> @ld3.nxv48i8_not_multiple_of_3_01(<vscale x 16 x i1> %Pg, <vscale x 16 x i8> *%addr) {
; CHECK-LABEL: ld3.nxv48i8_not_multiple_of_3_01:
; CHECK:      rdvl x[[OFFSET:[0-9]]], #4
; CHECK-NEXT: ld3b { z0.b, z1.b, z2.b }, p0/z, [x0, x[[OFFSET]]]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 4
%base_ptr = bitcast <vscale x 16 x i8>* %base to i8 *
%res = call <vscale x 48 x i8> @llvm.aarch64.sve.ld3.nxv48i8.nxv16i1.p0i8(<vscale x 16 x i1> %Pg, i8 *%base_ptr)
ret <vscale x 48 x i8> %res
}

define <vscale x 48 x i8> @ld3.nxv48i8_not_multiple_of_3_02(<vscale x 16 x i1> %Pg, <vscale x 16 x i8> *%addr) {
; CHECK-LABEL: ld3.nxv48i8_not_multiple_of_3_02:
; CHECK:      rdvl x[[OFFSET:[0-9]]], #5
; CHECK-NEXT: ld3b { z0.b, z1.b, z2.b }, p0/z, [x0, x[[OFFSET]]]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 5
%base_ptr = bitcast <vscale x 16 x i8>* %base to i8 *
%res = call <vscale x 48 x i8> @llvm.aarch64.sve.ld3.nxv48i8.nxv16i1.p0i8(<vscale x 16 x i1> %Pg, i8 *%base_ptr)
ret <vscale x 48 x i8> %res
}

define <vscale x 48 x i8> @ld3.nxv48i8_outside_lower_bound(<vscale x 16 x i1> %Pg, <vscale x 16 x i8> *%addr) {
; CHECK-LABEL: ld3.nxv48i8_outside_lower_bound:
; CHECK:      rdvl x[[OFFSET:[0-9]]], #-27
; CHECK-NEXT: ld3b { z0.b, z1.b, z2.b }, p0/z, [x0, x[[OFFSET]]]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 -27
%base_ptr = bitcast <vscale x 16 x i8>* %base to i8 *
%res = call <vscale x 48 x i8> @llvm.aarch64.sve.ld3.nxv48i8.nxv16i1.p0i8(<vscale x 16 x i1> %Pg, i8 *%base_ptr)
ret <vscale x 48 x i8> %res
}

define <vscale x 48 x i8> @ld3.nxv48i8_outside_upper_bound(<vscale x 16 x i1> %Pg, <vscale x 16 x i8> *%addr) {
; CHECK-LABEL: ld3.nxv48i8_outside_upper_bound:
; CHECK:      rdvl x[[OFFSET:[0-9]]], #24
; CHECK-NEXT: ld3b { z0.b, z1.b, z2.b }, p0/z, [x0, x[[OFFSET]]]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 24
%base_ptr = bitcast <vscale x 16 x i8>* %base to i8 *
%res = call <vscale x 48 x i8> @llvm.aarch64.sve.ld3.nxv48i8.nxv16i1.p0i8(<vscale x 16 x i1> %Pg, i8 *%base_ptr)
ret <vscale x 48 x i8> %res
}

; ld3h
define <vscale x 24 x i16> @ld3.nxv24i16(<vscale x 8 x i1> %Pg, <vscale x 8 x i16> *%addr) {
; CHECK-LABEL: ld3.nxv24i16:
; CHECK: ld3h { z0.h, z1.h, z2.h }, p0/z, [x0, #21, mul vl]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 8 x i16>, <vscale x 8 x i16>* %addr, i64 21
%base_ptr = bitcast <vscale x 8 x i16>* %base to i16 *
%res = call <vscale x 24 x i16> @llvm.aarch64.sve.ld3.nxv24i16.nxv8i1.p0i16(<vscale x 8 x i1> %Pg, i16 *%base_ptr)
ret <vscale x 24 x i16> %res
}

define <vscale x 24 x half> @ld3.nxv24f16(<vscale x 8 x i1> %Pg, <vscale x 8 x half> *%addr) {
; CHECK-LABEL: ld3.nxv24f16:
; CHECK: ld3h { z0.h, z1.h, z2.h }, p0/z, [x0, #21, mul vl]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 8 x half>, <vscale x 8 x half>* %addr, i64 21
%base_ptr = bitcast <vscale x 8 x half>* %base to half *
%res = call <vscale x 24 x half> @llvm.aarch64.sve.ld3.nxv24f16.nxv8i1.p0f16(<vscale x 8 x i1> %Pg, half *%base_ptr)
ret <vscale x 24 x half> %res
}

define <vscale x 24 x bfloat> @ld3.nxv24bf16(<vscale x 8 x i1> %Pg, <vscale x 8 x bfloat> *%addr) #0 {
; CHECK-LABEL: ld3.nxv24bf16:
; CHECK: ld3h { z0.h, z1.h, z2.h }, p0/z, [x0, #-24, mul vl]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 8 x bfloat>, <vscale x 8 x bfloat>* %addr, i64 -24
%base_ptr = bitcast <vscale x 8 x bfloat>* %base to bfloat *
%res = call <vscale x 24 x bfloat> @llvm.aarch64.sve.ld3.nxv24bf16.nxv8i1.p0bf16(<vscale x 8 x i1> %Pg, bfloat *%base_ptr)
ret <vscale x 24 x bfloat> %res
}

; ld3w
define <vscale x 12 x i32> @ld3.nxv12i32(<vscale x 4 x i1> %Pg, <vscale x 4 x i32> *%addr) {
; CHECK-LABEL: ld3.nxv12i32:
; CHECK: ld3w { z0.s, z1.s, z2.s }, p0/z, [x0, #21, mul vl]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %addr, i64 21
%base_ptr = bitcast <vscale x 4 x i32>* %base to i32 *
%res = call <vscale x 12 x i32> @llvm.aarch64.sve.ld3.nxv12i32.nxv4i1.p0i32(<vscale x 4 x i1> %Pg, i32 *%base_ptr)
ret <vscale x 12 x i32> %res
}

define <vscale x 12 x float> @ld3.nxv12f32(<vscale x 4 x i1> %Pg, <vscale x 4 x float> *%addr) {
; CHECK-LABEL: ld3.nxv12f32:
; CHECK: ld3w { z0.s, z1.s, z2.s }, p0/z, [x0, #-24, mul vl]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 4 x float>, <vscale x 4 x float>* %addr, i64 -24
%base_ptr = bitcast <vscale x 4 x float>* %base to float *
%res = call <vscale x 12 x float> @llvm.aarch64.sve.ld3.nxv12f32.nxv4i1.p0f32(<vscale x 4 x i1> %Pg, float *%base_ptr)
ret <vscale x 12 x float> %res
}

; ld3d
define <vscale x 6 x i64> @ld3.nxv6i64(<vscale x 2 x i1> %Pg, <vscale x 2 x i64> *%addr) {
; CHECK-LABEL: ld3.nxv6i64:
; CHECK: ld3d { z0.d, z1.d, z2.d }, p0/z, [x0, #21, mul vl]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 2 x i64>, <vscale x 2 x i64>* %addr, i64 21
%base_ptr = bitcast <vscale x 2 x i64>* %base to i64 *
%res = call <vscale x 6 x i64> @llvm.aarch64.sve.ld3.nxv6i64.nxv2i1.p0i64(<vscale x 2 x i1> %Pg, i64 *%base_ptr)
ret <vscale x 6 x i64> %res
}

define <vscale x 6 x double> @ld3.nxv6f64(<vscale x 2 x i1> %Pg, <vscale x 2 x double> *%addr) {
; CHECK-LABEL: ld3.nxv6f64:
; CHECK: ld3d { z0.d, z1.d, z2.d }, p0/z, [x0, #-24, mul vl]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 2 x double>, <vscale x 2 x double>* %addr, i64 -24
%base_ptr = bitcast <vscale x 2 x double>* %base to double *
%res = call <vscale x 6 x double> @llvm.aarch64.sve.ld3.nxv6f64.nxv2i1.p0f64(<vscale x 2 x i1> %Pg, double *%base_ptr)
ret <vscale x 6 x double> %res
}

; ; ld4b
define <vscale x 64 x i8> @ld4.nxv64i8(<vscale x 16 x i1> %Pg, <vscale x 16 x i8> *%addr) {
; CHECK-LABEL: ld4.nxv64i8:
; CHECK: ld4b { z0.b, z1.b, z2.b, z3.b }, p0/z, [x0, #4, mul vl]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 4
%base_ptr = bitcast <vscale x 16 x i8>* %base to i8 *
%res = call <vscale x 64 x i8> @llvm.aarch64.sve.ld4.nxv64i8.nxv16i1.p0i8(<vscale x 16 x i1> %Pg, i8 *%base_ptr)
ret <vscale x 64 x i8> %res
}

define <vscale x 64 x i8> @ld4.nxv64i8_lower_bound(<vscale x 16 x i1> %Pg, <vscale x 16 x i8> *%addr) {
; CHECK-LABEL: ld4.nxv64i8_lower_bound:
; CHECK: ld4b { z0.b, z1.b, z2.b, z3.b }, p0/z, [x0, #-32, mul vl]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 -32
%base_ptr = bitcast <vscale x 16 x i8>* %base to i8 *
%res = call <vscale x 64 x i8> @llvm.aarch64.sve.ld4.nxv64i8.nxv16i1.p0i8(<vscale x 16 x i1> %Pg, i8 *%base_ptr)
ret <vscale x 64 x i8> %res
}

define <vscale x 64 x i8> @ld4.nxv64i8_upper_bound(<vscale x 16 x i1> %Pg, <vscale x 16 x i8> *%addr) {
; CHECK-LABEL: ld4.nxv64i8_upper_bound:
; CHECK: ld4b { z0.b, z1.b, z2.b, z3.b }, p0/z, [x0, #28, mul vl]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 28
%base_ptr = bitcast <vscale x 16 x i8>* %base to i8 *
%res = call <vscale x 64 x i8> @llvm.aarch64.sve.ld4.nxv64i8.nxv16i1.p0i8(<vscale x 16 x i1> %Pg, i8 *%base_ptr)
ret <vscale x 64 x i8> %res
}

define <vscale x 64 x i8> @ld4.nxv64i8_not_multiple_of_4_01(<vscale x 16 x i1> %Pg, <vscale x 16 x i8> *%addr) {
; CHECK-LABEL: ld4.nxv64i8_not_multiple_of_4_01:
; CHECK:      rdvl x[[OFFSET:[0-9]]], #5
; CHECK-NEXT: ld4b { z0.b, z1.b, z2.b, z3.b }, p0/z, [x0, x[[OFFSET]]]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 5
%base_ptr = bitcast <vscale x 16 x i8>* %base to i8 *
%res = call <vscale x 64 x i8> @llvm.aarch64.sve.ld4.nxv64i8.nxv16i1.p0i8(<vscale x 16 x i1> %Pg, i8 *%base_ptr)
ret <vscale x 64 x i8> %res
}

define <vscale x 64 x i8> @ld4.nxv64i8_not_multiple_of_4_02(<vscale x 16 x i1> %Pg, <vscale x 16 x i8> *%addr) {
; CHECK-LABEL: ld4.nxv64i8_not_multiple_of_4_02:
; CHECK:      rdvl x[[OFFSET:[0-9]]], #6
; CHECK-NEXT: ld4b { z0.b, z1.b, z2.b, z3.b }, p0/z, [x0, x[[OFFSET]]]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 6
%base_ptr = bitcast <vscale x 16 x i8>* %base to i8 *
%res = call <vscale x 64 x i8> @llvm.aarch64.sve.ld4.nxv64i8.nxv16i1.p0i8(<vscale x 16 x i1> %Pg, i8 *%base_ptr)
ret <vscale x 64 x i8> %res
}

define <vscale x 64 x i8> @ld4.nxv64i8_not_multiple_of_4_03(<vscale x 16 x i1> %Pg, <vscale x 16 x i8> *%addr) {
; CHECK-LABEL: ld4.nxv64i8_not_multiple_of_4_03:
; CHECK:      rdvl x[[OFFSET:[0-9]]], #7
; CHECK-NEXT: ld4b { z0.b, z1.b, z2.b, z3.b }, p0/z, [x0, x[[OFFSET]]]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 7
%base_ptr = bitcast <vscale x 16 x i8>* %base to i8 *
%res = call <vscale x 64 x i8> @llvm.aarch64.sve.ld4.nxv64i8.nxv16i1.p0i8(<vscale x 16 x i1> %Pg, i8 *%base_ptr)
ret <vscale x 64 x i8> %res
}

define <vscale x 64 x i8> @ld4.nxv64i8_outside_lower_bound(<vscale x 16 x i1> %Pg, <vscale x 16 x i8> *%addr) {
; CHECK-LABEL: ld4.nxv64i8_outside_lower_bound:
; FIXME: optimize OFFSET computation so that xOFFSET = (mul (RDVL #4) #9)
; xM = -9 * 2^6
; xP = RDVL * 2^-4 
; xOFFSET = RDVL * 2^-4 * -9 * 2^6 = RDVL * -36
; CHECK:      rdvl x[[N:[0-9]]], #1
; CHECK-DAG:  mov  x[[M:[0-9]]], #-576
; CHECK-DAG:  lsr  x[[P:[0-9]]], x[[N]], #4
; CHECK-DAG:  mul  x[[OFFSET:[0-9]]], x[[P]], x[[M]]
; CHECK-NEXT: ld4b { z0.b, z1.b, z2.b, z3.b }, p0/z, [x0, x[[OFFSET]]]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 -36
%base_ptr = bitcast <vscale x 16 x i8>* %base to i8 *
%res = call <vscale x 64 x i8> @llvm.aarch64.sve.ld4.nxv64i8.nxv16i1.p0i8(<vscale x 16 x i1> %Pg, i8 *%base_ptr)
ret <vscale x 64 x i8> %res
}

define <vscale x 64 x i8> @ld4.nxv64i8_outside_upper_bound(<vscale x 16 x i1> %Pg, <vscale x 16 x i8> *%addr) {
; CHECK-LABEL: ld4.nxv64i8_outside_upper_bound:
; FIXME: optimize OFFSET computation so that xOFFSET = (mul (RDVL #16) #2)
; xM = 2^9
; xP = RDVL * 2^-4
; xOFFSET = RDVL * 2^-4 * 2^9 = RDVL * 32
; CHECK:      rdvl x[[N:[0-9]]], #1
; CHECK-DAG:  mov  w[[M:[0-9]]], #512
; CHECK-DAG:  lsr  x[[P:[0-9]]], x[[N]], #4
; CHECK-DAG:  mul  x[[OFFSET:[0-9]]], x[[P]], x[[M]]
; CHECK-NEXT: ld4b { z0.b, z1.b, z2.b, z3.b }, p0/z, [x0, x[[OFFSET]]]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %addr, i64 32
%base_ptr = bitcast <vscale x 16 x i8>* %base to i8 *
%res = call <vscale x 64 x i8> @llvm.aarch64.sve.ld4.nxv64i8.nxv16i1.p0i8(<vscale x 16 x i1> %Pg, i8 *%base_ptr)
ret <vscale x 64 x i8> %res
}

; ld4h
define <vscale x 32 x i16> @ld4.nxv32i16(<vscale x 8 x i1> %Pg, <vscale x 8 x i16> *%addr) {
; CHECK-LABEL: ld4.nxv32i16:
; CHECK: ld4h { z0.h, z1.h, z2.h, z3.h }, p0/z, [x0, #8, mul vl]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 8 x i16>, <vscale x 8 x i16>* %addr, i64 8
%base_ptr = bitcast <vscale x 8 x i16>* %base to i16 *
%res = call <vscale x 32 x i16> @llvm.aarch64.sve.ld4.nxv32i16.nxv8i1.p0i16(<vscale x 8 x i1> %Pg, i16 *%base_ptr)
ret <vscale x 32 x i16> %res
}

define <vscale x 32 x half> @ld4.nxv32f16(<vscale x 8 x i1> %Pg, <vscale x 8 x half> *%addr) {
; CHECK-LABEL: ld4.nxv32f16:
; CHECK: ld4h { z0.h, z1.h, z2.h, z3.h }, p0/z, [x0, #28, mul vl]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 8 x half>, <vscale x 8 x half>* %addr, i64 28
%base_ptr = bitcast <vscale x 8 x half>* %base to half *
%res = call <vscale x 32 x half> @llvm.aarch64.sve.ld4.nxv32f16.nxv8i1.p0f16(<vscale x 8 x i1> %Pg, half *%base_ptr)
ret <vscale x 32 x half> %res
}

define <vscale x 32 x bfloat> @ld4.nxv32bf16(<vscale x 8 x i1> %Pg, <vscale x 8 x bfloat> *%addr) #0 {
; CHECK-LABEL: ld4.nxv32bf16:
; CHECK: ld4h { z0.h, z1.h, z2.h, z3.h }, p0/z, [x0, #-32, mul vl]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 8 x bfloat>, <vscale x 8 x bfloat>* %addr, i64 -32
%base_ptr = bitcast <vscale x 8 x bfloat>* %base to bfloat *
%res = call <vscale x 32 x bfloat> @llvm.aarch64.sve.ld4.nxv32bf16.nxv8i1.p0bf16(<vscale x 8 x i1> %Pg, bfloat *%base_ptr)
ret <vscale x 32 x bfloat> %res
}

; ld4w
define <vscale x 16 x i32> @ld4.nxv16i32(<vscale x 4 x i1> %Pg, <vscale x 4 x i32> *%addr) {
; CHECK-LABEL: ld4.nxv16i32:
; CHECK: ld4w { z0.s, z1.s, z2.s, z3.s }, p0/z, [x0, #28, mul vl]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %addr, i64 28
%base_ptr = bitcast <vscale x 4 x i32>* %base to i32 *
%res = call <vscale x 16 x i32> @llvm.aarch64.sve.ld4.nxv16i32.nxv4i1.p0i32(<vscale x 4 x i1> %Pg, i32 *%base_ptr)
ret <vscale x 16 x i32> %res
}

define <vscale x 16 x float> @ld4.nxv16f32(<vscale x 4 x i1> %Pg, <vscale x 4 x float>* %addr) {
; CHECK-LABEL: ld4.nxv16f32:
; CHECK: ld4w { z0.s, z1.s, z2.s, z3.s }, p0/z, [x0, #-32, mul vl]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 4 x float>, <vscale x 4 x float>* %addr, i64 -32
%base_ptr = bitcast <vscale x 4 x float>* %base to float *
%res = call <vscale x 16 x float> @llvm.aarch64.sve.ld4.nxv16f32.nxv4i1.p0f32(<vscale x 4 x i1> %Pg, float *%base_ptr)
ret <vscale x 16 x float> %res
}

; ld4d
define <vscale x 8 x i64> @ld4.nxv8i64(<vscale x 2 x i1> %Pg, <vscale x 2 x i64> *%addr) {
; CHECK-LABEL: ld4.nxv8i64:
; CHECK: ld4d { z0.d, z1.d, z2.d, z3.d }, p0/z, [x0, #28, mul vl]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 2 x i64>, <vscale x 2 x i64>* %addr, i64 28
%base_ptr = bitcast <vscale x 2 x i64>* %base to i64 *
%res = call <vscale x 8 x i64> @llvm.aarch64.sve.ld4.nxv8i64.nxv2i1.p0i64(<vscale x 2 x i1> %Pg, i64 *%base_ptr)
ret <vscale x 8 x i64> %res
}

define <vscale x 8 x double> @ld4.nxv8f64(<vscale x 2 x i1> %Pg, <vscale x 2 x double> *%addr) {
; CHECK-LABEL: ld4.nxv8f64:
; CHECK: ld4d { z0.d, z1.d, z2.d, z3.d }, p0/z, [x0, #-32, mul vl]
; CHECK-NEXT: ret
%base = getelementptr <vscale x 2 x double>, <vscale x 2 x double>* %addr, i64 -32
%base_ptr = bitcast <vscale x 2 x double>* %base to double *
%res = call <vscale x 8 x double> @llvm.aarch64.sve.ld4.nxv8f64.nxv2i1.p0f64(<vscale x 2 x i1> %Pg, double * %base_ptr)
ret <vscale x 8 x double> %res
}

declare <vscale x 32 x i8> @llvm.aarch64.sve.ld2.nxv32i8.nxv16i1.p0i8(<vscale x 16 x i1>, i8*)
declare <vscale x 16 x i16> @llvm.aarch64.sve.ld2.nxv16i16.nxv8i1.p0i16(<vscale x 8 x i1>, i16*)
declare <vscale x 8 x i32> @llvm.aarch64.sve.ld2.nxv8i32.nxv4i1.p0i32(<vscale x 4 x i1>, i32*)
declare <vscale x 4 x i64> @llvm.aarch64.sve.ld2.nxv4i64.nxv2i1.p0i64(<vscale x 2 x i1>, i64*)
declare <vscale x 16 x half> @llvm.aarch64.sve.ld2.nxv16f16.nxv8i1.p0f16(<vscale x 8 x i1>, half*)
declare <vscale x 16 x bfloat> @llvm.aarch64.sve.ld2.nxv16bf16.nxv8i1.p0bf16(<vscale x 8 x i1>, bfloat*)
declare <vscale x 8 x float> @llvm.aarch64.sve.ld2.nxv8f32.nxv4i1.p0f32(<vscale x 4 x i1>, float*)
declare <vscale x 4 x double> @llvm.aarch64.sve.ld2.nxv4f64.nxv2i1.p0f64(<vscale x 2 x i1>, double*)

declare <vscale x 48 x i8> @llvm.aarch64.sve.ld3.nxv48i8.nxv16i1.p0i8(<vscale x 16 x i1>, i8*)
declare <vscale x 24 x i16> @llvm.aarch64.sve.ld3.nxv24i16.nxv8i1.p0i16(<vscale x 8 x i1>, i16*)
declare <vscale x 12 x i32> @llvm.aarch64.sve.ld3.nxv12i32.nxv4i1.p0i32(<vscale x 4 x i1>, i32*)
declare <vscale x 6 x i64> @llvm.aarch64.sve.ld3.nxv6i64.nxv2i1.p0i64(<vscale x 2 x i1>, i64*)
declare <vscale x 24 x half> @llvm.aarch64.sve.ld3.nxv24f16.nxv8i1.p0f16(<vscale x 8 x i1>, half*)
declare <vscale x 24 x bfloat> @llvm.aarch64.sve.ld3.nxv24bf16.nxv8i1.p0bf16(<vscale x 8 x i1>, bfloat*)
declare <vscale x 12 x float> @llvm.aarch64.sve.ld3.nxv12f32.nxv4i1.p0f32(<vscale x 4 x i1>, float*)
declare <vscale x 6 x double> @llvm.aarch64.sve.ld3.nxv6f64.nxv2i1.p0f64(<vscale x 2 x i1>, double*)

declare <vscale x 64 x i8> @llvm.aarch64.sve.ld4.nxv64i8.nxv16i1.p0i8(<vscale x 16 x i1>, i8*)
declare <vscale x 32 x i16> @llvm.aarch64.sve.ld4.nxv32i16.nxv8i1.p0i16(<vscale x 8 x i1>, i16*)
declare <vscale x 16 x i32> @llvm.aarch64.sve.ld4.nxv16i32.nxv4i1.p0i32(<vscale x 4 x i1>, i32*)
declare <vscale x 8 x i64> @llvm.aarch64.sve.ld4.nxv8i64.nxv2i1.p0i64(<vscale x 2 x i1>, i64*)
declare <vscale x 32 x half> @llvm.aarch64.sve.ld4.nxv32f16.nxv8i1.p0f16(<vscale x 8 x i1>, half*)
declare <vscale x 32 x bfloat> @llvm.aarch64.sve.ld4.nxv32bf16.nxv8i1.p0bf16(<vscale x 8 x i1>, bfloat*)
declare <vscale x 16 x float> @llvm.aarch64.sve.ld4.nxv16f32.nxv4i1.p0f32(<vscale x 4 x i1>, float*)
declare <vscale x 8 x double> @llvm.aarch64.sve.ld4.nxv8f64.nxv2i1.p0f64(<vscale x 2 x i1>, double*)

; +bf16 is required for the bfloat version.
attributes #0 = { "target-features"="+sve,+bf16" }
