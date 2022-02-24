; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

;
; LDFF1B, LDFF1W, LDFF1H, LDFF1D: vector base + immediate offset (index)
;   e.g. ldff1h { z0.s }, p0/z, [z0.s, #16]
;

; LDFF1B
define <vscale x 4 x i32> @gldff1b_s_imm_offset(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %base) {
; CHECK-LABEL: gldff1b_s_imm_offset:
; CHECK: ldff1b { z0.s }, p0/z, [z0.s, #16]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i8> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv4i8.nxv4i32(<vscale x 4 x i1> %pg,
                                                                                             <vscale x 4 x i32> %base,
                                                                                             i64 16)
  %res = zext <vscale x 4 x i8> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @gldff1b_d_imm_offset(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %base) {
; CHECK-LABEL: gldff1b_d_imm_offset:
; CHECK: ldff1b { z0.d }, p0/z, [z0.d, #16]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i8> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv2i8.nxv2i64(<vscale x 2 x i1> %pg,
                                                                                             <vscale x 2 x i64> %base,
                                                                                             i64 16)
  %res = zext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

; LDFF1H
define <vscale x 4 x i32> @gldff1h_s_imm_offset(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %base) {
; CHECK-LABEL: gldff1h_s_imm_offset:
; CHECK: ldff1h { z0.s }, p0/z, [z0.s, #16]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i16> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv4i16.nxv4i32(<vscale x 4 x i1> %pg,
                                                                                               <vscale x 4 x i32> %base,
                                                                                               i64 16)
  %res = zext <vscale x 4 x i16> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @gldff1h_d_imm_offset(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %base) {
; CHECK-LABEL: gldff1h_d_imm_offset:
; CHECK: ldff1h { z0.d }, p0/z, [z0.d, #16]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv2i16.nxv2i64(<vscale x 2 x i1> %pg,
                                                                                               <vscale x 2 x i64> %base,
                                                                                               i64 16)
  %res = zext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

; LDFF1W
define <vscale x 4 x i32> @gldff1w_s_imm_offset(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %base) {
; CHECK-LABEL: gldff1w_s_imm_offset:
; CHECK: ldff1w { z0.s }, p0/z, [z0.s, #16]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i32> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv4i32.nxv4i32(<vscale x 4 x i1> %pg,
                                                                                               <vscale x 4 x i32> %base,
                                                                                               i64 16)
  ret <vscale x 4 x i32> %load
}

define <vscale x 2 x i64> @gldff1w_d_imm_offset(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %base) {
; CHECK-LABEL: gldff1w_d_imm_offset:
; CHECK: ldff1w { z0.d }, p0/z, [z0.d, #16]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv2i32.nxv2i64(<vscale x 2 x i1> %pg,
                                                                                               <vscale x 2 x i64> %base,
                                                                                               i64 16)
  %res = zext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 4 x float> @gldff1w_s_imm_offset_float(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %base) {
; CHECK-LABEL: gldff1w_s_imm_offset_float:
; CHECK: ldff1w { z0.s }, p0/z, [z0.s, #16]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x float> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv4f32.nxv4i32(<vscale x 4 x i1> %pg,
                                                                                                 <vscale x 4 x i32> %base,
                                                                                                 i64 16)
  ret <vscale x 4 x float> %load
}

; LDFF1D
define <vscale x 2 x i64> @gldff1d_d_imm_offset(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %base) {
; CHECK-LABEL: gldff1d_d_imm_offset:
; CHECK: ldff1d { z0.d }, p0/z, [z0.d, #16]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i64> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv2i64.nxv2i64(<vscale x 2 x i1> %pg,
                                                                                               <vscale x 2 x i64> %base,
                                                                                               i64 16)
  ret <vscale x 2 x i64> %load
}

define <vscale x 2 x double> @gldff1d_d_imm_offset_double(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %base) {
; CHECK-LABEL: gldff1d_d_imm_offset_double:
; CHECK: ldff1d { z0.d }, p0/z, [z0.d, #16]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x double> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv2f64.nxv2i64(<vscale x 2 x i1> %pg,
                                                                                                  <vscale x 2 x i64> %base,
                                                                                                  i64 16)
  ret <vscale x 2 x double> %load
}

;
; LDFF1SB, LDFF1SW, LDFF1SH: vector base + immediate offset (index)
;   e.g. ldff1sh { z0.s }, p0/z, [z0.s, #16]
;

; LDFF1SB
define <vscale x 4 x i32> @gldff1sb_s_imm_offset(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %base) {
; CHECK-LABEL: gldff1sb_s_imm_offset:
; CHECK: ldff1sb { z0.s }, p0/z, [z0.s, #16]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i8> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv4i8.nxv4i32(<vscale x 4 x i1> %pg,
                                                                                             <vscale x 4 x i32> %base,
                                                                                             i64 16)
  %res = sext <vscale x 4 x i8> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @gldff1sb_d_imm_offset(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %base) {
; CHECK-LABEL: gldff1sb_d_imm_offset:
; CHECK: ldff1sb { z0.d }, p0/z, [z0.d, #16]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i8> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv2i8.nxv2i64(<vscale x 2 x i1> %pg,
                                                                                             <vscale x 2 x i64> %base,
                                                                                             i64 16)
  %res = sext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

; LDFF1SH
define <vscale x 4 x i32> @gldff1sh_s_imm_offset(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %base) {
; CHECK-LABEL: gldff1sh_s_imm_offset:
; CHECK: ldff1sh { z0.s }, p0/z, [z0.s, #16]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i16> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv4i16.nxv4i32(<vscale x 4 x i1> %pg,
                                                                                               <vscale x 4 x i32> %base,
                                                                                               i64 16)
  %res = sext <vscale x 4 x i16> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @gldff1sh_d_imm_offset(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %base) {
; CHECK-LABEL: gldff1sh_d_imm_offset:
; CHECK: ldff1sh { z0.d }, p0/z, [z0.d, #16]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv2i16.nxv2i64(<vscale x 2 x i1> %pg,
                                                                                               <vscale x 2 x i64> %base,
                                                                                               i64 16)
  %res = sext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

; LDFF1SW
define <vscale x 2 x i64> @gldff1sw_d_imm_offset(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %base) {
; CHECK-LABEL: gldff1sw_d_imm_offset:
; CHECK: ldff1sw { z0.d }, p0/z, [z0.d, #16]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv2i32.nxv2i64(<vscale x 2 x i1> %pg,
                                                                                               <vscale x 2 x i64> %base,
                                                                                               i64 16)
  %res = sext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

;
; LDFF1B, LDFF1W, LDFF1H, LDFF1D: vector base + out of range immediate offset
;   e.g. ldff1b { z0.d }, p0/z, [x0, z0.d]
;

; LDFF1B
define <vscale x 4 x i32> @gldff1b_s_imm_offset_out_of_range(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %base) {
; CHECK-LABEL: gldff1b_s_imm_offset_out_of_range:
; CHECK: mov	w8, #32
; CHECK-NEXT: ldff1b { z0.s }, p0/z, [x8, z0.s, uxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i8> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv4i8.nxv4i32(<vscale x 4 x i1> %pg,
                                                                                             <vscale x 4 x i32> %base,
                                                                                             i64 32)
  %res = zext <vscale x 4 x i8> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @gldff1b_d_imm_offset_out_of_range(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %base) {
; CHECK-LABEL: gldff1b_d_imm_offset_out_of_range:
; CHECK: mov	w8, #32
; CHECK-NEXT: ldff1b { z0.d }, p0/z, [x8, z0.d]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i8> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv2i8.nxv2i64(<vscale x 2 x i1> %pg,
                                                                                             <vscale x 2 x i64> %base,
                                                                                             i64 32)
  %res = zext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

; LDFF1H
define <vscale x 4 x i32> @gldff1h_s_imm_offset_out_of_range(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %base) {
; CHECK-LABEL: gldff1h_s_imm_offset_out_of_range:
; CHECK: mov	w8, #63
; CHECK-NEXT: ldff1h { z0.s }, p0/z, [x8, z0.s, uxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i16> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv4i16.nxv4i32(<vscale x 4 x i1> %pg,
                                                                                               <vscale x 4 x i32> %base,
                                                                                               i64 63)
  %res = zext <vscale x 4 x i16> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @gldff1h_d_imm_offset_out_of_range(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %base) {
; CHECK-LABEL: gldff1h_d_imm_offset_out_of_range:
; CHECK: mov	w8, #63
; CHECK-NEXT: ldff1h { z0.d }, p0/z, [x8, z0.d]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv2i16.nxv2i64(<vscale x 2 x i1> %pg,
                                                                                               <vscale x 2 x i64> %base,
                                                                                               i64 63)
  %res = zext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

; LDFF1W
define <vscale x 4 x i32> @gldff1w_s_imm_offset_out_of_range(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %base) {
; CHECK-LABEL: gldff1w_s_imm_offset_out_of_range:
; CHECK: mov	w8, #125
; CHECK-NEXT: ldff1w { z0.s }, p0/z, [x8, z0.s, uxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i32> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv4i32.nxv4i32(<vscale x 4 x i1> %pg,
                                                                                               <vscale x 4 x i32> %base,
                                                                                               i64 125)
  ret <vscale x 4 x i32> %load
}

define <vscale x 2 x i64> @gldff1w_d_imm_offset_out_of_range(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %base) {
; CHECK-LABEL: gldff1w_d_imm_offset_out_of_range:
; CHECK: mov	w8, #125
; CHECK-NEXT: ldff1w { z0.d }, p0/z, [x8, z0.d]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv2i32.nxv2i64(<vscale x 2 x i1> %pg,
                                                                                               <vscale x 2 x i64> %base,
                                                                                               i64 125)
  %res = zext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 4 x float> @gldff1w_s_imm_offset_out_of_range_float(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %base) {
; CHECK-LABEL: gldff1w_s_imm_offset_out_of_range_float:
; CHECK: mov	w8, #125
; CHECK-NEXT: ldff1w { z0.s }, p0/z, [x8, z0.s, uxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x float> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv4f32.nxv4i32(<vscale x 4 x i1> %pg,
                                                                                                 <vscale x 4 x i32> %base,
                                                                                                 i64 125)
  ret <vscale x 4 x float> %load
}

; LDFF1D
define <vscale x 2 x i64> @gldff1d_d_imm_offset_out_of_range(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %base) {
; CHECK-LABEL: gldff1d_d_imm_offset_out_of_range:
; CHECK: mov	w8, #249
; CHECK-NEXT: ldff1d { z0.d }, p0/z, [x8, z0.d]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i64> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv2i64.nxv2i64(<vscale x 2 x i1> %pg,
                                                                                               <vscale x 2 x i64> %base,
                                                                                               i64 249)
  ret <vscale x 2 x i64> %load
}

define <vscale x 2 x double> @gldff1d_d_imm_offset_out_of_range_double(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %base) {
; CHECK-LABEL: gldff1d_d_imm_offset_out_of_range_double:
; CHECK: mov	w8, #249
; CHECK-NEXT: ldff1d { z0.d }, p0/z, [x8, z0.d]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x double> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv2f64.nxv2i64(<vscale x 2 x i1> %pg,
                                                                                                  <vscale x 2 x i64> %base,
                                                                                                  i64 249)
  ret <vscale x 2 x double> %load
}

;
; LDFF1SB, LDFF1SW, LDFF1SH: vector base + out of range immediate offset
;     e.g. ldff1sb { z0.s }, p0/z, [x8, z0.s, uxtw]
;

; LDFF1SB
define <vscale x 4 x i32> @gldff1sb_s_imm_offset_out_of_range(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %base) {
; CHECK-LABEL: gldff1sb_s_imm_offset_out_of_range:
; CHECK: mov	w8, #32
; CHECK-NEXT: ldff1sb { z0.s }, p0/z, [x8, z0.s, uxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i8> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv4i8.nxv4i32(<vscale x 4 x i1> %pg,
                                                                                             <vscale x 4 x i32> %base,
                                                                                             i64 32)
  %res = sext <vscale x 4 x i8> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @gldff1sb_d_imm_offset_out_of_range(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %base) {
; CHECK-LABEL: gldff1sb_d_imm_offset_out_of_range:
; CHECK: mov	w8, #32
; CHECK-NEXT: ldff1sb { z0.d }, p0/z, [x8, z0.d]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i8> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv2i8.nxv2i64(<vscale x 2 x i1> %pg,
                                                                                             <vscale x 2 x i64> %base,
                                                                                             i64 32)
  %res = sext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

; LDFF1SH
define <vscale x 4 x i32> @gldff1sh_s_imm_offset_out_of_range(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %base) {
; CHECK-LABEL: gldff1sh_s_imm_offset_out_of_range:
; CHECK: mov	w8, #63
; CHECK-NEXT: ldff1sh { z0.s }, p0/z, [x8, z0.s, uxtw]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i16> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv4i16.nxv4i32(<vscale x 4 x i1> %pg,
                                                                                               <vscale x 4 x i32> %base,
                                                                                               i64 63)
  %res = sext <vscale x 4 x i16> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

define <vscale x 2 x i64> @gldff1sh_d_imm_offset_out_of_range(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %base) {
; CHECK-LABEL: gldff1sh_d_imm_offset_out_of_range:
; CHECK: mov	w8, #63
; CHECK-NEXT: ldff1sh { z0.d }, p0/z, [x8, z0.d]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv2i16.nxv2i64(<vscale x 2 x i1> %pg,
                                                                                               <vscale x 2 x i64> %base,
                                                                                               i64 63)
  %res = sext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

; LDFF1SW
define <vscale x 2 x i64> @gldff1sw_d_imm_offset_out_of_range(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %base) {
; CHECK-LABEL: gldff1sw_d_imm_offset_out_of_range:
; CHECK: mov	w8, #125
; CHECK-NEXT: ldff1sw { z0.d }, p0/z, [x8, z0.d]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv2i32.nxv2i64(<vscale x 2 x i1> %pg,
                                                                                               <vscale x 2 x i64> %base,
                                                                                               i64 125)
  %res = sext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

; LDFF1B/LDFF1SB
declare <vscale x 4 x i8> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv4i8.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, i64)
declare <vscale x 2 x i8> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv2i8.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, i64)

; LDFF1H/LDFF1SH
declare <vscale x 4 x i16> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv4i16.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, i64)
declare <vscale x 2 x i16> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv2i16.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, i64)

; LDFF1W/LDFF1SW
declare <vscale x 4 x i32> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv4i32.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, i64)
declare <vscale x 2 x i32> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv2i32.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, i64)

declare <vscale x 4 x float> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv4f32.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, i64)

; LDFF1D
declare <vscale x 2 x i64> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv2i64.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, i64)

declare <vscale x 2 x double> @llvm.aarch64.sve.ldff1.gather.scalar.offset.nxv2f64.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, i64)
