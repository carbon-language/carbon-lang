; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

;
; LD1H, LD1W, LD1D: base + 64-bit scaled offset
;   e.g. ld1h z0.d, p0/z, [x0, z0.d, lsl #1]
;

define <vscale x 2 x i64> @gld1h_index(<vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gld1h_index
; CHECK:	    ld1h	{ z0.d }, p0/z, [x0, z0.d, lsl #1]
; CHECK-NEXT:	ret
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.index.nxv2i16(<vscale x 2 x i1> %pg,
                                                                             i16* %base,
                                                                             <vscale x 2 x i64> %b)
  %res = zext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gld1w_index(<vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gld1w_index
; CHECK:	    ld1w	{ z0.d }, p0/z, [x0, z0.d, lsl #2]
; CHECK-NEXT:	ret
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ld1.gather.index.nxv2i32(<vscale x 2 x i1> %pg,
                                                                             i32* %base,
                                                                             <vscale x 2 x i64> %b)
  %res = zext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gld1d_index(<vscale x 2 x i1> %pg, i64* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gld1d_index
; CHECK:	    ld1d	{ z0.d }, p0/z, [x0, z0.d, lsl #3]
; CHECK-NEXT:	ret
  %load = call <vscale x 2 x i64> @llvm.aarch64.sve.ld1.gather.index.nxv2i64(<vscale x 2 x i1> %pg,
                                                                             i64* %base,
                                                                             <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %load
}

define <vscale x 2 x double> @gld1d_index_double(<vscale x 2 x i1> %pg, double* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gld1d_index_double
; CHECK:	    ld1d	{ z0.d }, p0/z, [x0, z0.d, lsl #3]
; CHECK-NEXT:	ret
  %load = call <vscale x 2 x double> @llvm.aarch64.sve.ld1.gather.index.nxv2f64(<vscale x 2 x i1> %pg,
                                                                                double* %base,
                                                                                <vscale x 2 x i64> %b)
  ret <vscale x 2 x double> %load
}

;
; LD1SH, LD1SW: base + 64-bit scaled offset
;   e.g. ld1sh z0.d, p0/z, [x0, z0.d, lsl #1]
;

define <vscale x 2 x i64> @gld1sh_index(<vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gld1sh_index
; CHECK:	    ld1sh	{ z0.d }, p0/z, [x0, z0.d, lsl #1]
; CHECK-NEXT:	ret
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.index.nxv2i16(<vscale x 2 x i1> %pg,
                                                                             i16* %base,
                                                                             <vscale x 2 x i64> %b)
  %res = sext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gld1sw_index(<vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gld1sw_index
; CHECK:	    ld1sw	{ z0.d }, p0/z, [x0, z0.d, lsl #2]
; CHECK-NEXT:	ret
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ld1.gather.index.nxv2i32(<vscale x 2 x i1> %pg,
                                                                             i32* %base,
                                                                             <vscale x 2 x i64> %b)
  %res = sext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

;
; LD1H, LD1W, LD1D: base + 64-bit sxtw'd scaled offset
;   e.g. ld1h z0.d, p0/z, [x0, z0.d, sxtw #1]
;

define <vscale x 2 x i64> @gld1h_index_sxtw(<vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gld1h_index_sxtw
; CHECK:	    ld1h	{ z0.d }, p0/z, [x0, z0.d, sxtw #1]
; CHECK-NEXT:	ret
  %sxtw = call <vscale x 2 x i64> @llvm.aarch64.sve.sxtw.nxv2i64(<vscale x 2 x i64> undef,
                                                                 <vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %b)
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.index.nxv2i16(<vscale x 2 x i1> %pg,
                                                                             i16* %base,
                                                                             <vscale x 2 x i64> %sxtw)
  %res = zext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gld1w_index_sxtw(<vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gld1w_index_sxtw
; CHECK:	    ld1w	{ z0.d }, p0/z, [x0, z0.d, sxtw #2]
; CHECK-NEXT:	ret
  %sxtw = call <vscale x 2 x i64> @llvm.aarch64.sve.sxtw.nxv2i64(<vscale x 2 x i64> undef,
                                                                 <vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %b)
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ld1.gather.index.nxv2i32(<vscale x 2 x i1> %pg,
                                                                             i32* %base,
                                                                             <vscale x 2 x i64> %sxtw)
  %res = zext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gld1d_index_sxtw(<vscale x 2 x i1> %pg, i64* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gld1d_index_sxtw
; CHECK:	    ld1d	{ z0.d }, p0/z, [x0, z0.d, sxtw #3]
; CHECK-NEXT:	ret
  %sxtw = call <vscale x 2 x i64> @llvm.aarch64.sve.sxtw.nxv2i64(<vscale x 2 x i64> undef,
                                                                 <vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %b)
  %load = call <vscale x 2 x i64> @llvm.aarch64.sve.ld1.gather.index.nxv2i64(<vscale x 2 x i1> %pg,
                                                                             i64* %base,
                                                                             <vscale x 2 x i64> %sxtw)
  ret <vscale x 2 x i64> %load
}

define <vscale x 2 x double> @gld1d_index_double_sxtw(<vscale x 2 x i1> %pg, double* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gld1d_index_double_sxtw
; CHECK:	    ld1d	{ z0.d }, p0/z, [x0, z0.d, sxtw #3]
; CHECK-NEXT:	ret
  %sxtw = call <vscale x 2 x i64> @llvm.aarch64.sve.sxtw.nxv2i64(<vscale x 2 x i64> undef,
                                                                 <vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %b)
  %load = call <vscale x 2 x double> @llvm.aarch64.sve.ld1.gather.index.nxv2f64(<vscale x 2 x i1> %pg,
                                                                                double* %base,
                                                                                <vscale x 2 x i64> %sxtw)
  ret <vscale x 2 x double> %load
}

;
; LD1SH, LD1SW: base + 64-bit sxtw'd scaled offset
;   e.g. ld1sh z0.d, p0/z, [x0, z0.d, sxtw #1]
;

define <vscale x 2 x i64> @gld1sh_index_sxtw(<vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gld1sh_index_sxtw
; CHECK:	    ld1sh	{ z0.d }, p0/z, [x0, z0.d, sxtw #1]
; CHECK-NEXT:	ret
  %sxtw = call <vscale x 2 x i64> @llvm.aarch64.sve.sxtw.nxv2i64(<vscale x 2 x i64> undef,
                                                                 <vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %b)
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.index.nxv2i16(<vscale x 2 x i1> %pg,
                                                                             i16* %base,
                                                                             <vscale x 2 x i64> %sxtw)
  %res = sext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gld1sw_index_sxtw(<vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gld1sw_index_sxtw
; CHECK:	    ld1sw	{ z0.d }, p0/z, [x0, z0.d, sxtw #2]
; CHECK-NEXT:	ret
  %sxtw = call <vscale x 2 x i64> @llvm.aarch64.sve.sxtw.nxv2i64(<vscale x 2 x i64> undef,
                                                                 <vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %b)
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ld1.gather.index.nxv2i32(<vscale x 2 x i1> %pg,
                                                                             i32* %base,
                                                                             <vscale x 2 x i64> %sxtw)
  %res = sext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

;
; LD1H, LD1W, LD1D: base + 64-bit sxtw'd scaled offset
;   e.g. ld1h z0.d, p0/z, [x0, z0.d, uxtw #1]
;

define <vscale x 2 x i64> @gld1h_index_uxtw(<vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gld1h_index_uxtw
; CHECK:	    ld1h	{ z0.d }, p0/z, [x0, z0.d, uxtw #1]
; CHECK-NEXT:	ret
  %uxtw = call <vscale x 2 x i64> @llvm.aarch64.sve.uxtw.nxv2i64(<vscale x 2 x i64> undef,
                                                                 <vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %b)
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.index.nxv2i16(<vscale x 2 x i1> %pg,
                                                                             i16* %base,
                                                                             <vscale x 2 x i64> %uxtw)
  %res = zext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gld1w_index_uxtw(<vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gld1w_index_uxtw
; CHECK:	    ld1w	{ z0.d }, p0/z, [x0, z0.d, uxtw #2]
; CHECK-NEXT:	ret
  %uxtw = call <vscale x 2 x i64> @llvm.aarch64.sve.uxtw.nxv2i64(<vscale x 2 x i64> undef,
                                                                 <vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %b)
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ld1.gather.index.nxv2i32(<vscale x 2 x i1> %pg,
                                                                             i32* %base,
                                                                             <vscale x 2 x i64> %uxtw)
  %res = zext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gld1d_index_uxtw(<vscale x 2 x i1> %pg, i64* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gld1d_index_uxtw
; CHECK:	    ld1d	{ z0.d }, p0/z, [x0, z0.d, uxtw #3]
; CHECK-NEXT:	ret
  %uxtw = call <vscale x 2 x i64> @llvm.aarch64.sve.uxtw.nxv2i64(<vscale x 2 x i64> undef,
                                                                 <vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %b)
  %load = call <vscale x 2 x i64> @llvm.aarch64.sve.ld1.gather.index.nxv2i64(<vscale x 2 x i1> %pg,
                                                                             i64* %base,
                                                                             <vscale x 2 x i64> %uxtw)
  ret <vscale x 2 x i64> %load
}

define <vscale x 2 x double> @gld1d_index_double_uxtw(<vscale x 2 x i1> %pg, double* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gld1d_index_double_uxtw
; CHECK:	    ld1d	{ z0.d }, p0/z, [x0, z0.d, uxtw #3]
; CHECK-NEXT:	ret
  %uxtw = call <vscale x 2 x i64> @llvm.aarch64.sve.uxtw.nxv2i64(<vscale x 2 x i64> undef,
                                                                 <vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %b)
  %load = call <vscale x 2 x double> @llvm.aarch64.sve.ld1.gather.index.nxv2f64(<vscale x 2 x i1> %pg,
                                                                                double* %base,
                                                                                <vscale x 2 x i64> %uxtw)
  ret <vscale x 2 x double> %load
}

;
; LD1SH, LD1SW: base + 64-bit uxtw'd scaled offset
;   e.g. ld1sh z0.d, p0/z, [x0, z0.d, uxtw #1]
;

define <vscale x 2 x i64> @gld1sh_index_uxtw(<vscale x 2 x i1> %pg, i16* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gld1sh_index_uxtw
; CHECK:	    ld1sh	{ z0.d }, p0/z, [x0, z0.d, uxtw #1]
; CHECK-NEXT:	ret
  %uxtw = call <vscale x 2 x i64> @llvm.aarch64.sve.uxtw.nxv2i64(<vscale x 2 x i64> undef,
                                                                 <vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %b)
  %load = call <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.index.nxv2i16(<vscale x 2 x i1> %pg,
                                                                             i16* %base,
                                                                             <vscale x 2 x i64> %uxtw)
  %res = sext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x i64> @gld1sw_index_uxtw(<vscale x 2 x i1> %pg, i32* %base, <vscale x 2 x i64> %b) {
; CHECK-LABEL: gld1sw_index_uxtw
; CHECK:	    ld1sw	{ z0.d }, p0/z, [x0, z0.d, uxtw #2]
; CHECK-NEXT:	ret
  %uxtw = call <vscale x 2 x i64> @llvm.aarch64.sve.uxtw.nxv2i64(<vscale x 2 x i64> undef,
                                                                 <vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %b)
  %load = call <vscale x 2 x i32> @llvm.aarch64.sve.ld1.gather.index.nxv2i32(<vscale x 2 x i1> %pg,
                                                                             i32* %base,
                                                                             <vscale x 2 x i64> %uxtw)
  %res = sext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %res
}

declare <vscale x 2 x i16> @llvm.aarch64.sve.ld1.gather.index.nxv2i16(<vscale x 2 x i1>, i16*, <vscale x 2 x i64>)
declare <vscale x 2 x i32> @llvm.aarch64.sve.ld1.gather.index.nxv2i32(<vscale x 2 x i1>, i32*, <vscale x 2 x i64>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.ld1.gather.index.nxv2i64(<vscale x 2 x i1>, i64*, <vscale x 2 x i64>)
declare <vscale x 2 x double> @llvm.aarch64.sve.ld1.gather.index.nxv2f64(<vscale x 2 x i1>, double*, <vscale x 2 x i64>)

declare <vscale x 2 x i64> @llvm.aarch64.sve.sxtw.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, <vscale x 2 x i64>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.uxtw.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, <vscale x 2 x i64>)
