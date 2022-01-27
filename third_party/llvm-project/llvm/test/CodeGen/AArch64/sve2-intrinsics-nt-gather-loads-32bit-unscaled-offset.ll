; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2 < %s | FileCheck %s

;
; LDNT1B, LDNT1W, LDNT1H, LDNT1D: base + 32-bit unscaled offsets, zero (uxtw)
; extended to 64 bits.
;   e.g. ldnt1h { z0.s }, p0/z, [z0.s, x0]
;

; LDNT1B
define <vscale x 4 x i32> @gldnt1b_s_uxtw(<vscale x 4 x i1> %pg, i8* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gldnt1b_s_uxtw:
; CHECK: ldnt1b { z0.s }, p0/z, [z0.s, x0]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i8> @llvm.aarch64.sve.ldnt1.gather.uxtw.nxv4i8(<vscale x 4 x i1> %pg,
                                                                            i8* %base,
                                                                            <vscale x 4 x i32> %b)
  %res = zext <vscale x 4 x i8> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

; LDNT1H
define <vscale x 4 x i32> @gldnt1h_s_uxtw(<vscale x 4 x i1> %pg, i16* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gldnt1h_s_uxtw:
; CHECK: ldnt1h { z0.s }, p0/z, [z0.s, x0]
; CHECK-NEXT:	ret
  %load = call <vscale x 4 x i16> @llvm.aarch64.sve.ldnt1.gather.uxtw.nxv4i16(<vscale x 4 x i1> %pg,
                                                                              i16* %base,
                                                                              <vscale x 4 x i32> %b)
  %res = zext <vscale x 4 x i16> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

; LDNT1W
define <vscale x 4 x i32> @gldnt1w_s_uxtw(<vscale x 4 x i1> %pg, i32* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gldnt1w_s_uxtw:
; CHECK: ldnt1w { z0.s }, p0/z, [z0.s, x0]
; CHECK-NEXT:	ret
  %load = call <vscale x 4 x i32> @llvm.aarch64.sve.ldnt1.gather.uxtw.nxv4i32(<vscale x 4 x i1> %pg,
                                                                              i32* %base,
                                                                              <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %load
}

define <vscale x 4 x float> @gldnt1w_s_uxtw_float(<vscale x 4 x i1> %pg, float* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gldnt1w_s_uxtw_float:
; CHECK: ldnt1w { z0.s }, p0/z, [z0.s, x0]
; CHECK-NEXT:	ret
  %load = call <vscale x 4 x float> @llvm.aarch64.sve.ldnt1.gather.uxtw.nxv4f32(<vscale x 4 x i1> %pg,
                                                                                float* %base,
                                                                                <vscale x 4 x i32> %b)
  ret <vscale x 4 x float> %load
}

; LDNT1SB, LDNT1SW, LDNT1SH: base + 32-bit unscaled offsets, zero (uxtw)
; extended to 64 bits.
;   e.g. ldnt1sh { z0.s }, p0/z, [z0.s, x0]
;

; LDNT1SB
define <vscale x 4 x i32> @gldnt1sb_s_uxtw(<vscale x 4 x i1> %pg, i8* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gldnt1sb_s_uxtw:
; CHECK: ldnt1sb { z0.s }, p0/z, [z0.s, x0]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i8> @llvm.aarch64.sve.ldnt1.gather.uxtw.nxv4i8(<vscale x 4 x i1> %pg,
                                                                            i8* %base,
                                                                            <vscale x 4 x i32> %b)
  %res = sext <vscale x 4 x i8> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

; LDNT1SH
define <vscale x 4 x i32> @gldnt1sh_s_uxtw(<vscale x 4 x i1> %pg, i16* %base, <vscale x 4 x i32> %b) {
; CHECK-LABEL: gldnt1sh_s_uxtw:
; CHECK: ldnt1sh { z0.s }, p0/z, [z0.s, x0]
; CHECK-NEXT:	ret
  %load = call <vscale x 4 x i16> @llvm.aarch64.sve.ldnt1.gather.uxtw.nxv4i16(<vscale x 4 x i1> %pg,
                                                                              i16* %base,
                                                                              <vscale x 4 x i32> %b)
  %res = sext <vscale x 4 x i16> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %res
}

; LDNT1B/LDNT1SB
declare <vscale x 4 x i8> @llvm.aarch64.sve.ldnt1.gather.uxtw.nxv4i8(<vscale x 4 x i1>, i8*, <vscale x 4 x i32>)
declare <vscale x 4 x i8> @llvm.aarch64.sve.ldnt1.gather.sxtw.nxv4i8(<vscale x 4 x i1>, i8*, <vscale x 4 x i32>)

; LDNT1H/LDNT1SH
declare <vscale x 4 x i16> @llvm.aarch64.sve.ldnt1.gather.sxtw.nxv4i16(<vscale x 4 x i1>, i16*, <vscale x 4 x i32>)
declare <vscale x 4 x i16> @llvm.aarch64.sve.ldnt1.gather.uxtw.nxv4i16(<vscale x 4 x i1>, i16*, <vscale x 4 x i32>)

; LDNT1W/LDNT1SW
declare <vscale x 4 x i32> @llvm.aarch64.sve.ldnt1.gather.sxtw.nxv4i32(<vscale x 4 x i1>, i32*, <vscale x 4 x i32>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.ldnt1.gather.uxtw.nxv4i32(<vscale x 4 x i1>, i32*, <vscale x 4 x i32>)

declare <vscale x 4 x float> @llvm.aarch64.sve.ldnt1.gather.sxtw.nxv4f32(<vscale x 4 x i1>, float*, <vscale x 4 x i32>)
declare <vscale x 4 x float> @llvm.aarch64.sve.ldnt1.gather.uxtw.nxv4f32(<vscale x 4 x i1>, float*, <vscale x 4 x i32>)
