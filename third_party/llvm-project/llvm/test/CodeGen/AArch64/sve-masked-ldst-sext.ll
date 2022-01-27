; RUN: llc -mtriple=aarch64--linux-gnu -mattr=+sve < %s | FileCheck %s

;
; Masked Loads
;

define <vscale x 2 x i64> @masked_sload_nxv2i8(<vscale x 2 x i8> *%a, <vscale x 2 x i1> %mask) {
; CHECK-LABEL: masked_sload_nxv2i8:
; CHECK: ld1sb { [[IN:z[0-9]+]].d }, [[PG:p[0-9]+]]/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i8> @llvm.masked.load.nxv2i8(<vscale x 2 x i8> *%a, i32 1, <vscale x 2 x i1> %mask, <vscale x 2 x i8> undef)
  %ext = sext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %ext
}

define <vscale x 2 x i64> @masked_sload_nxv2i16(<vscale x 2 x i16> *%a, <vscale x 2 x i1> %mask) {
; CHECK-LABEL: masked_sload_nxv2i16:
; CHECK: ld1sh { [[IN:z[0-9]+]].d }, [[PG:p[0-9]+]]/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i16> @llvm.masked.load.nxv2i16(<vscale x 2 x i16> *%a, i32 1, <vscale x 2 x i1> %mask, <vscale x 2 x i16> undef)
  %ext = sext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %ext
}

define <vscale x 2 x i64> @masked_sload_nxv2i32(<vscale x 2 x i32> *%a, <vscale x 2 x i1> %mask) {
; CHECK-LABEL: masked_sload_nxv2i32:
; CHECK: ld1sw { [[IN:z[0-9]+]].d }, [[PG:p[0-9]+]]/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i32> @llvm.masked.load.nxv2i32(<vscale x 2 x i32> *%a, i32 1, <vscale x 2 x i1> %mask, <vscale x 2 x i32> undef)
  %ext = sext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %ext
}

define <vscale x 4 x i32> @masked_sload_nxv4i8(<vscale x 4 x i8> *%a, <vscale x 4 x i1> %mask) {
; CHECK-LABEL: masked_sload_nxv4i8:
; CHECK: ld1sb { [[IN:z[0-9]+]].s }, [[PG:p[0-9]+]]/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i8> @llvm.masked.load.nxv4i8(<vscale x 4 x i8> *%a, i32 1, <vscale x 4 x i1> %mask, <vscale x 4 x i8> undef)
  %ext = sext <vscale x 4 x i8> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %ext
}

define <vscale x 4 x i32> @masked_sload_nxv4i16(<vscale x 4 x i16> *%a, <vscale x 4 x i1> %mask) {
; CHECK-LABEL: masked_sload_nxv4i16:
; CHECK: ld1sh { [[IN:z[0-9]+]].s }, [[PG:p[0-9]+]]/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i16> @llvm.masked.load.nxv4i16(<vscale x 4 x i16> *%a, i32 1, <vscale x 4 x i1> %mask, <vscale x 4 x i16> undef)
  %ext = sext <vscale x 4 x i16> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %ext
}

define <vscale x 8 x i16> @masked_sload_nxv8i8(<vscale x 8 x i8> *%a, <vscale x 8 x i1> %mask) {
; CHECK-LABEL: masked_sload_nxv8i8:
; CHECK: ld1sb { [[IN:z[0-9]+]].h }, [[PG:p[0-9]+]]/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 8 x i8> @llvm.masked.load.nxv8i8(<vscale x 8 x i8> *%a, i32 1, <vscale x 8 x i1> %mask, <vscale x 8 x i8> undef)
  %ext = sext <vscale x 8 x i8> %load to <vscale x 8 x i16>
  ret <vscale x 8 x i16> %ext
}

define <vscale x 2 x i64> @masked_sload_passthru(<vscale x 2 x i32> *%a, <vscale x 2 x i1> %mask, <vscale x 2 x i32> %passthru) {
; CHECK-LABEL: masked_sload_passthru:
; CHECK: ld1sw { [[IN:z[0-9]+]].d }, [[PG1:p[0-9]+]]/z, [x0]
; CHECK-NEXT: ptrue [[PG2:p[0-9]+]].d
; CHECK-NEXT: sxtw z0.d, [[PG2]]/m, z0.d
; CHECK-NEXT: mov z0.d, [[PG1]]/m, [[IN]].d
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i32> @llvm.masked.load.nxv2i32(<vscale x 2 x i32> *%a, i32 1, <vscale x 2 x i1> %mask, <vscale x 2 x i32> %passthru)
  %ext = sext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %ext
}

; Return type requires splitting
define <vscale x 16 x i32> @masked_sload_nxv16i8(<vscale x 16 x i8>* %a, <vscale x 16 x i1> %mask) {
; CHECK-LABEL: masked_sload_nxv16i8:
; CHECK:         punpklo p1.h, p0.b
; CHECK-NEXT:    punpkhi p0.h, p0.b
; CHECK-NEXT:    punpklo p2.h, p1.b
; CHECK-NEXT:    punpkhi p1.h, p1.b
; CHECK-NEXT:    ld1sb { z0.s }, p2/z, [x0]
; CHECK-NEXT:    punpklo p2.h, p0.b
; CHECK-NEXT:    punpkhi p0.h, p0.b
; CHECK-NEXT:    ld1sb { z1.s }, p1/z, [x0, #1, mul vl]
; CHECK-NEXT:    ld1sb { z2.s }, p2/z, [x0, #2, mul vl]
; CHECK-NEXT:    ld1sb { z3.s }, p0/z, [x0, #3, mul vl]
; CHECK-NEXT:    ret
  %load = call <vscale x 16 x i8> @llvm.masked.load.nxv16i8(<vscale x 16 x i8>* %a, i32 2, <vscale x 16 x i1> %mask, <vscale x 16 x i8> undef)
  %ext = sext <vscale x 16 x i8> %load to <vscale x 16 x i32>
  ret <vscale x 16 x i32> %ext
}

; Masked load requires promotion
define <vscale x 4 x double> @masked_sload_4i8_4f32(<vscale x 4 x i8>* noalias %in, <vscale x 4 x i1> %mask) {
; CHECK-LABEL: masked_sload_4i8_4f32:
; CHECK:       punpkhi p2.h, p0.b
; CHECK-NEXT:  punpklo p0.h, p0.b
; CHECK-NEXT:  ld1sb { z1.d }, p2/z, [x0, #1, mul vl]
; CHECK-NEXT:  ld1sb { z0.d }, p0/z, [x0]
; CHECK-NEXT:  ptrue p1.d
; CHECK-NEXT:  scvtf z0.d, p1/m, z0.d
; CHECK-NEXT:  scvtf z1.d, p1/m, z1.d
; CHECK-NEXT:  ret
  %wide.load = call <vscale x 4 x i8> @llvm.masked.load.nxv4i8(<vscale x 4 x i8>* %in, i32 2, <vscale x 4 x i1> %mask, <vscale x 4 x i8> undef)
  %sext = sext <vscale x 4 x i8> %wide.load to <vscale x 4 x i64>
  %res = sitofp <vscale x 4 x i64> %sext to <vscale x 4 x double>
  ret <vscale x 4 x double> %res
}

declare <vscale x 2 x i8> @llvm.masked.load.nxv2i8(<vscale x 2 x i8>*, i32, <vscale x 2 x i1>, <vscale x 2 x i8>)
declare <vscale x 2 x i16> @llvm.masked.load.nxv2i16(<vscale x 2 x i16>*, i32, <vscale x 2 x i1>, <vscale x 2 x i16>)
declare <vscale x 2 x i32> @llvm.masked.load.nxv2i32(<vscale x 2 x i32>*, i32, <vscale x 2 x i1>, <vscale x 2 x i32>)
declare <vscale x 4 x i8> @llvm.masked.load.nxv4i8(<vscale x 4 x i8>*, i32, <vscale x 4 x i1>, <vscale x 4 x i8>)
declare <vscale x 4 x i16> @llvm.masked.load.nxv4i16(<vscale x 4 x i16>*, i32, <vscale x 4 x i1>, <vscale x 4 x i16>)
declare <vscale x 8 x i8> @llvm.masked.load.nxv8i8(<vscale x 8 x i8>*, i32, <vscale x 8 x i1>, <vscale x 8 x i8>)
declare <vscale x 16 x i8> @llvm.masked.load.nxv16i8(<vscale x 16 x i8>*, i32, <vscale x 16 x i1>, <vscale x 16 x i8>)
