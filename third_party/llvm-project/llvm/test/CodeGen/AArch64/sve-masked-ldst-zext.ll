; RUN: llc -mtriple=aarch64--linux-gnu -mattr=+sve < %s | FileCheck %s

;
; Masked Loads
;

define <vscale x 2 x i64> @masked_zload_nxv2i8(<vscale x 2 x i8>* %src, <vscale x 2 x i1> %mask) {
; CHECK-LABEL: masked_zload_nxv2i8:
; CHECK-NOT: ld1sb
; CHECK: ld1b { [[IN:z[0-9]+]].d }, [[PG:p[0-9]+]]/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i8> @llvm.masked.load.nxv2i8(<vscale x 2 x i8>* %src, i32 1, <vscale x 2 x i1> %mask, <vscale x 2 x i8> undef)
  %ext = zext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %ext
}

define <vscale x 2 x i64> @masked_zload_nxv2i16(<vscale x 2 x i16>* %src, <vscale x 2 x i1> %mask) {
; CHECK-LABEL: masked_zload_nxv2i16:
; CHECK-NOT: ld1sh
; CHECK: ld1h { [[IN:z[0-9]+]].d }, [[PG:p[0-9]+]]/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i16> @llvm.masked.load.nxv2i16(<vscale x 2 x i16>* %src, i32 1, <vscale x 2 x i1> %mask, <vscale x 2 x i16> undef)
  %ext = zext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %ext
}

define <vscale x 2 x i64> @masked_zload_nxv2i32(<vscale x 2 x i32>* %src, <vscale x 2 x i1> %mask) {
; CHECK-LABEL: masked_zload_nxv2i32:
; CHECK-NOT: ld1sw
; CHECK: ld1w { [[IN:z[0-9]+]].d }, [[PG:p[0-9]+]]/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i32> @llvm.masked.load.nxv2i32(<vscale x 2 x i32>* %src, i32 1, <vscale x 2 x i1> %mask, <vscale x 2 x i32> undef)
  %ext = zext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %ext
}

define <vscale x 4 x i32> @masked_zload_nxv4i8(<vscale x 4 x i8>* %src, <vscale x 4 x i1> %mask) {
; CHECK-LABEL: masked_zload_nxv4i8:
; CHECK-NOT: ld1sb
; CHECK: ld1b { [[IN:z[0-9]+]].s }, [[PG:p[0-9]+]]/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i8> @llvm.masked.load.nxv4i8(<vscale x 4 x i8>* %src, i32 1, <vscale x 4 x i1> %mask, <vscale x 4 x i8> undef)
  %ext = zext <vscale x 4 x i8> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %ext
}

define <vscale x 4 x i32> @masked_zload_nxv4i16(<vscale x 4 x i16>* %src, <vscale x 4 x i1> %mask) {
; CHECK-LABEL: masked_zload_nxv4i16:
; CHECK-NOT: ld1sh
; CHECK: ld1h { [[IN:z[0-9]+]].s }, [[PG:p[0-9]+]]/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i16> @llvm.masked.load.nxv4i16(<vscale x 4 x i16>* %src, i32 1, <vscale x 4 x i1> %mask, <vscale x 4 x i16> undef)
  %ext = zext <vscale x 4 x i16> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %ext
}

define <vscale x 8 x i16> @masked_zload_nxv8i8(<vscale x 8 x i8>* %src, <vscale x 8 x i1> %mask) {
; CHECK-LABEL: masked_zload_nxv8i8:
; CHECK-NOT: ld1sb
; CHECK: ld1b { [[IN:z[0-9]+]].h }, [[PG:p[0-9]+]]/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 8 x i8> @llvm.masked.load.nxv8i8(<vscale x 8 x i8>* %src, i32 1, <vscale x 8 x i1> %mask, <vscale x 8 x i8> undef)
  %ext = zext <vscale x 8 x i8> %load to <vscale x 8 x i16>
  ret <vscale x 8 x i16> %ext
}

define <vscale x 2 x i64> @masked_zload_passthru(<vscale x 2 x i32>* %src, <vscale x 2 x i1> %mask, <vscale x 2 x i32> %passthru) {
; CHECK-LABEL: masked_zload_passthru:
; CHECK-NOT: ld1sw
; CHECK: ld1w { [[IN:z[0-9]+]].d }, [[PG:p[0-9]+]]/z, [x0]
; CHECK-NEXT: and z0.d, z0.d, #0xffffffff
; CHECK-NEXT: mov z0.d, [[PG]]/m, [[IN]].d
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i32> @llvm.masked.load.nxv2i32(<vscale x 2 x i32>* %src, i32 1, <vscale x 2 x i1> %mask, <vscale x 2 x i32> %passthru)
  %ext = zext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %ext
}

declare <vscale x 2 x i8> @llvm.masked.load.nxv2i8(<vscale x 2 x i8>*, i32, <vscale x 2 x i1>, <vscale x 2 x i8>)
declare <vscale x 2 x i16> @llvm.masked.load.nxv2i16(<vscale x 2 x i16>*, i32, <vscale x 2 x i1>, <vscale x 2 x i16>)
declare <vscale x 2 x i32> @llvm.masked.load.nxv2i32(<vscale x 2 x i32>*, i32, <vscale x 2 x i1>, <vscale x 2 x i32>)
declare <vscale x 4 x i8> @llvm.masked.load.nxv4i8(<vscale x 4 x i8>*, i32, <vscale x 4 x i1>, <vscale x 4 x i8>)
declare <vscale x 4 x i16> @llvm.masked.load.nxv4i16(<vscale x 4 x i16>*, i32, <vscale x 4 x i1>, <vscale x 4 x i16>)
declare <vscale x 8 x i8> @llvm.masked.load.nxv8i8(<vscale x 8 x i8>*, i32, <vscale x 8 x i1>, <vscale x 8 x i8>)
