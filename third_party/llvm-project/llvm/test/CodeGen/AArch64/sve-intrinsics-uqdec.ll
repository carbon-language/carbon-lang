; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve -asm-verbose=0 < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sme -asm-verbose=0 < %s | FileCheck %s

; Since UQDEC{B|H|W|D|P} and UQINC{B|H|W|D|P} have identical semantics, the tests for
;   * @llvm.aarch64.sve.uqinc{b|h|w|d|p}, and
;   * @llvm.aarch64.sve.uqdec{b|h|w|d|p}
; should also be identical (with the instruction name being adjusted). When
; updating this file remember to make similar changes in the file testing the
; other intrinsic.

;
; UQDECH (vector)
;

define <vscale x 8 x i16> @uqdech(<vscale x 8 x i16> %a) {
; CHECK-LABEL: uqdech:
; CHECK: uqdech z0.h, pow2
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.uqdech.nxv8i16(<vscale x 8 x i16> %a,
                                                                  i32 0, i32 1)
  ret <vscale x 8 x i16> %out
}

;
; UQDECW (vector)
;

define <vscale x 4 x i32> @uqdecw(<vscale x 4 x i32> %a) {
; CHECK-LABEL: uqdecw:
; CHECK: uqdecw z0.s, vl1, mul #2
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.uqdecw.nxv4i32(<vscale x 4 x i32> %a,
                                                                  i32 1, i32 2)
  ret <vscale x 4 x i32> %out
}

;
; UQDECD (vector)
;

define <vscale x 2 x i64> @uqdecd(<vscale x 2 x i64> %a) {
; CHECK-LABEL: uqdecd:
; CHECK: uqdecd z0.d, vl2, mul #3
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.uqdecd.nxv2i64(<vscale x 2 x i64> %a,
                                                                  i32 2, i32 3)
  ret <vscale x 2 x i64> %out
}

;
; UQDECP (vector)
;

define <vscale x 8 x i16> @uqdecp_b16(<vscale x 8 x i16> %a, <vscale x 8 x i1> %b) {
; CHECK-LABEL: uqdecp_b16:
; CHECK: uqdecp z0.h, p0
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.uqdecp.nxv8i16(<vscale x 8 x i16> %a,
                                                                  <vscale x 8 x i1> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @uqdecp_b32(<vscale x 4 x i32> %a, <vscale x 4 x i1> %b) {
; CHECK-LABEL: uqdecp_b32:
; CHECK: uqdecp z0.s, p0
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.uqdecp.nxv4i32(<vscale x 4 x i32> %a,
                                                                  <vscale x 4 x i1> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @uqdecp_b64(<vscale x 2 x i64> %a, <vscale x 2 x i1> %b) {
; CHECK-LABEL: uqdecp_b64:
; CHECK: uqdecp z0.d, p0
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.uqdecp.nxv2i64(<vscale x 2 x i64> %a,
                                                                  <vscale x 2 x i1> %b)
  ret <vscale x 2 x i64> %out
}

;
; UQDECB (scalar)
;

define i32 @uqdecb_n32(i32 %a) {
; CHECK-LABEL: uqdecb_n32:
; CHECK: uqdecb w0, vl3, mul #4
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.uqdecb.n32(i32 %a, i32 3, i32 4)
  ret i32 %out
}

define i64 @uqdecb_n64(i64 %a) {
; CHECK-LABEL: uqdecb_n64:
; CHECK: uqdecb x0, vl4, mul #5
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.uqdecb.n64(i64 %a, i32 4, i32 5)
  ret i64 %out
}

;
; UQDECH (scalar)
;

define i32 @uqdech_n32(i32 %a) {
; CHECK-LABEL: uqdech_n32:
; CHECK: uqdech w0, vl5, mul #6
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.uqdech.n32(i32 %a, i32 5, i32 6)
  ret i32 %out
}

define i64 @uqdech_n64(i64 %a) {
; CHECK-LABEL: uqdech_n64:
; CHECK: uqdech x0, vl6, mul #7
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.uqdech.n64(i64 %a, i32 6, i32 7)
  ret i64 %out
}

;
; UQDECW (scalar)
;

define i32 @uqdecw_n32(i32 %a) {
; CHECK-LABEL: uqdecw_n32:
; CHECK: uqdecw w0, vl7, mul #8
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.uqdecw.n32(i32 %a, i32 7, i32 8)
  ret i32 %out
}

define i64 @uqdecw_n64(i64 %a) {
; CHECK-LABEL: uqdecw_n64:
; CHECK: uqdecw x0, vl8, mul #9
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.uqdecw.n64(i64 %a, i32 8, i32 9)
  ret i64 %out
}

;
; UQDECD (scalar)
;

define i32 @uqdecd_n32(i32 %a) {
; CHECK-LABEL: uqdecd_n32:
; CHECK: uqdecd w0, vl16, mul #10
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.uqdecd.n32(i32 %a, i32 9, i32 10)
  ret i32 %out
}

define i64 @uqdecd_n64(i64 %a) {
; CHECK-LABEL: uqdecd_n64:
; CHECK: uqdecd x0, vl32, mul #11
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.uqdecd.n64(i64 %a, i32 10, i32 11)
  ret i64 %out
}

;
; UQDECP (scalar)
;

define i32 @uqdecp_n32_b8(i32 %a, <vscale x 16 x i1> %b) {
; CHECK-LABEL: uqdecp_n32_b8:
; CHECK: uqdecp w0, p0.b
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.uqdecp.n32.nxv16i1(i32 %a, <vscale x 16 x i1> %b)
  ret i32 %out
}

define i32 @uqdecp_n32_b16(i32 %a, <vscale x 8 x i1> %b) {
; CHECK-LABEL: uqdecp_n32_b16:
; CHECK: uqdecp w0, p0.h
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.uqdecp.n32.nxv8i1(i32 %a, <vscale x 8 x i1> %b)
  ret i32 %out
}

define i32 @uqdecp_n32_b32(i32 %a, <vscale x 4 x i1> %b) {
; CHECK-LABEL: uqdecp_n32_b32:
; CHECK: uqdecp w0, p0.s
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.uqdecp.n32.nxv4i1(i32 %a, <vscale x 4 x i1> %b)
  ret i32 %out
}

define i32 @uqdecp_n32_b64(i32 %a, <vscale x 2 x i1> %b) {
; CHECK-LABEL: uqdecp_n32_b64:
; CHECK: uqdecp w0, p0.d
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.uqdecp.n32.nxv2i1(i32 %a, <vscale x 2 x i1> %b)
  ret i32 %out
}

define i64 @uqdecp_n64_b8(i64 %a, <vscale x 16 x i1> %b) {
; CHECK-LABEL: uqdecp_n64_b8:
; CHECK: uqdecp x0, p0.b
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.uqdecp.n64.nxv16i1(i64 %a, <vscale x 16 x i1> %b)
  ret i64 %out
}

define i64 @uqdecp_n64_b16(i64 %a, <vscale x 8 x i1> %b) {
; CHECK-LABEL: uqdecp_n64_b16:
; CHECK: uqdecp x0, p0.h
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.uqdecp.n64.nxv8i1(i64 %a, <vscale x 8 x i1> %b)
  ret i64 %out
}

define i64 @uqdecp_n64_b32(i64 %a, <vscale x 4 x i1> %b) {
; CHECK-LABEL: uqdecp_n64_b32:
; CHECK: uqdecp x0, p0.s
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.uqdecp.n64.nxv4i1(i64 %a, <vscale x 4 x i1> %b)
  ret i64 %out
}

define i64 @uqdecp_n64_b64(i64 %a, <vscale x 2 x i1> %b) {
; CHECK-LABEL: uqdecp_n64_b64:
; CHECK: uqdecp x0, p0.d
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.uqdecp.n64.nxv2i1(i64 %a, <vscale x 2 x i1> %b)
  ret i64 %out
}

; uqdec{h|w|d}(vector, pattern, multiplier)
declare <vscale x 8 x i16> @llvm.aarch64.sve.uqdech.nxv8i16(<vscale x 8 x i16>, i32, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.uqdecw.nxv4i32(<vscale x 4 x i32>, i32, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.uqdecd.nxv2i64(<vscale x 2 x i64>, i32, i32)

; uqdec{b|h|w|d}(scalar, pattern, multiplier)
declare i32 @llvm.aarch64.sve.uqdecb.n32(i32, i32, i32)
declare i64 @llvm.aarch64.sve.uqdecb.n64(i64, i32, i32)
declare i32 @llvm.aarch64.sve.uqdech.n32(i32, i32, i32)
declare i64 @llvm.aarch64.sve.uqdech.n64(i64, i32, i32)
declare i32 @llvm.aarch64.sve.uqdecw.n32(i32, i32, i32)
declare i64 @llvm.aarch64.sve.uqdecw.n64(i64, i32, i32)
declare i32 @llvm.aarch64.sve.uqdecd.n32(i32, i32, i32)
declare i64 @llvm.aarch64.sve.uqdecd.n64(i64, i32, i32)

; uqdecp(scalar, predicate)
declare i32 @llvm.aarch64.sve.uqdecp.n32.nxv16i1(i32, <vscale x 16 x i1>)
declare i32 @llvm.aarch64.sve.uqdecp.n32.nxv8i1(i32, <vscale x 8 x i1>)
declare i32 @llvm.aarch64.sve.uqdecp.n32.nxv4i1(i32, <vscale x 4 x i1>)
declare i32 @llvm.aarch64.sve.uqdecp.n32.nxv2i1(i32, <vscale x 2 x i1>)

declare i64 @llvm.aarch64.sve.uqdecp.n64.nxv16i1(i64, <vscale x 16 x i1>)
declare i64 @llvm.aarch64.sve.uqdecp.n64.nxv8i1(i64, <vscale x 8 x i1>)
declare i64 @llvm.aarch64.sve.uqdecp.n64.nxv4i1(i64, <vscale x 4 x i1>)
declare i64 @llvm.aarch64.sve.uqdecp.n64.nxv2i1(i64, <vscale x 2 x i1>)

; uqdecp(vector, predicate)
declare <vscale x 8 x i16> @llvm.aarch64.sve.uqdecp.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i1>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.uqdecp.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i1>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.uqdecp.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>)
