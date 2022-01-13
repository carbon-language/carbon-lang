; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

;
; CMPEQ
;

define <vscale x 16 x i1> @cmpeq_b(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: cmpeq_b:
; CHECK: cmpeq p0.b, p0/z, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.cmpeq.nxv16i8(<vscale x 16 x i1> %pg,
                                                                 <vscale x 16 x i8> %a,
                                                                 <vscale x 16 x i8> %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @cmpeq_h(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: cmpeq_h:
; CHECK: cmpeq p0.h, p0/z, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.cmpeq.nxv8i16(<vscale x 8 x i1> %pg,
                                                                <vscale x 8 x i16> %a,
                                                                <vscale x 8 x i16> %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @cmpeq_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: cmpeq_s:
; CHECK: cmpeq p0.s, p0/z, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.cmpeq.nxv4i32(<vscale x 4 x i1> %pg,
                                                                <vscale x 4 x i32> %a,
                                                                <vscale x 4 x i32> %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @cmpeq_d(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmpeq_d:
; CHECK: cmpeq p0.d, p0/z, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.cmpeq.nxv2i64(<vscale x 2 x i1> %pg,
                                                                <vscale x 2 x i64> %a,
                                                                <vscale x 2 x i64> %b)
  ret <vscale x 2 x i1> %out
}

define <vscale x 16 x i1> @cmpeq_wide_b(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmpeq_wide_b:
; CHECK: cmpeq p0.b, p0/z, z0.b, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.cmpeq.wide.nxv16i8(<vscale x 16 x i1> %pg,
                                                                      <vscale x 16 x i8> %a,
                                                                      <vscale x 2 x i64> %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @cmpeq_wide_h(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmpeq_wide_h:
; CHECK: cmpeq p0.h, p0/z, z0.h, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.cmpeq.wide.nxv8i16(<vscale x 8 x i1> %pg,
                                                                     <vscale x 8 x i16> %a,
                                                                     <vscale x 2 x i64> %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @cmpeq_wide_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmpeq_wide_s:
; CHECK: cmpeq p0.s, p0/z, z0.s, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.cmpeq.wide.nxv4i32(<vscale x 4 x i1> %pg,
                                                                     <vscale x 4 x i32> %a,
                                                                     <vscale x 2 x i64> %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 16 x i1> @cmpeq_ir_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: cmpeq_ir_b:
; CHECK: ptrue p0.b
; CHECK-NEXT: cmpeq p0.b, p0/z, z0.b, z1.b
; CHECK-NEXT: ret
  %out = icmp eq <vscale x 16 x i8> %a, %b
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @cmpeq_ir_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: cmpeq_ir_h:
; CHECK: ptrue p0.h
; CHECK-NEXT: cmpeq p0.h, p0/z, z0.h, z1.h
; CHECK-NEXT: ret
  %out = icmp eq <vscale x 8 x i16> %a, %b
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @cmpeq_ir_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: cmpeq_ir_s:
; CHECK: ptrue p0.s
; CHECK-NEXT: cmpeq p0.s, p0/z, z0.s, z1.s
; CHECK-NEXT: ret
  %out = icmp eq <vscale x 4 x i32> %a, %b
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @cmpeq_ir_d(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmpeq_ir_d:
; CHECK: ptrue p0.d
; CHECK-NEXT: cmpeq p0.d, p0/z, z0.d, z1.d
; CHECK-NEXT: ret
  %out = icmp eq <vscale x 2 x i64> %a, %b
  ret <vscale x 2 x i1> %out
}

;
; CMPGE
;

define <vscale x 16 x i1> @cmpge_b(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: cmpge_b:
; CHECK: cmpge p0.b, p0/z, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.cmpge.nxv16i8(<vscale x 16 x i1> %pg,
                                                                 <vscale x 16 x i8> %a,
                                                                 <vscale x 16 x i8> %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @cmpge_h(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: cmpge_h:
; CHECK: cmpge p0.h, p0/z, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.cmpge.nxv8i16(<vscale x 8 x i1> %pg,
                                                                <vscale x 8 x i16> %a,
                                                                <vscale x 8 x i16> %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @cmpge_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: cmpge_s:
; CHECK: cmpge p0.s, p0/z, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.cmpge.nxv4i32(<vscale x 4 x i1> %pg,
                                                                <vscale x 4 x i32> %a,
                                                                <vscale x 4 x i32> %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @cmpge_d(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmpge_d:
; CHECK: cmpge p0.d, p0/z, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.cmpge.nxv2i64(<vscale x 2 x i1> %pg,
                                                                <vscale x 2 x i64> %a,
                                                                <vscale x 2 x i64> %b)
  ret <vscale x 2 x i1> %out
}

define <vscale x 16 x i1> @cmpge_wide_b(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmpge_wide_b:
; CHECK: cmpge p0.b, p0/z, z0.b, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.cmpge.wide.nxv16i8(<vscale x 16 x i1> %pg,
                                                                      <vscale x 16 x i8> %a,
                                                                      <vscale x 2 x i64> %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @cmpge_wide_h(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmpge_wide_h:
; CHECK: cmpge p0.h, p0/z, z0.h, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.cmpge.wide.nxv8i16(<vscale x 8 x i1> %pg,
                                                                     <vscale x 8 x i16> %a,
                                                                     <vscale x 2 x i64> %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @cmpge_wide_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmpge_wide_s:
; CHECK: cmpge p0.s, p0/z, z0.s, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.cmpge.wide.nxv4i32(<vscale x 4 x i1> %pg,
                                                                     <vscale x 4 x i32> %a,
                                                                     <vscale x 2 x i64> %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 16 x i1> @cmpge_ir_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: cmpge_ir_b:
; CHECK: ptrue p0.b
; CHECK-NEXT: cmpge p0.b, p0/z, z0.b, z1.b
; CHECK-NEXT: ret
  %out = icmp sge <vscale x 16 x i8> %a, %b
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @cmpge_ir_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: cmpge_ir_h:
; CHECK: ptrue p0.h
; CHECK-NEXT: cmpge p0.h, p0/z, z0.h, z1.h
; CHECK-NEXT: ret
  %out = icmp sge <vscale x 8 x i16> %a, %b
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @cmpge_ir_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: cmpge_ir_s:
; CHECK: ptrue p0.s
; CHECK-NEXT: cmpge p0.s, p0/z, z0.s, z1.s
; CHECK-NEXT: ret
  %out = icmp sge <vscale x 4 x i32> %a, %b
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @cmpge_ir_d(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmpge_ir_d:
; CHECK: ptrue p0.d
; CHECK-NEXT: cmpge p0.d, p0/z, z0.d, z1.d
; CHECK-NEXT: ret
  %out = icmp sge <vscale x 2 x i64> %a, %b
  ret <vscale x 2 x i1> %out
}

define <vscale x 16 x i1> @cmpge_ir_comm_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: cmpge_ir_comm_b:
; CHECK: ptrue p0.b
; CHECK-NEXT: cmpge p0.b, p0/z, z1.b, z0.b
; CHECK-NEXT: ret
  %out = icmp sle <vscale x 16 x i8> %a, %b
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @cmpge_ir_comm_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: cmpge_ir_comm_h:
; CHECK: ptrue p0.h
; CHECK-NEXT: cmpge p0.h, p0/z, z1.h, z0.h
; CHECK-NEXT: ret
  %out = icmp sle <vscale x 8 x i16> %a, %b
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @cmpge_ir_comm_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: cmpge_ir_comm_s:
; CHECK: ptrue p0.s
; CHECK-NEXT: cmpge p0.s, p0/z, z1.s, z0.s
; CHECK-NEXT: ret
  %out = icmp sle <vscale x 4 x i32> %a, %b
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @cmpge_ir_comm_d(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmpge_ir_comm_d:
; CHECK: ptrue p0.d
; CHECK-NEXT: cmpge p0.d, p0/z, z1.d, z0.d
; CHECK-NEXT: ret
  %out = icmp sle <vscale x 2 x i64> %a, %b
  ret <vscale x 2 x i1> %out
}

;
; CMPGT
;

define <vscale x 16 x i1> @cmpgt_b(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: cmpgt_b:
; CHECK: cmpgt p0.b, p0/z, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.cmpgt.nxv16i8(<vscale x 16 x i1> %pg,
                                                                 <vscale x 16 x i8> %a,
                                                                 <vscale x 16 x i8> %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @cmpgt_h(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: cmpgt_h:
; CHECK: cmpgt p0.h, p0/z, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.cmpgt.nxv8i16(<vscale x 8 x i1> %pg,
                                                                <vscale x 8 x i16> %a,
                                                                <vscale x 8 x i16> %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @cmpgt_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: cmpgt_s:
; CHECK: cmpgt p0.s, p0/z, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.cmpgt.nxv4i32(<vscale x 4 x i1> %pg,
                                                                <vscale x 4 x i32> %a,
                                                                <vscale x 4 x i32> %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @cmpgt_d(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmpgt_d:
; CHECK: cmpgt p0.d, p0/z, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.cmpgt.nxv2i64(<vscale x 2 x i1> %pg,
                                                                <vscale x 2 x i64> %a,
                                                                <vscale x 2 x i64> %b)
  ret <vscale x 2 x i1> %out
}

define <vscale x 16 x i1> @cmpgt_wide_b(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmpgt_wide_b:
; CHECK: cmpgt p0.b, p0/z, z0.b, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.cmpgt.wide.nxv16i8(<vscale x 16 x i1> %pg,
                                                                      <vscale x 16 x i8> %a,
                                                                      <vscale x 2 x i64> %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @cmpgt_wide_h(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmpgt_wide_h:
; CHECK: cmpgt p0.h, p0/z, z0.h, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.cmpgt.wide.nxv8i16(<vscale x 8 x i1> %pg,
                                                                     <vscale x 8 x i16> %a,
                                                                     <vscale x 2 x i64> %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @cmpgt_wide_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmpgt_wide_s:
; CHECK: cmpgt p0.s, p0/z, z0.s, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.cmpgt.wide.nxv4i32(<vscale x 4 x i1> %pg,
                                                                     <vscale x 4 x i32> %a,
                                                                     <vscale x 2 x i64> %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 16 x i1> @cmpgt_ir_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: cmpgt_ir_b:
; CHECK: ptrue p0.b
; CHECK-NEXT: cmpgt p0.b, p0/z, z0.b, z1.b
; CHECK-NEXT: ret
  %out = icmp sgt <vscale x 16 x i8> %a, %b
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @cmpgt_ir_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: cmpgt_ir_h:
; CHECK: ptrue p0.h
; CHECK-NEXT: cmpgt p0.h, p0/z, z0.h, z1.h
; CHECK-NEXT: ret
  %out = icmp sgt <vscale x 8 x i16> %a, %b
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @cmpgt_ir_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: cmpgt_ir_s:
; CHECK: ptrue p0.s
; CHECK-NEXT: cmpgt p0.s, p0/z, z0.s, z1.s
; CHECK-NEXT: ret
  %out = icmp sgt <vscale x 4 x i32> %a, %b
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @cmpgt_ir_d(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmpgt_ir_d:
; CHECK: ptrue p0.d
; CHECK-NEXT: cmpgt p0.d, p0/z, z0.d, z1.d
; CHECK-NEXT: ret
  %out = icmp sgt <vscale x 2 x i64> %a, %b
  ret <vscale x 2 x i1> %out
}

define <vscale x 16 x i1> @cmpgt_ir_comm_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: cmpgt_ir_comm_b:
; CHECK: ptrue p0.b
; CHECK-NEXT: cmpgt p0.b, p0/z, z1.b, z0.b
; CHECK-NEXT: ret
  %out = icmp slt <vscale x 16 x i8> %a, %b
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @cmpgt_ir_comm_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: cmpgt_ir_comm_h:
; CHECK: ptrue p0.h
; CHECK-NEXT: cmpgt p0.h, p0/z, z1.h, z0.h
; CHECK-NEXT: ret
  %out = icmp slt <vscale x 8 x i16> %a, %b
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @cmpgt_ir_comm_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: cmpgt_ir_comm_s:
; CHECK: ptrue p0.s
; CHECK-NEXT: cmpgt p0.s, p0/z, z1.s, z0.s
; CHECK-NEXT: ret
  %out = icmp slt <vscale x 4 x i32> %a, %b
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @cmpgt_ir_comm_d(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmpgt_ir_comm_d:
; CHECK: ptrue p0.d
; CHECK-NEXT: cmpgt p0.d, p0/z, z1.d, z0.d
; CHECK-NEXT: ret
  %out = icmp slt <vscale x 2 x i64> %a, %b
  ret <vscale x 2 x i1> %out
}

;
; CMPHI
;

define <vscale x 16 x i1> @cmphi_b(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: cmphi_b:
; CHECK: cmphi p0.b, p0/z, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.cmphi.nxv16i8(<vscale x 16 x i1> %pg,
                                                                 <vscale x 16 x i8> %a,
                                                                 <vscale x 16 x i8> %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @cmphi_h(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: cmphi_h:
; CHECK: cmphi p0.h, p0/z, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.cmphi.nxv8i16(<vscale x 8 x i1> %pg,
                                                                <vscale x 8 x i16> %a,
                                                                <vscale x 8 x i16> %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @cmphi_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: cmphi_s:
; CHECK: cmphi p0.s, p0/z, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.cmphi.nxv4i32(<vscale x 4 x i1> %pg,
                                                                <vscale x 4 x i32> %a,
                                                                <vscale x 4 x i32> %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @cmphi_d(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmphi_d:
; CHECK: cmphi p0.d, p0/z, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.cmphi.nxv2i64(<vscale x 2 x i1> %pg,
                                                                <vscale x 2 x i64> %a,
                                                                <vscale x 2 x i64> %b)
  ret <vscale x 2 x i1> %out
}

define <vscale x 16 x i1> @cmphi_wide_b(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmphi_wide_b:
; CHECK: cmphi p0.b, p0/z, z0.b, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.cmphi.wide.nxv16i8(<vscale x 16 x i1> %pg,
                                                                      <vscale x 16 x i8> %a,
                                                                      <vscale x 2 x i64> %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @cmphi_wide_h(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmphi_wide_h:
; CHECK: cmphi p0.h, p0/z, z0.h, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.cmphi.wide.nxv8i16(<vscale x 8 x i1> %pg,
                                                                     <vscale x 8 x i16> %a,
                                                                     <vscale x 2 x i64> %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @cmphi_wide_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmphi_wide_s:
; CHECK: cmphi p0.s, p0/z, z0.s, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.cmphi.wide.nxv4i32(<vscale x 4 x i1> %pg,
                                                                     <vscale x 4 x i32> %a,
                                                                     <vscale x 2 x i64> %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 16 x i1> @cmphi_ir_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: cmphi_ir_b:
; CHECK: ptrue p0.b
; CHECK-NEXT: cmphi p0.b, p0/z, z0.b, z1.b
; CHECK-NEXT: ret
  %out = icmp ugt <vscale x 16 x i8> %a, %b
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @cmphi_ir_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: cmphi_ir_h:
; CHECK: ptrue p0.h
; CHECK-NEXT: cmphi p0.h, p0/z, z0.h, z1.h
; CHECK-NEXT: ret
  %out = icmp ugt <vscale x 8 x i16> %a, %b
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @cmphi_ir_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: cmphi_ir_s:
; CHECK: ptrue p0.s
; CHECK-NEXT: cmphi p0.s, p0/z, z0.s, z1.s
; CHECK-NEXT: ret
  %out = icmp ugt <vscale x 4 x i32> %a, %b
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @cmphi_ir_d(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmphi_ir_d:
; CHECK: ptrue p0.d
; CHECK-NEXT: cmphi p0.d, p0/z, z0.d, z1.d
; CHECK-NEXT: ret
  %out = icmp ugt <vscale x 2 x i64> %a, %b
  ret <vscale x 2 x i1> %out
}

define <vscale x 16 x i1> @cmphi_ir_comm_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: cmphi_ir_comm_b:
; CHECK: ptrue p0.b
; CHECK-NEXT: cmphi p0.b, p0/z, z1.b, z0.b
; CHECK-NEXT: ret
  %out = icmp ult <vscale x 16 x i8> %a, %b
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @cmphi_ir_comm_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: cmphi_ir_comm_h:
; CHECK: ptrue p0.h
; CHECK-NEXT: cmphi p0.h, p0/z, z1.h, z0.h
; CHECK-NEXT: ret
  %out = icmp ult <vscale x 8 x i16> %a, %b
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @cmphi_ir_comm_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: cmphi_ir_comm_s:
; CHECK: ptrue p0.s
; CHECK-NEXT: cmphi p0.s, p0/z, z1.s, z0.s
; CHECK-NEXT: ret
  %out = icmp ult <vscale x 4 x i32> %a, %b
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @cmphi_ir_comm_d(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmphi_ir_comm_d:
; CHECK: ptrue p0.d
; CHECK-NEXT: cmphi p0.d, p0/z, z1.d, z0.d
; CHECK-NEXT: ret
  %out = icmp ult <vscale x 2 x i64> %a, %b
  ret <vscale x 2 x i1> %out
}

;
; CMPHS
;

define <vscale x 16 x i1> @cmphs_b(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: cmphs_b:
; CHECK: cmphs p0.b, p0/z, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.cmphs.nxv16i8(<vscale x 16 x i1> %pg,
                                                                 <vscale x 16 x i8> %a,
                                                                 <vscale x 16 x i8> %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @cmphs_h(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: cmphs_h:
; CHECK: cmphs p0.h, p0/z, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.cmphs.nxv8i16(<vscale x 8 x i1> %pg,
                                                                <vscale x 8 x i16> %a,
                                                                <vscale x 8 x i16> %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @cmphs_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: cmphs_s:
; CHECK: cmphs p0.s, p0/z, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.cmphs.nxv4i32(<vscale x 4 x i1> %pg,
                                                                <vscale x 4 x i32> %a,
                                                                <vscale x 4 x i32> %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @cmphs_d(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmphs_d:
; CHECK: cmphs p0.d, p0/z, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.cmphs.nxv2i64(<vscale x 2 x i1> %pg,
                                                                <vscale x 2 x i64> %a,
                                                                <vscale x 2 x i64> %b)
  ret <vscale x 2 x i1> %out
}

define <vscale x 16 x i1> @cmphs_wide_b(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmphs_wide_b:
; CHECK: cmphs p0.b, p0/z, z0.b, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.cmphs.wide.nxv16i8(<vscale x 16 x i1> %pg,
                                                                      <vscale x 16 x i8> %a,
                                                                      <vscale x 2 x i64> %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @cmphs_wide_h(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmphs_wide_h:
; CHECK: cmphs p0.h, p0/z, z0.h, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.cmphs.wide.nxv8i16(<vscale x 8 x i1> %pg,
                                                                     <vscale x 8 x i16> %a,
                                                                     <vscale x 2 x i64> %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @cmphs_wide_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmphs_wide_s:
; CHECK: cmphs p0.s, p0/z, z0.s, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.cmphs.wide.nxv4i32(<vscale x 4 x i1> %pg,
                                                                     <vscale x 4 x i32> %a,
                                                                     <vscale x 2 x i64> %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 16 x i1> @cmphs_ir_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: cmphs_ir_b:
; CHECK: ptrue p0.b
; CHECK-NEXT: cmphs p0.b, p0/z, z0.b, z1.b
; CHECK-NEXT: ret
  %out = icmp uge <vscale x 16 x i8> %a, %b
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @cmphs_ir_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: cmphs_ir_h:
; CHECK: ptrue p0.h
; CHECK-NEXT: cmphs p0.h, p0/z, z0.h, z1.h
; CHECK-NEXT: ret
  %out = icmp uge <vscale x 8 x i16> %a, %b
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @cmphs_ir_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: cmphs_ir_s:
; CHECK: ptrue p0.s
; CHECK-NEXT: cmphs p0.s, p0/z, z0.s, z1.s
; CHECK-NEXT: ret
  %out = icmp uge <vscale x 4 x i32> %a, %b
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @cmphs_ir_d(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmphs_ir_d:
; CHECK: ptrue p0.d
; CHECK-NEXT: cmphs p0.d, p0/z, z0.d, z1.d
; CHECK-NEXT: ret
  %out = icmp uge <vscale x 2 x i64> %a, %b
  ret <vscale x 2 x i1> %out
}

define <vscale x 16 x i1> @cmphs_ir_comm_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: cmphs_ir_comm_b:
; CHECK: ptrue p0.b
; CHECK-NEXT: cmphs p0.b, p0/z, z1.b, z0.b
; CHECK-NEXT: ret
  %out = icmp ule <vscale x 16 x i8> %a, %b
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @cmphs_ir_comm_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: cmphs_ir_comm_h:
; CHECK: ptrue p0.h
; CHECK-NEXT: cmphs p0.h, p0/z, z1.h, z0.h
; CHECK-NEXT: ret
  %out = icmp ule <vscale x 8 x i16> %a, %b
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @cmphs_ir_comm_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: cmphs_ir_comm_s:
; CHECK: ptrue p0.s
; CHECK-NEXT: cmphs p0.s, p0/z, z1.s, z0.s
; CHECK-NEXT: ret
  %out = icmp ule <vscale x 4 x i32> %a, %b
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @cmphs_ir_comm_d(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmphs_ir_comm_d:
; CHECK: ptrue p0.d
; CHECK-NEXT: cmphs p0.d, p0/z, z1.d, z0.d
; CHECK-NEXT: ret
  %out = icmp ule <vscale x 2 x i64> %a, %b
  ret <vscale x 2 x i1> %out
}

;
; CMPLE
;

define <vscale x 16 x i1> @cmple_wide_b(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmple_wide_b:
; CHECK: cmple p0.b, p0/z, z0.b, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.cmple.wide.nxv16i8(<vscale x 16 x i1> %pg,
                                                                      <vscale x 16 x i8> %a,
                                                                      <vscale x 2 x i64> %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @cmple_wide_h(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmple_wide_h:
; CHECK: cmple p0.h, p0/z, z0.h, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.cmple.wide.nxv8i16(<vscale x 8 x i1> %pg,
                                                                     <vscale x 8 x i16> %a,
                                                                     <vscale x 2 x i64> %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @cmple_wide_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmple_wide_s:
; CHECK: cmple p0.s, p0/z, z0.s, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.cmple.wide.nxv4i32(<vscale x 4 x i1> %pg,
                                                                     <vscale x 4 x i32> %a,
                                                                     <vscale x 2 x i64> %b)
  ret <vscale x 4 x i1> %out
}

;
; CMPLO
;

define <vscale x 16 x i1> @cmplo_wide_b(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmplo_wide_b:
; CHECK: cmplo p0.b, p0/z, z0.b, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.cmplo.wide.nxv16i8(<vscale x 16 x i1> %pg,
                                                                      <vscale x 16 x i8> %a,
                                                                      <vscale x 2 x i64> %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @cmplo_wide_h(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmplo_wide_h:
; CHECK: cmplo p0.h, p0/z, z0.h, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.cmplo.wide.nxv8i16(<vscale x 8 x i1> %pg,
                                                                     <vscale x 8 x i16> %a,
                                                                     <vscale x 2 x i64> %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @cmplo_wide_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmplo_wide_s:
; CHECK: cmplo p0.s, p0/z, z0.s, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.cmplo.wide.nxv4i32(<vscale x 4 x i1> %pg,
                                                                     <vscale x 4 x i32> %a,
                                                                     <vscale x 2 x i64> %b)
  ret <vscale x 4 x i1> %out
}

;
; CMPLS
;

define <vscale x 16 x i1> @cmpls_wide_b(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmpls_wide_b:
; CHECK: cmpls p0.b, p0/z, z0.b, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.cmpls.wide.nxv16i8(<vscale x 16 x i1> %pg,
                                                                      <vscale x 16 x i8> %a,
                                                                      <vscale x 2 x i64> %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @cmpls_wide_h(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmpls_wide_h:
; CHECK: cmpls p0.h, p0/z, z0.h, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.cmpls.wide.nxv8i16(<vscale x 8 x i1> %pg,
                                                                     <vscale x 8 x i16> %a,
                                                                     <vscale x 2 x i64> %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @cmpls_wide_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmpls_wide_s:
; CHECK: cmpls p0.s, p0/z, z0.s, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.cmpls.wide.nxv4i32(<vscale x 4 x i1> %pg,
                                                                     <vscale x 4 x i32> %a,
                                                                     <vscale x 2 x i64> %b)
  ret <vscale x 4 x i1> %out
}

;
; CMPLT
;

define <vscale x 16 x i1> @cmplt_wide_b(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmplt_wide_b:
; CHECK: cmplt p0.b, p0/z, z0.b, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.cmplt.wide.nxv16i8(<vscale x 16 x i1> %pg,
                                                                      <vscale x 16 x i8> %a,
                                                                      <vscale x 2 x i64> %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @cmplt_wide_h(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmplt_wide_h:
; CHECK: cmplt p0.h, p0/z, z0.h, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.cmplt.wide.nxv8i16(<vscale x 8 x i1> %pg,
                                                                     <vscale x 8 x i16> %a,
                                                                     <vscale x 2 x i64> %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @cmplt_wide_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmplt_wide_s:
; CHECK: cmplt p0.s, p0/z, z0.s, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.cmplt.wide.nxv4i32(<vscale x 4 x i1> %pg,
                                                                     <vscale x 4 x i32> %a,
                                                                     <vscale x 2 x i64> %b)
  ret <vscale x 4 x i1> %out
}

;
; CMPNE
;

define <vscale x 16 x i1> @cmpne_b(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: cmpne_b:
; CHECK: cmpne p0.b, p0/z, z0.b, z1.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.cmpne.nxv16i8(<vscale x 16 x i1> %pg,
                                                                 <vscale x 16 x i8> %a,
                                                                 <vscale x 16 x i8> %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @cmpne_h(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: cmpne_h:
; CHECK: cmpne p0.h, p0/z, z0.h, z1.h
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.cmpne.nxv8i16(<vscale x 8 x i1> %pg,
                                                                <vscale x 8 x i16> %a,
                                                                <vscale x 8 x i16> %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @cmpne_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: cmpne_s:
; CHECK: cmpne p0.s, p0/z, z0.s, z1.s
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.cmpne.nxv4i32(<vscale x 4 x i1> %pg,
                                                                <vscale x 4 x i32> %a,
                                                                <vscale x 4 x i32> %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @cmpne_d(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmpne_d:
; CHECK: cmpne p0.d, p0/z, z0.d, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i1> @llvm.aarch64.sve.cmpne.nxv2i64(<vscale x 2 x i1> %pg,
                                                                <vscale x 2 x i64> %a,
                                                                <vscale x 2 x i64> %b)
  ret <vscale x 2 x i1> %out
}

define <vscale x 16 x i1> @cmpne_wide_b(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmpne_wide_b:
; CHECK: cmpne p0.b, p0/z, z0.b, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.cmpne.wide.nxv16i8(<vscale x 16 x i1> %pg,
                                                                      <vscale x 16 x i8> %a,
                                                                      <vscale x 2 x i64> %b)
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @cmpne_wide_h(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmpne_wide_h:
; CHECK: cmpne p0.h, p0/z, z0.h, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i1> @llvm.aarch64.sve.cmpne.wide.nxv8i16(<vscale x 8 x i1> %pg,
                                                                     <vscale x 8 x i16> %a,
                                                                     <vscale x 2 x i64> %b)
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @cmpne_wide_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmpne_wide_s:
; CHECK: cmpne p0.s, p0/z, z0.s, z1.d
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.cmpne.wide.nxv4i32(<vscale x 4 x i1> %pg,
                                                                     <vscale x 4 x i32> %a,
                                                                     <vscale x 2 x i64> %b)
  ret <vscale x 4 x i1> %out
}

define <vscale x 16 x i1> @cmpne_ir_b(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: cmpne_ir_b:
; CHECK: ptrue p0.b
; CHECK-NEXT: cmpne p0.b, p0/z, z0.b, z1.b
; CHECK-NEXT: ret
  %out = icmp ne <vscale x 16 x i8> %a, %b
  ret <vscale x 16 x i1> %out
}

define <vscale x 8 x i1> @cmpne_ir_h(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: cmpne_ir_h:
; CHECK: ptrue p0.h
; CHECK-NEXT: cmpne p0.h, p0/z, z0.h, z1.h
; CHECK-NEXT: ret
  %out = icmp ne <vscale x 8 x i16> %a, %b
  ret <vscale x 8 x i1> %out
}

define <vscale x 4 x i1> @cmpne_ir_s(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: cmpne_ir_s:
; CHECK: ptrue p0.s
; CHECK-NEXT: cmpne p0.s, p0/z, z0.s, z1.s
; CHECK-NEXT: ret
  %out = icmp ne <vscale x 4 x i32> %a, %b
  ret <vscale x 4 x i1> %out
}

define <vscale x 2 x i1> @cmpne_ir_d(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: cmpne_ir_d:
; CHECK: ptrue p0.d
; CHECK-NEXT: cmpne p0.d, p0/z, z0.d, z1.d
; CHECK-NEXT: ret
  %out = icmp ne <vscale x 2 x i64> %a, %b
  ret <vscale x 2 x i1> %out
}


define <vscale x 16 x i1> @cmpgt_wide_splat_b(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, i64 %b) {
; CHECK-LABEL: cmpgt_wide_splat_b:
; CHECK: cmpgt p0.b, p0/z, z0.b, z1.d
; CHECK-NEXT: ret
  %splat = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 %b)
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.cmpgt.wide.nxv16i8(<vscale x 16 x i1> %pg,
                                                                      <vscale x 16 x i8> %a,
                                                                      <vscale x 2 x i64> %splat)
  ret <vscale x 16 x i1> %out
}

define <vscale x 4 x i1> @cmpls_wide_splat_s(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, i64 %b) {
; CHECK-LABEL: cmpls_wide_splat_s:
; CHECK: cmpls p0.s, p0/z, z0.s, z1.d
; CHECK-NEXT: ret
  %splat = call <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64 %b)
  %out = call <vscale x 4 x i1> @llvm.aarch64.sve.cmpls.wide.nxv4i32(<vscale x 4 x i1> %pg,
                                                                     <vscale x 4 x i32> %a,
                                                                     <vscale x 2 x i64> %splat)
  ret <vscale x 4 x i1> %out
}


declare <vscale x 16 x i1> @llvm.aarch64.sve.cmpeq.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.cmpeq.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.cmpeq.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.cmpeq.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.cmpeq.wide.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 2 x i64>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.cmpeq.wide.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 2 x i64>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.cmpeq.wide.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 2 x i64>)

declare <vscale x 16 x i1> @llvm.aarch64.sve.cmpge.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.cmpge.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.cmpge.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.cmpge.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.cmpge.wide.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 2 x i64>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.cmpge.wide.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 2 x i64>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.cmpge.wide.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 2 x i64>)

declare <vscale x 16 x i1> @llvm.aarch64.sve.cmpgt.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.cmpgt.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.cmpgt.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.cmpgt.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.cmpgt.wide.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 2 x i64>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.cmpgt.wide.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 2 x i64>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.cmpgt.wide.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 2 x i64>)

declare <vscale x 16 x i1> @llvm.aarch64.sve.cmphi.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.cmphi.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.cmphi.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.cmphi.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.cmphi.wide.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 2 x i64>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.cmphi.wide.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 2 x i64>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.cmphi.wide.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 2 x i64>)

declare <vscale x 16 x i1> @llvm.aarch64.sve.cmphs.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.cmphs.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.cmphs.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.cmphs.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.cmphs.wide.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 2 x i64>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.cmphs.wide.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 2 x i64>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.cmphs.wide.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 2 x i64>)

declare <vscale x 16 x i1> @llvm.aarch64.sve.cmple.wide.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 2 x i64>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.cmple.wide.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 2 x i64>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.cmple.wide.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 2 x i64>)

declare <vscale x 16 x i1> @llvm.aarch64.sve.cmplo.wide.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 2 x i64>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.cmplo.wide.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 2 x i64>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.cmplo.wide.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 2 x i64>)

declare <vscale x 16 x i1> @llvm.aarch64.sve.cmpls.wide.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 2 x i64>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.cmpls.wide.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 2 x i64>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.cmpls.wide.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 2 x i64>)

declare <vscale x 16 x i1> @llvm.aarch64.sve.cmplt.wide.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 2 x i64>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.cmplt.wide.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 2 x i64>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.cmplt.wide.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 2 x i64>)

declare <vscale x 16 x i1> @llvm.aarch64.sve.cmpne.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.cmpne.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.cmpne.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.cmpne.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.cmpne.wide.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 2 x i64>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.cmpne.wide.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 2 x i64>)
declare <vscale x 4 x i1> @llvm.aarch64.sve.cmpne.wide.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 2 x i64>)

declare <vscale x 2 x i64> @llvm.aarch64.sve.dup.x.nxv2i64(i64)
