; RUN: llc -mtriple=aarch64-linux-gnu < %s | FileCheck %s

;
; AND
;

define <vscale x 16 x i8> @and_i8(<vscale x 16 x i8> %a) #0 {
; CHECK-LABEL: and_i8:
; CHECK: and z0.b, z0.b, #0x7
; CHECK-NEXT: ret
  %pg = shufflevector <vscale x 16 x i1> insertelement (<vscale x 16 x i1> undef, i1 true, i32 0), <vscale x 16 x i1> undef, <vscale x 16 x i32> zeroinitializer
  %b = shufflevector <vscale x 16 x i8> insertelement (<vscale x 16 x i8> undef, i8 7, i32 0), <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.and.nxv16i8(<vscale x 16 x i1> %pg,
                                                               <vscale x 16 x i8> %a,
                                                               <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @and_i16(<vscale x 8 x i16> %a) #0 {
; CHECK-LABEL: and_i16:
; CHECK: and z0.h, z0.h, #0xf0
; CHECK-NEXT: ret
  %pg = shufflevector <vscale x 8 x i1> insertelement (<vscale x 8 x i1> undef, i1 true, i32 0), <vscale x 8 x i1> undef, <vscale x 8 x i32> zeroinitializer
  %b = shufflevector <vscale x 8 x i16> insertelement (<vscale x 8 x i16> undef, i16 240, i32 0), <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.and.nxv8i16(<vscale x 8 x i1> %pg,
                                                               <vscale x 8 x i16> %a,
                                                               <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @and_i32(<vscale x 4 x i32> %a) #0 {
; CHECK-LABEL: and_i32:
; CHECK: and z0.s, z0.s, #0xffff00
; CHECK-NEXT: ret
  %pg = shufflevector <vscale x 4 x i1> insertelement (<vscale x 4 x i1> undef, i1 true, i32 0), <vscale x 4 x i1> undef, <vscale x 4 x i32> zeroinitializer
  %b = shufflevector <vscale x 4 x i32> insertelement (<vscale x 4 x i32> undef, i32 16776960, i32 0), <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.and.nxv4i32(<vscale x 4 x i1> %pg,
                                                               <vscale x 4 x i32> %a,
                                                               <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @and_i64(<vscale x 2 x i64> %a) #0 {
; CHECK-LABEL: and_i64:
; CHECK: and z0.d, z0.d, #0xfffc000000000000
; CHECK-NEXT: ret
  %pg = shufflevector <vscale x 2 x i1> insertelement (<vscale x 2 x i1> undef, i1 true, i32 0), <vscale x 2 x i1> undef, <vscale x 2 x i32> zeroinitializer
  %b = shufflevector <vscale x 2 x i64> insertelement (<vscale x 2 x i64> undef, i64 18445618173802708992, i32 0), <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.and.nxv2i64(<vscale x 2 x i1> %pg,
                                                               <vscale x 2 x i64> %a,
                                                               <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

;
; BIC
;

define <vscale x 16 x i8> @bic_i8(<vscale x 16 x i8> %a) #0 {
; CHECK-LABEL: bic_i8:
; CHECK: and z0.b, z0.b, #0x1
; CHECK-NEXT: ret
  %pg = shufflevector <vscale x 16 x i1> insertelement (<vscale x 16 x i1> undef, i1 true, i32 0), <vscale x 16 x i1> undef, <vscale x 16 x i32> zeroinitializer
  %b = shufflevector <vscale x 16 x i8> insertelement (<vscale x 16 x i8> undef, i8 254, i32 0), <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.bic.nxv16i8(<vscale x 16 x i1> %pg,
                                                               <vscale x 16 x i8> %a,
                                                               <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @bic_i16(<vscale x 8 x i16> %a) #0 {
; CHECK-LABEL: bic_i16:
; CHECK: and z0.h, z0.h, #0x1
; CHECK-NEXT: ret
  %pg = shufflevector <vscale x 8 x i1> insertelement (<vscale x 8 x i1> undef, i1 true, i32 0), <vscale x 8 x i1> undef, <vscale x 8 x i32> zeroinitializer
  %b = shufflevector <vscale x 8 x i16> insertelement (<vscale x 8 x i16> undef, i16 65534, i32 0), <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.bic.nxv8i16(<vscale x 8 x i1> %pg,
                                                               <vscale x 8 x i16> %a,
                                                               <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @bic_i32(<vscale x 4 x i32> %a) #0 {
; CHECK-LABEL: bic_i32:
; CHECK: and z0.s, z0.s, #0xff0000ff
; CHECK-NEXT: ret
  %pg = shufflevector <vscale x 4 x i1> insertelement (<vscale x 4 x i1> undef, i1 true, i32 0), <vscale x 4 x i1> undef, <vscale x 4 x i32> zeroinitializer
  %b = shufflevector <vscale x 4 x i32> insertelement (<vscale x 4 x i32> undef, i32 16776960, i32 0), <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.bic.nxv4i32(<vscale x 4 x i1> %pg,
                                                               <vscale x 4 x i32> %a,
                                                               <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @bic_i64(<vscale x 2 x i64> %a) #0 {
; CHECK-LABEL: bic_i64:
; CHECK: and z0.d, z0.d, #0x3ffffffffffff
; CHECK-NEXT: ret
  %pg = shufflevector <vscale x 2 x i1> insertelement (<vscale x 2 x i1> undef, i1 true, i32 0), <vscale x 2 x i1> undef, <vscale x 2 x i32> zeroinitializer
  %b = shufflevector <vscale x 2 x i64> insertelement (<vscale x 2 x i64> undef, i64 18445618173802708992, i32 0), <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.bic.nxv2i64(<vscale x 2 x i1> %pg,
                                                               <vscale x 2 x i64> %a,
                                                               <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

;
; EOR
;

define <vscale x 16 x i8> @eor_i8(<vscale x 16 x i8> %a) #0 {
; CHECK-LABEL: eor_i8:
; CHECK: eor z0.b, z0.b, #0xf
; CHECK-NEXT: ret
  %pg = shufflevector <vscale x 16 x i1> insertelement (<vscale x 16 x i1> undef, i1 true, i32 0), <vscale x 16 x i1> undef, <vscale x 16 x i32> zeroinitializer
  %b = shufflevector <vscale x 16 x i8> insertelement (<vscale x 16 x i8> undef, i8 15, i32 0), <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.eor.nxv16i8(<vscale x 16 x i1> %pg,
                                                               <vscale x 16 x i8> %a,
                                                               <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @eor_i16(<vscale x 8 x i16> %a) #0 {
; CHECK-LABEL: eor_i16:
; CHECK: eor z0.h, z0.h, #0xfc07
; CHECK-NEXT: ret
  %pg = shufflevector <vscale x 8 x i1> insertelement (<vscale x 8 x i1> undef, i1 true, i32 0), <vscale x 8 x i1> undef, <vscale x 8 x i32> zeroinitializer
  %b = shufflevector <vscale x 8 x i16> insertelement (<vscale x 8 x i16> undef, i16 64519, i32 0), <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.eor.nxv8i16(<vscale x 8 x i1> %pg,
                                                               <vscale x 8 x i16> %a,
                                                               <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @eor_i32(<vscale x 4 x i32> %a) #0 {
; CHECK-LABEL: eor_i32:
; CHECK: eor z0.s, z0.s, #0xffff00
; CHECK-NEXT: ret
  %pg = shufflevector <vscale x 4 x i1> insertelement (<vscale x 4 x i1> undef, i1 true, i32 0), <vscale x 4 x i1> undef, <vscale x 4 x i32> zeroinitializer
  %b = shufflevector <vscale x 4 x i32> insertelement (<vscale x 4 x i32> undef, i32 16776960, i32 0), <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.eor.nxv4i32(<vscale x 4 x i1> %pg,
                                                               <vscale x 4 x i32> %a,
                                                               <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @eor_i64(<vscale x 2 x i64> %a) #0 {
; CHECK-LABEL: eor_i64:
; CHECK: eor z0.d, z0.d, #0x1000000000000
; CHECK-NEXT: ret
  %pg = shufflevector <vscale x 2 x i1> insertelement (<vscale x 2 x i1> undef, i1 true, i32 0), <vscale x 2 x i1> undef, <vscale x 2 x i32> zeroinitializer
  %b = shufflevector <vscale x 2 x i64> insertelement (<vscale x 2 x i64> undef, i64 281474976710656, i32 0), <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.eor.nxv2i64(<vscale x 2 x i1> %pg,
                                                               <vscale x 2 x i64> %a,
                                                               <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

;
; ORR
;

define <vscale x 16 x i8> @orr_i8(<vscale x 16 x i8> %a) #0 {
; CHECK-LABEL: orr_i8:
; CHECK: orr z0.b, z0.b, #0x6
; CHECK-NEXT: ret
  %pg = shufflevector <vscale x 16 x i1> insertelement (<vscale x 16 x i1> undef, i1 true, i32 0), <vscale x 16 x i1> undef, <vscale x 16 x i32> zeroinitializer
  %b = shufflevector <vscale x 16 x i8> insertelement (<vscale x 16 x i8> undef, i8 6, i32 0), <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.orr.nxv16i8(<vscale x 16 x i1> %pg,
                                                               <vscale x 16 x i8> %a,
                                                               <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @orr_i16(<vscale x 8 x i16> %a) #0 {
; CHECK-LABEL: orr_i16:
; CHECK: orr z0.h, z0.h, #0x8001
; CHECK-NEXT: ret
  %pg = shufflevector <vscale x 8 x i1> insertelement (<vscale x 8 x i1> undef, i1 true, i32 0), <vscale x 8 x i1> undef, <vscale x 8 x i32> zeroinitializer
  %b = shufflevector <vscale x 8 x i16> insertelement (<vscale x 8 x i16> undef, i16 32769, i32 0), <vscale x 8 x i16> undef, <vscale x 8 x i32> zeroinitializer
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.orr.nxv8i16(<vscale x 8 x i1> %pg,
                                                               <vscale x 8 x i16> %a,
                                                               <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @orr_i32(<vscale x 4 x i32> %a) #0 {
; CHECK-LABEL: orr_i32:
; CHECK: orr z0.s, z0.s, #0xffff
; CHECK-NEXT: ret
  %pg = shufflevector <vscale x 4 x i1> insertelement (<vscale x 4 x i1> undef, i1 true, i32 0), <vscale x 4 x i1> undef, <vscale x 4 x i32> zeroinitializer
  %b = shufflevector <vscale x 4 x i32> insertelement (<vscale x 4 x i32> undef, i32 65535, i32 0), <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.orr.nxv4i32(<vscale x 4 x i1> %pg,
                                                               <vscale x 4 x i32> %a,
                                                               <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @orr_i64(<vscale x 2 x i64> %a) #0 {
; CHECK-LABEL: orr_i64:
; CHECK: orr z0.d, z0.d, #0x7ffc000000000000
; CHECK-NEXT: ret
  %pg = shufflevector <vscale x 2 x i1> insertelement (<vscale x 2 x i1> undef, i1 true, i32 0), <vscale x 2 x i1> undef, <vscale x 2 x i32> zeroinitializer
  %b = shufflevector <vscale x 2 x i64> insertelement (<vscale x 2 x i64> undef, i64 9222246136947933184, i32 0), <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.orr.nxv2i64(<vscale x 2 x i1> %pg,
                                                               <vscale x 2 x i64> %a,
                                                               <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

; As orr_i32 but where pg is i8 based and thus compatible for i32.
define <vscale x 4 x i32> @orr_i32_ptrue_all_b(<vscale x 4 x i32> %a) #0 {
; CHECK-LABEL: orr_i32_ptrue_all_b:
; CHECK: orr z0.s, z0.s, #0xffff
; CHECK-NEXT: ret
  %pg.b = tail call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 31)
  %pg.s = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg.b)
  %b = tail call <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32 65535)
  %out = tail call <vscale x 4 x i32> @llvm.aarch64.sve.orr.nxv4i32(<vscale x 4 x i1> %pg.s,
                                                                    <vscale x 4 x i32> %a,
                                                                    <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

; As orr_i32 but where pg is i16 based and thus compatible for i32.
define <vscale x 4 x i32> @orr_i32_ptrue_all_h(<vscale x 4 x i32> %a) #0 {
; CHECK-LABEL: orr_i32_ptrue_all_h:
; CHECK: orr z0.s, z0.s, #0xffff
; CHECK-NEXT: ret
  %pg.h = tail call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
  %pg.b = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %pg.h)
  %pg.s = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg.b)
  %b = tail call <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32 65535)
  %out = tail call <vscale x 4 x i32> @llvm.aarch64.sve.orr.nxv4i32(<vscale x 4 x i1> %pg.s,
                                                                    <vscale x 4 x i32> %a,
                                                                    <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

; As orr_i32 but where pg is i64 based, which is not compatibile for i32 and
; thus inactive lanes are important and the immediate form cannot be used.
define <vscale x 4 x i32> @orr_i32_ptrue_all_d(<vscale x 4 x i32> %a) #0 {
; CHECK-LABEL: orr_i32_ptrue_all_d:
; CHECK-DAG: mov [[IMM:w[0-9]+]], #65535
; CHECK-DAG: ptrue [[PG:p[0-9]+]].d
; CHECK-DAG: mov [[DUP:z[0-9]+]].s, [[IMM]]
; CHECK-DAG: orr z0.s, [[PG]]/m, z0.s, [[DUP]].s
; CHECK-NEXT: ret
  %pg.d = tail call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  %pg.b = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %pg.d)
  %pg.s = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg.b)
  %b = tail call <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32 65535)
  %out = tail call <vscale x 4 x i32> @llvm.aarch64.sve.orr.nxv4i32(<vscale x 4 x i1> %pg.s,
                                                                    <vscale x 4 x i32> %a,
                                                                    <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

declare <vscale x 16 x i8> @llvm.aarch64.sve.and.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.and.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.and.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.and.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.bic.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.bic.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.bic.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.bic.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.eor.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.eor.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.eor.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.eor.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 16 x i8> @llvm.aarch64.sve.orr.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 8 x i16> @llvm.aarch64.sve.orr.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.orr.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 2 x i64> @llvm.aarch64.sve.orr.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, <vscale x 2 x i64>)

declare <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1>)

declare <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 16 x i1>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1>)

declare <vscale x 4 x i32> @llvm.aarch64.sve.dup.x.nxv4i32(i32)

declare <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32)
declare <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32)
declare <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32)
declare <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32)

attributes #0 = { "target-features"="+sve" }
