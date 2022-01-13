; RUN: llc -aarch64-sve-vector-bits-min=128  -asm-verbose=0 < %s | FileCheck %s -check-prefix=NO_SVE
; RUN: llc -aarch64-sve-vector-bits-min=256  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_EQ_256
; RUN: llc -aarch64-sve-vector-bits-min=384  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_EQ_256
; RUN: llc -aarch64-sve-vector-bits-min=512  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_EQ_512
; RUN: llc -aarch64-sve-vector-bits-min=640  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_EQ_512
; RUN: llc -aarch64-sve-vector-bits-min=768  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_EQ_512
; RUN: llc -aarch64-sve-vector-bits-min=896  -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_EQ_512
; RUN: llc -aarch64-sve-vector-bits-min=1024 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024,VBITS_EQ_1024
; RUN: llc -aarch64-sve-vector-bits-min=1152 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024,VBITS_EQ_1024
; RUN: llc -aarch64-sve-vector-bits-min=1280 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024,VBITS_EQ_1024
; RUN: llc -aarch64-sve-vector-bits-min=1408 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024,VBITS_EQ_1024
; RUN: llc -aarch64-sve-vector-bits-min=1536 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024,VBITS_EQ_1024
; RUN: llc -aarch64-sve-vector-bits-min=1664 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024,VBITS_EQ_1024
; RUN: llc -aarch64-sve-vector-bits-min=1792 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024,VBITS_EQ_1024
; RUN: llc -aarch64-sve-vector-bits-min=1920 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024,VBITS_EQ_1024
; RUN: llc -aarch64-sve-vector-bits-min=2048 -asm-verbose=0 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024,VBITS_GE_2048,VBITS_EQ_2048

target triple = "aarch64-unknown-linux-gnu"

; Don't use SVE when its registers are no bigger than NEON.
; NO_SVE-NOT: ptrue

;
; SDIV
;

; Vector vXi8 sdiv are not legal for NEON so use SVE when available.
define <8 x i8> @sdiv_v8i8(<8 x i8> %op1, <8 x i8> %op2) #0 {
; CHECK-LABEL: sdiv_v8i8:
; CHECK: ptrue [[PG0:p[0-9]+]].s, vl8
; CHECK-NEXT: sunpklo [[OP2_LO:z[0-9]+]].h, z1.b
; CHECK-NEXT: sunpklo [[OP1_LO:z[0-9]+]].h, z0.b
; CHECK-NEXT: sunpklo [[OP2_LO_LO:z[0-9]+]].s, z1.h
; CHECK-NEXT: sunpklo [[OP1_LO_LO:z[0-9]+]].s, z0.h
; CHECK-NEXT: sdiv [[DIV:z[0-9]+]].s, [[PG0]]/m, [[OP1_LO_LO]].s, [[OP2_LO_LO]].s
; CHECK-NEXT: uzp1 [[RES:z[0-9]+]].h, [[DIV]].h, [[DIV]].h
; CHECK-NEXT: umov [[SCALAR0:w[0-9]+]], [[VEC:v[0-9]+]].h[0]
; CHECK-NEXT: umov [[SCALAR1:w[0-9]+]], [[VEC]].h[1]
; CHECK-NEXT: fmov s0, [[SCALAR0]]
; CHECK-NEXT: umov [[SCALAR2:w[0-9]+]], [[VEC]].h[2]
; CHECK-NEXT: mov [[FINAL:v[0-9]+]].b[1], [[SCALAR1]]
; CHECK-NEXT: mov [[FINAL]].b[2], [[SCALAR2]]
; CHECK-NEXT: umov [[SCALAR3:w[0-9]+]], [[VEC]].h[3]
; CHECK-NEXT: mov [[FINAL]].b[3], [[SCALAR3]]
; CHECK-NEXT: umov [[SCALAR4:w[0-9]+]], [[VEC]].h[4]
; CHECK-NEXT: mov [[FINAL]].b[4], [[SCALAR4]]
; CHECK-NEXT: umov [[SCALAR5:w[0-9]+]], [[VEC]].h[5]
; CHECK-NEXT: mov [[FINAL]].b[5], [[SCALAR5]]
; CHECK-NEXT: umov [[SCALAR6:w[0-9]+]], [[VEC]].h[6]
; CHECK-NEXT: mov [[FINAL]].b[6], [[SCALAR6]]
; CHECK-NEXT: umov [[SCALAR7:w[0-9]+]], [[VEC]].h[7]
; CHECK-NEXT: mov [[FINAL]].b[7], [[SCALAR7]]
; CHECK: ret
  %res = sdiv <8 x i8> %op1, %op2
  ret <8 x i8> %res
}

define <16 x i8> @sdiv_v16i8(<16 x i8> %op1, <16 x i8> %op2) #0 {
; CHECK-LABEL: sdiv_v16i8:

; HALF VECTOR:
; VBITS_EQ_256: ptrue [[PG:p[0-9]+]].s, vl8
; VBITS_EQ_256-NEXT: sunpklo [[OP2_LO:z[0-9]+]].h, z1.b
; VBITS_EQ_256-NEXT: sunpklo [[OP1_LO:z[0-9]+]].h, z0.b
; VBITS_EQ_256-NEXT: sunpkhi [[OP2_HI:z[0-9]+]].s, z1.h
; VBITS_EQ_256-NEXT: sunpkhi [[OP1_HI:z[0-9]+]].s, z0.h
; VBITS_EQ_256-NEXT: sunpklo [[OP2_LO_LO:z[0-9]+]].s, z1.h
; VBITS_EQ_256-NEXT: sunpklo [[OP1_LO_LO:z[0-9]+]].s, z0.h
; VBITS_EQ_256-NEXT: sdivr [[DIV1:z[0-9]+]].s, [[PG]]/m, [[OP2_HI]].s, [[OP1_HI]].s
; VBITS_EQ_256-NEXT: sdiv [[DIV2:z[0-9]+]].s, [[PG]]/m, [[OP1_LO]].s, [[OP2_LO]].s
; VBITS_EQ_256-NEXT: uzp1 [[RES1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_256-NEXT: uzp1 [[RES2:z[0-9]+]].b, [[RES1]].b, [[RES1]].b

; QUARTER VECTOR OR SMALLER:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: sunpklo [[OP2_LO:z[0-9]+]].h, z1.b
; VBITS_GE_512-NEXT: sunpklo [[OP1_LO:z[0-9]+]].h, z0.b
; VBITS_GE_512-NEXT: sunpklo [[OP2_LO_LO:z[0-9]+]].s, z1.h
; VBITS_GE_512-NEXT: sunpklo [[OP1_LO_LO:z[0-9]+]].s, z0.h
; VBITS_GE_512-NEXT: sdiv [[DIV:z[0-9]+]].s, [[PG]]/m, [[OP1_LO_LO]].s, [[OP2_LO_LO]].s
; VBITS_GE_512-NEXT: uzp1 [[RES1:z[0-9]+]].h, [[DIV]].h, [[DIV]].h
; VBITS_GE_512-NEXT: uzp1 [[RES2:z[0-9]+]].b, [[RES1]].b, [[RES1]].b
; CHECK: ret
  %res = sdiv <16 x i8> %op1, %op2
  ret <16 x i8> %res
}

define void @sdiv_v32i8(<32 x i8>* %a, <32 x i8>* %b) #0 {
; CHECK-LABEL: sdiv_v32i8:

; FULL VECTOR:
; VBITS_EQ_256: ptrue [[PG1:p[0-9]+]].b, vl32
; VBITS_EQ_256-NEXT: ptrue [[PG2:p[0-9]+]].s, vl8
; VBITS_EQ_256-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_EQ_256-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_EQ_256-NEXT: sunpkhi [[OP2_HI:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_256-NEXT: sunpkhi [[OP1_HI:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_256-NEXT: sunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_256-NEXT: sunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_256-NEXT: sunpkhi [[OP2_HI_HI:z[0-9]+]].s, [[OP2_HI]].h
; VBITS_EQ_256-NEXT: sunpkhi [[OP1_HI_HI:z[0-9]+]].s, [[OP1_HI]].h
; VBITS_EQ_256-NEXT: sunpklo [[OP2_HI_LO:z[0-9]+]].s, [[OP2_HI]].h
; VBITS_EQ_256-NEXT: sunpklo [[OP1_HI_LO:z[0-9]+]].s, [[OP1_HI]].h
; VBITS_EQ_256-NEXT: sdivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI_HI]].s, [[OP1_HI_HI]].s
; VBITS_EQ_256-NEXT: sunpkhi [[OP2_LO_HI:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_256-NEXT: sdivr [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI_LO]].s, [[OP1_HI_LO]].s
; VBITS_EQ_256-NEXT: sunpkhi [[OP1_LO_HI:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_256-NEXT: sunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_256-NEXT: sunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_256-NEXT: sdiv [[DIV3:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_HI]].s, [[OP2_LO_HI]].s
; VBITS_EQ_256-NEXT: sdiv [[DIV4:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_LO]].s, [[OP2_LO_LO]].s
; VBITS_EQ_256-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_256-NEXT: uzp1 [[UZP2:z[0-9]+]].h, [[DIV4]].h, [[DIV3]].h
; VBITS_EQ_256-NEXT: uzp1 [[UZP3:z[0-9]+]].b, [[UZP2]].b, [[UZP1]].b
; VBITS_EQ_256-NEXT: st1b { [[UZP3:z[0-9]+]].b }, [[PG1]], [x0]

; HALF VECTOR:
; VBITS_EQ_512: ptrue [[PG1:p[0-9]+]].b, vl32
; VBITS_EQ_512-NEXT: ptrue [[PG2:p[0-9]+]].s, vl16
; VBITS_EQ_512-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_EQ_512-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_EQ_512-NEXT: sunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_512-NEXT: sunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_512-NEXT: sunpkhi [[OP2_LO_HI:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_512-NEXT: sunpkhi [[OP1_LO_HI:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_512-NEXT: sunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_512-NEXT: sunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_512-NEXT: sdivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_LO_HI]].s, [[OP1_LO_HI]].s
; VBITS_EQ_512-NEXT: sdiv [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_LO]].s, [[OP2_LO_LO]].s
; VBITS_EQ_512-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_512-NEXT: uzp1 [[UZP2:z[0-9]+]].b, [[UZP1]].b, [[UZP1]].b
; VBITS_EQ_512-NEXT: st1b { [[UZP2:z[0-9]+]].b }, [[PG1]], [x0]

; QUARTER VECTOR OR SMALLER:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].b, vl32
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_GE_1024-NEXT: sunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_GE_1024-NEXT: sunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_GE_1024-NEXT: sunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_GE_1024-NEXT: sunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_GE_1024-NEXT: sdiv [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_LO]].s, [[OP2_LO_LO]].s
; VBITS_GE_1024-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV1]].h, [[DIV1]].h
; VBITS_GE_1024-NEXT: uzp1 [[UZP2:z[0-9]+]].b, [[UZP1]].b, [[UZP1]].b
; VBITS_GE_1024-NEXT: st1b { [[UZP2:z[0-9]+]].b }, [[PG1]], [x0]
; CHECK: ret
  %op1 = load <32 x i8>, <32 x i8>* %a
  %op2 = load <32 x i8>, <32 x i8>* %b
  %res = sdiv <32 x i8> %op1, %op2
  store <32 x i8> %res, <32 x i8>* %a
  ret void
}

define void @sdiv_v64i8(<64 x i8>* %a, <64 x i8>* %b) #0 {
; CHECK-LABEL: sdiv_v64i8:

; FULL VECTOR:
; VBITS_EQ_512: ptrue [[PG1:p[0-9]+]].b, vl64
; VBITS_EQ_512-NEXT: ptrue [[PG2:p[0-9]+]].s, vl16
; VBITS_EQ_512-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_EQ_512-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_EQ_512-NEXT: sunpkhi [[OP2_HI:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_512-NEXT: sunpkhi [[OP1_HI:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_512-NEXT: sunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_512-NEXT: sunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_512-NEXT: sunpkhi [[OP2_HI_HI:z[0-9]+]].s, [[OP2_HI]].h
; VBITS_EQ_512-NEXT: sunpkhi [[OP1_HI_HI:z[0-9]+]].s, [[OP1_HI]].h
; VBITS_EQ_512-NEXT: sunpklo [[OP2_HI_LO:z[0-9]+]].s, [[OP2_HI]].h
; VBITS_EQ_512-NEXT: sunpklo [[OP1_HI_LO:z[0-9]+]].s, [[OP1_HI]].h
; VBITS_EQ_512-NEXT: sdivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI_HI]].s, [[OP1_HI_HI]].s
; VBITS_EQ_512-NEXT: sunpkhi [[OP2_LO_HI:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_512-NEXT: sdivr [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI_LO]].s, [[OP1_HI_LO]].s
; VBITS_EQ_512-NEXT: sunpkhi [[OP1_LO_HI:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_512-NEXT: sunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_512-NEXT: sunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_512-NEXT: sdiv [[DIV3:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_HI]].s, [[OP2_LO_HI]].s
; VBITS_EQ_512-NEXT: sdiv [[DIV4:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_LO]].s, [[OP2_LO_LO]].s
; VBITS_EQ_512-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_512-NEXT: uzp1 [[UZP2:z[0-9]+]].h, [[DIV4]].h, [[DIV3]].h
; VBITS_EQ_512-NEXT: uzp1 [[UZP3:z[0-9]+]].b, [[UZP2]].b, [[UZP1]].b
; VBITS_EQ_512-NEXT: st1b { [[UZP3:z[0-9]+]].b }, [[PG1]], [x0]

; HALF VECTOR:
; VBITS_EQ_1024: ptrue [[PG1:p[0-9]+]].b, vl64
; VBITS_EQ_1024-NEXT: ptrue [[PG2:p[0-9]+]].s, vl32
; VBITS_EQ_1024-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_EQ_1024-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_EQ_1024-NEXT: sunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_1024-NEXT: sunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_1024-NEXT: sunpkhi [[OP2_LO_HI:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_1024-NEXT: sunpkhi [[OP1_LO_HI:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_1024-NEXT: sunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_1024-NEXT: sunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_1024-NEXT: sdivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_LO_HI]].s, [[OP1_LO_HI]].s
; VBITS_EQ_1024-NEXT: sdiv [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_LO]].s, [[OP2_LO_LO]].s
; VBITS_EQ_1024-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_1024-NEXT: uzp1 [[UZP2:z[0-9]+]].b, [[UZP1]].b, [[UZP1]].b
; VBITS_EQ_1024-NEXT: st1b { [[UZP2:z[0-9]+]].b }, [[PG1]], [x0]

; QUARTER VECTOR OR SMALLER:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].b, vl64
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_GE_2048-NEXT: sunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_GE_2048-NEXT: sunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_GE_2048-NEXT: sunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_GE_2048-NEXT: sunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_GE_2048-NEXT: sdiv [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_LO]].s, [[OP2_LO_LO]].s
; VBITS_GE_2048-NEXT: uzp1 [[RES1:z[0-9]+]].h, [[DIV1]].h, [[DIV1]].h
; VBITS_GE_2048-NEXT: uzp1 [[RES2:z[0-9]+]].b, [[RES1]].b, [[RES1]].b
; VBITS_GE_2048-NEXT: st1b { [[RES2]].b }, [[PG1]], [x0]
; CHECK: ret
  %op1 = load <64 x i8>, <64 x i8>* %a
  %op2 = load <64 x i8>, <64 x i8>* %b
  %res = sdiv <64 x i8> %op1, %op2
  store <64 x i8> %res, <64 x i8>* %a
  ret void
}

define void @sdiv_v128i8(<128 x i8>* %a, <128 x i8>* %b) #0 {
; CHECK-LABEL: sdiv_v128i8:

; FULL VECTOR:
; VBITS_EQ_1024: ptrue [[PG1:p[0-9]+]].b, vl128
; VBITS_EQ_1024-NEXT: ptrue [[PG2:p[0-9]+]].s, vl32
; VBITS_EQ_1024-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_EQ_1024-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_EQ_1024-NEXT: sunpkhi [[OP2_HI:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_1024-NEXT: sunpkhi [[OP1_HI:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_1024-NEXT: sunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_1024-NEXT: sunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_1024-NEXT: sunpkhi [[OP2_HI_HI:z[0-9]+]].s, [[OP2_HI]].h
; VBITS_EQ_1024-NEXT: sunpkhi [[OP1_HI_HI:z[0-9]+]].s, [[OP1_HI]].h
; VBITS_EQ_1024-NEXT: sunpklo [[OP2_HI_LO:z[0-9]+]].s, [[OP2_HI]].h
; VBITS_EQ_1024-NEXT: sunpklo [[OP1_HI_LO:z[0-9]+]].s, [[OP1_HI]].h
; VBITS_EQ_1024-NEXT: sdivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI_HI]].s, [[OP1_HI_HI]].s
; VBITS_EQ_1024-NEXT: sunpkhi [[OP2_LO_HI:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_1024-NEXT: sdivr [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI_LO]].s, [[OP1_HI_LO]].s
; VBITS_EQ_1024-NEXT: sunpkhi [[OP1_LO_HI:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_1024-NEXT: sunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_1024-NEXT: sunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_1024-NEXT: sdiv [[DIV3:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_HI]].s, [[OP2_LO_HI]].s
; VBITS_EQ_1024-NEXT: sdiv [[DIV4:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_LO]].s, [[OP2_LO_LO]].s
; VBITS_EQ_1024-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_1024-NEXT: uzp1 [[UZP2:z[0-9]+]].h, [[DIV4]].h, [[DIV3]].h
; VBITS_EQ_1024-NEXT: uzp1 [[UZP3:z[0-9]+]].b, [[UZP2]].b, [[UZP1]].b
; VBITS_EQ_1024-NEXT: st1b { [[UZP3:z[0-9]+]].b }, [[PG1]], [x0]

; HALF VECTOR:
; VBITS_EQ_2048: ptrue [[PG1:p[0-9]+]].b, vl128
; VBITS_EQ_2048-NEXT: ptrue [[PG2:p[0-9]+]].s, vl64
; VBITS_EQ_2048-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_EQ_2048-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_EQ_2048-NEXT: sunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_2048-NEXT: sunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_2048-NEXT: sunpkhi [[OP2_LO_HI:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_2048-NEXT: sunpkhi [[OP1_LO_HI:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_2048-NEXT: sunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_2048-NEXT: sunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_2048-NEXT: sdivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_LO_HI]].s, [[OP1_LO_HI]].s
; VBITS_EQ_2048-NEXT: sdiv [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_LO]].s, [[OP2_LO_LO]].s
; VBITS_EQ_2048-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_2048-NEXT: uzp1 [[UZP2:z[0-9]+]].b, [[UZP1]].b, [[UZP1]].b
; VBITS_EQ_2048-NEXT: st1b { [[UZP2:z[0-9]+]].b }, [[PG1]], [x0]
; CHECK: ret
  %op1 = load <128 x i8>, <128 x i8>* %a
  %op2 = load <128 x i8>, <128 x i8>* %b
  %res = sdiv <128 x i8> %op1, %op2
  store <128 x i8> %res, <128 x i8>* %a
  ret void
}

define void @sdiv_v256i8(<256 x i8>* %a, <256 x i8>* %b) #0 {
; CHECK-LABEL: sdiv_v256i8:

; FULL VECTOR:
; VBITS_EQ_2048: ptrue [[PG1:p[0-9]+]].b, vl256
; VBITS_EQ_2048-NEXT: ptrue [[PG2:p[0-9]+]].s, vl64
; VBITS_EQ_2048-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_EQ_2048-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_EQ_2048-NEXT: sunpkhi [[OP2_HI:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_2048-NEXT: sunpkhi [[OP1_HI:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_2048-NEXT: sunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_2048-NEXT: sunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_2048-NEXT: sunpkhi [[OP2_HI_HI:z[0-9]+]].s, [[OP2_HI]].h
; VBITS_EQ_2048-NEXT: sunpkhi [[OP1_HI_HI:z[0-9]+]].s, [[OP1_HI]].h
; VBITS_EQ_2048-NEXT: sunpklo [[OP2_HI_LO:z[0-9]+]].s, [[OP2_HI]].h
; VBITS_EQ_2048-NEXT: sunpklo [[OP1_HI_LO:z[0-9]+]].s, [[OP1_HI]].h
; VBITS_EQ_2048-NEXT: sdivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI_HI]].s, [[OP1_HI_HI]].s
; VBITS_EQ_2048-NEXT: sunpkhi [[OP2_LO_HI:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_2048-NEXT: sdivr [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI_LO]].s, [[OP1_HI_LO]].s
; VBITS_EQ_2048-NEXT: sunpkhi [[OP1_LO_HI:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_2048-NEXT: sunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_2048-NEXT: sunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_2048-NEXT: sdiv [[DIV3:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_HI]].s, [[OP2_LO_HI]].s
; VBITS_EQ_2048-NEXT: sdiv [[DIV4:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_LO]].s, [[OP2_LO_LO]].s
; VBITS_EQ_2048-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_2048-NEXT: uzp1 [[UZP2:z[0-9]+]].h, [[DIV4]].h, [[DIV3]].h
; VBITS_EQ_2048-NEXT: uzp1 [[UZP3:z[0-9]+]].b, [[UZP2]].b, [[UZP1]].b
; VBITS_EQ_2048-NEXT: st1b { [[UZP3:z[0-9]+]].b }, [[PG1]], [x0]
; CHECK: ret
  %op1 = load <256 x i8>, <256 x i8>* %a
  %op2 = load <256 x i8>, <256 x i8>* %b
  %res = sdiv <256 x i8> %op1, %op2
  store <256 x i8> %res, <256 x i8>* %a
  ret void
}

; Vector vXi16 sdiv are not legal for NEON so use SVE when available.
define <4 x i16> @sdiv_v4i16(<4 x i16> %op1, <4 x i16> %op2) #0 {
; CHECK-LABEL: sdiv_v4i16:
; CHECK: sshll v1.4s, v1.4h, #0
; CHECK-NEXT: sshll v0.4s, v0.4h, #0
; CHECK-NEXT: ptrue [[PG1:p[0-9]+]].s, vl4
; CHECK-NEXT: sdivr [[DIV1:z[0-9]+]].s, [[PG1]]/m, [[OP2:z[0-9]+]].s, [[OP1:z[0-9]+]].s
; CHECK-NEXT: mov w8, v1.s[1]
; CHECK-NEXT: mov w9, v1.s[2]
; CHECK-NEXT: mov v0.16b, v1.16b
; CHECK-NEXT: mov v0.h[1], w8
; CHECK-NEXT: mov w8, v1.s[3]
; CHECK-NEXT: mov v0.h[2], w9
; CHECK-NEXT: mov v0.h[3], w8
; CHECK: ret
  %res = sdiv <4 x i16> %op1, %op2
  ret <4 x i16> %res
}

define <8 x i16> @sdiv_v8i16(<8 x i16> %op1, <8 x i16> %op2) #0 {
; CHECK-LABEL: sdiv_v8i16:
; CHECK: ptrue [[PG1:p[0-9]+]].s, vl8
; CHECK-NEXT: sunpklo [[OP2_LO:z[0-9]+]].s, [[OP2]].h
; CHECK-NEXT: sunpklo [[OP1_LO:z[0-9]+]].s, [[OP1]].h
; CHECK-NEXT: sdiv [[DIV1:z[0-9]+]].s, [[PG1]]/m, [[OP1_LO]].s, [[OP2_LO]].s
; CHECK-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV1]].h, [[DIV1]].h
; CHECK: ret
  %res = sdiv <8 x i16> %op1, %op2
  ret <8 x i16> %res
}

define void @sdiv_v16i16(<16 x i16>* %a, <16 x i16>* %b) #0 {
; CHECK-LABEL: sdiv_v16i16:

; FULL VECTOR:
; VBITS_EQ_256: ptrue [[PG1:p[0-9]+]].h, vl16
; VBITS_EQ_256-NEXT: ptrue [[PG2:p[0-9]+]].s, vl8
; VBITS_EQ_256-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_EQ_256-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; VBITS_EQ_256-NEXT: sunpkhi [[OP2_HI:z[0-9]+]].s, [[OP2]].h
; VBITS_EQ_256-NEXT: sunpkhi [[OP1_HI:z[0-9]+]].s, [[OP1]].h
; VBITS_EQ_256-NEXT: sunpklo [[OP2_LO:z[0-9]+]].s, [[OP2]].h
; VBITS_EQ_256-NEXT: sunpklo [[OP1_LO:z[0-9]+]].s, [[OP1]].h
; VBITS_EQ_256-NEXT: sdivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI]].s, [[OP1_HI]].s
; VBITS_EQ_256-NEXT: sdiv [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO]].s, [[OP2_LO]].s
; VBITS_EQ_256-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_256-NEXT: st1h { [[UZP1:z[0-9]+]].h }, [[PG1]], [x0]

; HALF VECTOR OR SMALLER:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].h, vl16
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; VBITS_GE_512-NEXT: sunpklo [[OP2_LO:z[0-9]+]].s, [[OP2]].h
; VBITS_GE_512-NEXT: sunpklo [[OP1_LO:z[0-9]+]].s, [[OP1]].h
; VBITS_GE_512-NEXT: sdiv [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO]].s, [[OP2_LO]].s
; VBITS_GE_512-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV1]].h, [[DIV1]].h
; VBITS_GE_512-NEXT: st1h { [[UZP1:z[0-9]+]].h }, [[PG1]], [x0]
; CHECK: ret
  %op1 = load <16 x i16>, <16 x i16>* %a
  %op2 = load <16 x i16>, <16 x i16>* %b
  %res = sdiv <16 x i16> %op1, %op2
  store <16 x i16> %res, <16 x i16>* %a
  ret void
}

define void @sdiv_v32i16(<32 x i16>* %a, <32 x i16>* %b) #0 {
; CHECK-LABEL: sdiv_v32i16:

; FULL VECTOR:
; VBITS_EQ_512: ptrue [[PG1:p[0-9]+]].h, vl32
; VBITS_EQ_512-NEXT: ptrue [[PG2:p[0-9]+]].s, vl16
; VBITS_EQ_512-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_EQ_512-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; VBITS_EQ_512-NEXT: sunpkhi [[OP2_HI:z[0-9]+]].s, [[OP2]].h
; VBITS_EQ_512-NEXT: sunpkhi [[OP1_HI:z[0-9]+]].s, [[OP1]].h
; VBITS_EQ_512-NEXT: sunpklo [[OP2_LO:z[0-9]+]].s, [[OP2]].h
; VBITS_EQ_512-NEXT: sunpklo [[OP1_LO:z[0-9]+]].s, [[OP1]].h
; VBITS_EQ_512-NEXT: sdivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI]].s, [[OP1_HI]].s
; VBITS_EQ_512-NEXT: sdiv [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO]].s, [[OP2_LO]].s
; VBITS_EQ_512-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_512-NEXT: st1h { [[UZP1:z[0-9]+]].h }, [[PG1]], [x0]

; HALF VECTOR OR SMALLER:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].h, vl32
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; VBITS_GE_1024-NEXT: sunpklo [[OP2_LO:z[0-9]+]].s, [[OP2]].h
; VBITS_GE_1024-NEXT: sunpklo [[OP1_LO:z[0-9]+]].s, [[OP1]].h
; VBITS_GE_1024-NEXT: sdiv [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO]].s, [[OP2_LO]].s
; VBITS_GE_1024-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV1]].h, [[DIV1]].h
; VBITS_GE_1024-NEXT: st1h { [[UZP1:z[0-9]+]].h }, [[PG1]], [x0]
; CHECK: ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %op2 = load <32 x i16>, <32 x i16>* %b
  %res = sdiv <32 x i16> %op1, %op2
  store <32 x i16> %res, <32 x i16>* %a
  ret void
}

define void @sdiv_v64i16(<64 x i16>* %a, <64 x i16>* %b) #0 {
; CHECK-LABEL: sdiv_v64i16:

; FULL VECTOR:
; VBITS_EQ_1024: ptrue [[PG1:p[0-9]+]].h, vl64
; VBITS_EQ_1024-NEXT: ptrue [[PG2:p[0-9]+]].s, vl32
; VBITS_EQ_1024-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_EQ_1024-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; VBITS_EQ_1024-NEXT: sunpkhi [[OP2_HI:z[0-9]+]].s, [[OP2]].h
; VBITS_EQ_1024-NEXT: sunpkhi [[OP1_HI:z[0-9]+]].s, [[OP1]].h
; VBITS_EQ_1024-NEXT: sunpklo [[OP2_LO:z[0-9]+]].s, [[OP2]].h
; VBITS_EQ_1024-NEXT: sunpklo [[OP1_LO:z[0-9]+]].s, [[OP1]].h
; VBITS_EQ_1024-NEXT: sdivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI]].s, [[OP1_HI]].s
; VBITS_EQ_1024-NEXT: sdiv [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO]].s, [[OP2_LO]].s
; VBITS_EQ_1024-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_1024-NEXT: st1h { [[UZP1:z[0-9]+]].h }, [[PG1]], [x0]

; HALF VECTOR OR SMALLER:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].h, vl64
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; VBITS_GE_2048-NEXT: sunpklo [[OP2_LO:z[0-9]+]].s, [[OP2]].h
; VBITS_GE_2048-NEXT: sunpklo [[OP1_LO:z[0-9]+]].s, [[OP1]].h
; VBITS_GE_2048-NEXT: sdiv [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO]].s, [[OP2_LO]].s
; VBITS_GE_2048-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV1]].h, [[DIV1]].h
; VBITS_GE_2048-NEXT: st1h { [[UZP1:z[0-9]+]].h }, [[PG1]], [x0]
; CHECK: ret
  %op1 = load <64 x i16>, <64 x i16>* %a
  %op2 = load <64 x i16>, <64 x i16>* %b
  %res = sdiv <64 x i16> %op1, %op2
  store <64 x i16> %res, <64 x i16>* %a
  ret void
}

define void @sdiv_v128i16(<128 x i16>* %a, <128 x i16>* %b) #0 {
; CHECK-LABEL: sdiv_v128i16:
; VBITS_EQ_2048: ptrue [[PG1:p[0-9]+]].h, vl128
; VBITS_EQ_2048-NEXT: ptrue [[PG2:p[0-9]+]].s, vl64
; VBITS_EQ_2048-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_EQ_2048-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; VBITS_EQ_2048-NEXT: sunpkhi [[OP2_HI:z[0-9]+]].s, [[OP2]].h
; VBITS_EQ_2048-NEXT: sunpkhi [[OP1_HI:z[0-9]+]].s, [[OP1]].h
; VBITS_EQ_2048-NEXT: sunpklo [[OP2_LO:z[0-9]+]].s, [[OP2]].h
; VBITS_EQ_2048-NEXT: sunpklo [[OP1_LO:z[0-9]+]].s, [[OP1]].h
; VBITS_EQ_2048-NEXT: sdivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI]].s, [[OP1_HI]].s
; VBITS_EQ_2048-NEXT: sdiv [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO]].s, [[OP2_LO]].s
; VBITS_EQ_2048-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_2048-NEXT: st1h { [[UZP1:z[0-9]+]].h }, [[PG1]], [x0]
; CHECK: ret
  %op1 = load <128 x i16>, <128 x i16>* %a
  %op2 = load <128 x i16>, <128 x i16>* %b
  %res = sdiv <128 x i16> %op1, %op2
  store <128 x i16> %res, <128 x i16>* %a
  ret void
}

; Vector v2i32 sdiv are not legal for NEON so use SVE when available.
define <2 x i32> @sdiv_v2i32(<2 x i32> %op1, <2 x i32> %op2) #0 {
; CHECK-LABEL: sdiv_v2i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl2
; CHECK: sdiv z0.s, [[PG]]/m, z0.s, z1.s
; CHECK: ret
  %res = sdiv <2 x i32> %op1, %op2
  ret <2 x i32> %res
}

; Vector v4i32 sdiv are not legal for NEON so use SVE when available.
define <4 x i32> @sdiv_v4i32(<4 x i32> %op1, <4 x i32> %op2) #0 {
; CHECK-LABEL: sdiv_v4i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl4
; CHECK: sdiv z0.s, [[PG]]/m, z0.s, z1.s
; CHECK: ret
  %res = sdiv <4 x i32> %op1, %op2
  ret <4 x i32> %res
}

define void @sdiv_v8i32(<8 x i32>* %a, <8 x i32>* %b) #0 {
; CHECK-LABEL: sdiv_v8i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-NEXT: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-NEXT: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK-NEXT: sdiv [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %op2 = load <8 x i32>, <8 x i32>* %b
  %res = sdiv <8 x i32> %op1, %op2
  store <8 x i32> %res, <8 x i32>* %a
  ret void
}

define void @sdiv_v16i32(<16 x i32>* %a, <16 x i32>* %b) #0 {
; CHECK-LABEL: sdiv_v16i32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: sdiv [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_GE_512-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %op2 = load <16 x i32>, <16 x i32>* %b
  %res = sdiv <16 x i32> %op1, %op2
  store <16 x i32> %res, <16 x i32>* %a
  ret void
}

define void @sdiv_v32i32(<32 x i32>* %a, <32 x i32>* %b) #0 {
; CHECK-LABEL: sdiv_v32i32:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: sdiv [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %op2 = load <32 x i32>, <32 x i32>* %b
  %res = sdiv <32 x i32> %op1, %op2
  store <32 x i32> %res, <32 x i32>* %a
  ret void
}

define void @sdiv_v64i32(<64 x i32>* %a, <64 x i32>* %b) #0 {
; CHECK-LABEL: sdiv_v64i32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: sdiv [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <64 x i32>, <64 x i32>* %a
  %op2 = load <64 x i32>, <64 x i32>* %b
  %res = sdiv <64 x i32> %op1, %op2
  store <64 x i32> %res, <64 x i32>* %a
  ret void
}

; Vector i64 sdiv are not legal for NEON so use SVE when available.
define <1 x i64> @sdiv_v1i64(<1 x i64> %op1, <1 x i64> %op2) #0 {
; CHECK-LABEL: sdiv_v1i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl1
; CHECK: sdiv z0.d, [[PG]]/m, z0.d, z1.d
; CHECK: ret
  %res = sdiv <1 x i64> %op1, %op2
  ret <1 x i64> %res
}

; Vector i64 sdiv are not legal for NEON so use SVE when available.
define <2 x i64> @sdiv_v2i64(<2 x i64> %op1, <2 x i64> %op2) #0 {
; CHECK-LABEL: sdiv_v2i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl2
; CHECK: sdiv z0.d, [[PG]]/m, z0.d, z1.d
; CHECK: ret
  %res = sdiv <2 x i64> %op1, %op2
  ret <2 x i64> %res
}

define void @sdiv_v4i64(<4 x i64>* %a, <4 x i64>* %b) #0 {
; CHECK-LABEL: sdiv_v4i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-NEXT: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK-NEXT: sdiv [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %op2 = load <4 x i64>, <4 x i64>* %b
  %res = sdiv <4 x i64> %op1, %op2
  store <4 x i64> %res, <4 x i64>* %a
  ret void
}

define void @sdiv_v8i64(<8 x i64>* %a, <8 x i64>* %b) #0 {
; CHECK-LABEL: sdiv_v8i64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: sdiv [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_512-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %op2 = load <8 x i64>, <8 x i64>* %b
  %res = sdiv <8 x i64> %op1, %op2
  store <8 x i64> %res, <8 x i64>* %a
  ret void
}

define void @sdiv_v16i64(<16 x i64>* %a, <16 x i64>* %b) #0 {
; CHECK-LABEL: sdiv_v16i64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: sdiv [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %op2 = load <16 x i64>, <16 x i64>* %b
  %res = sdiv <16 x i64> %op1, %op2
  store <16 x i64> %res, <16 x i64>* %a
  ret void
}

define void @sdiv_v32i64(<32 x i64>* %a, <32 x i64>* %b) #0 {
; CHECK-LABEL: sdiv_v32i64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: sdiv [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %op2 = load <32 x i64>, <32 x i64>* %b
  %res = sdiv <32 x i64> %op1, %op2
  store <32 x i64> %res, <32 x i64>* %a
  ret void
}

;
; UDIV
;

; Vector vXi8 udiv are not legal for NEON so use SVE when available.
define <8 x i8> @udiv_v8i8(<8 x i8> %op1, <8 x i8> %op2) #0 {
; CHECK-LABEL: udiv_v8i8:
; CHECK: ptrue [[PG0:p[0-9]+]].s, vl8
; CHECK-NEXT: uunpklo [[OP2_LO:z[0-9]+]].h, z1.b
; CHECK-NEXT: uunpklo [[OP1_LO:z[0-9]+]].h, z0.b
; CHECK-NEXT: uunpklo [[OP2_LO_LO:z[0-9]+]].s, z1.h
; CHECK-NEXT: uunpklo [[OP1_LO_LO:z[0-9]+]].s, z0.h
; CHECK-NEXT: udiv [[DIV:z[0-9]+]].s, [[PG0]]/m, [[OP1_LO_LO]].s, [[OP2_LO_LO]].s
; CHECK-NEXT: uzp1 [[RES:z[0-9]+]].h, [[DIV]].h, [[DIV]].h
; CHECK-NEXT: umov [[SCALAR0:w[0-9]+]], [[VEC:v[0-9]+]].h[0]
; CHECK-NEXT: umov [[SCALAR1:w[0-9]+]], [[VEC]].h[1]
; CHECK-NEXT: fmov s0, [[SCALAR0]]
; CHECK-NEXT: umov [[SCALAR2:w[0-9]+]], [[VEC]].h[2]
; CHECK-NEXT: mov [[FINAL:v[0-9]+]].b[1], [[SCALAR1]]
; CHECK-NEXT: mov [[FINAL]].b[2], [[SCALAR2]]
; CHECK-NEXT: umov [[SCALAR3:w[0-9]+]], [[VEC]].h[3]
; CHECK-NEXT: mov [[FINAL]].b[3], [[SCALAR3]]
; CHECK-NEXT: umov [[SCALAR4:w[0-9]+]], [[VEC]].h[4]
; CHECK-NEXT: mov [[FINAL]].b[4], [[SCALAR4]]
; CHECK-NEXT: umov [[SCALAR5:w[0-9]+]], [[VEC]].h[5]
; CHECK-NEXT: mov [[FINAL]].b[5], [[SCALAR5]]
; CHECK-NEXT: umov [[SCALAR6:w[0-9]+]], [[VEC]].h[6]
; CHECK-NEXT: mov [[FINAL]].b[6], [[SCALAR6]]
; CHECK-NEXT: umov [[SCALAR7:w[0-9]+]], [[VEC]].h[7]
; CHECK-NEXT: mov [[FINAL]].b[7], [[SCALAR7]]
; CHECK: ret
  %res = udiv <8 x i8> %op1, %op2
  ret <8 x i8> %res
}

define <16 x i8> @udiv_v16i8(<16 x i8> %op1, <16 x i8> %op2) #0 {
; CHECK-LABEL: udiv_v16i8:

; HALF VECTOR:
; VBITS_EQ_256: ptrue [[PG:p[0-9]+]].s, vl8
; VBITS_EQ_256-NEXT: uunpklo [[OP2_LO:z[0-9]+]].h, z1.b
; VBITS_EQ_256-NEXT: uunpklo [[OP1_LO:z[0-9]+]].h, z0.b
; VBITS_EQ_256-NEXT: uunpkhi [[OP2_HI:z[0-9]+]].s, z1.h
; VBITS_EQ_256-NEXT: uunpkhi [[OP1_HI:z[0-9]+]].s, z0.h
; VBITS_EQ_256-NEXT: uunpklo [[OP2_LO_LO:z[0-9]+]].s, z1.h
; VBITS_EQ_256-NEXT: uunpklo [[OP1_LO_LO:z[0-9]+]].s, z0.h
; VBITS_EQ_256-NEXT: udivr [[DIV1:z[0-9]+]].s, [[PG]]/m, [[OP2_HI]].s, [[OP1_HI]].s
; VBITS_EQ_256-NEXT: udiv [[DIV2:z[0-9]+]].s, [[PG]]/m, [[OP1_LO]].s, [[OP2_LO]].s
; VBITS_EQ_256-NEXT: uzp1 [[RES1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_256-NEXT: uzp1 [[RES2:z[0-9]+]].b, [[RES1]].b, [[RES1]].b

; QUARTER VECTOR OR SMALLER:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: uunpklo [[OP2_LO:z[0-9]+]].h, z1.b
; VBITS_GE_512-NEXT: uunpklo [[OP1_LO:z[0-9]+]].h, z0.b
; VBITS_GE_512-NEXT: uunpklo [[OP2_LO_LO:z[0-9]+]].s, z1.h
; VBITS_GE_512-NEXT: uunpklo [[OP1_LO_LO:z[0-9]+]].s, z0.h
; VBITS_GE_512-NEXT: udiv [[DIV:z[0-9]+]].s, [[PG]]/m, [[OP1_LO_LO]].s, [[OP2_LO_LO]].s
; VBITS_GE_512-NEXT: uzp1 [[RES1:z[0-9]+]].h, [[DIV]].h, [[DIV]].h
; VBITS_GE_512-NEXT: uzp1 [[RES2:z[0-9]+]].b, [[RES1]].b, [[RES1]].b
; CHECK: ret
  %res = udiv <16 x i8> %op1, %op2
  ret <16 x i8> %res
}

define void @udiv_v32i8(<32 x i8>* %a, <32 x i8>* %b) #0 {
; CHECK-LABEL: udiv_v32i8:

; FULL VECTOR:
; VBITS_EQ_256: ptrue [[PG1:p[0-9]+]].b, vl32
; VBITS_EQ_256-NEXT: ptrue [[PG2:p[0-9]+]].s, vl8
; VBITS_EQ_256-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_EQ_256-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_EQ_256-NEXT: uunpkhi [[OP2_HI:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_256-NEXT: uunpkhi [[OP1_HI:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_256-NEXT: uunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_256-NEXT: uunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_256-NEXT: uunpkhi [[OP2_HI_HI:z[0-9]+]].s, [[OP2_HI]].h
; VBITS_EQ_256-NEXT: uunpkhi [[OP1_HI_HI:z[0-9]+]].s, [[OP1_HI]].h
; VBITS_EQ_256-NEXT: uunpklo [[OP2_HI_LO:z[0-9]+]].s, [[OP2_HI]].h
; VBITS_EQ_256-NEXT: uunpklo [[OP1_HI_LO:z[0-9]+]].s, [[OP1_HI]].h
; VBITS_EQ_256-NEXT: udivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI_HI]].s, [[OP1_HI_HI]].s
; VBITS_EQ_256-NEXT: uunpkhi [[OP2_LO_HI:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_256-NEXT: udivr [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI_LO]].s, [[OP1_HI_LO]].s
; VBITS_EQ_256-NEXT: uunpkhi [[OP1_LO_HI:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_256-NEXT: uunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_256-NEXT: uunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_256-NEXT: udiv [[DIV3:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_HI]].s, [[OP2_LO_HI]].s
; VBITS_EQ_256-NEXT: udiv [[DIV4:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_LO]].s, [[OP2_LO_LO]].s
; VBITS_EQ_256-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_256-NEXT: uzp1 [[UZP2:z[0-9]+]].h, [[DIV4]].h, [[DIV3]].h
; VBITS_EQ_256-NEXT: uzp1 [[UZP3:z[0-9]+]].b, [[UZP2]].b, [[UZP1]].b
; VBITS_EQ_256-NEXT: st1b { [[UZP3:z[0-9]+]].b }, [[PG1]], [x0]

; HALF VECTOR:
; VBITS_EQ_512: ptrue [[PG1:p[0-9]+]].b, vl32
; VBITS_EQ_512-NEXT: ptrue [[PG2:p[0-9]+]].s, vl16
; VBITS_EQ_512-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_EQ_512-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_EQ_512-NEXT: uunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_512-NEXT: uunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_512-NEXT: uunpkhi [[OP2_LO_HI:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_512-NEXT: uunpkhi [[OP1_LO_HI:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_512-NEXT: uunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_512-NEXT: uunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_512-NEXT: udivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_LO_HI]].s, [[OP1_LO_HI]].s
; VBITS_EQ_512-NEXT: udiv [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_LO]].s, [[OP2_LO_LO]].s
; VBITS_EQ_512-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_512-NEXT: uzp1 [[UZP2:z[0-9]+]].b, [[UZP1]].b, [[UZP1]].b
; VBITS_EQ_512-NEXT: st1b { [[UZP2:z[0-9]+]].b }, [[PG1]], [x0]

; QUARTER VECTOR OR SMALLER:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].b, vl32
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_GE_1024-NEXT: uunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_GE_1024-NEXT: uunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_GE_1024-NEXT: uunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_GE_1024-NEXT: uunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_GE_1024-NEXT: udiv [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_LO]].s, [[OP2_LO_LO]].s
; VBITS_GE_1024-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV1]].h, [[DIV1]].h
; VBITS_GE_1024-NEXT: uzp1 [[UZP2:z[0-9]+]].b, [[UZP1]].b, [[UZP1]].b
; VBITS_GE_1024-NEXT: st1b { [[UZP2:z[0-9]+]].b }, [[PG1]], [x0]
; CHECK: ret
  %op1 = load <32 x i8>, <32 x i8>* %a
  %op2 = load <32 x i8>, <32 x i8>* %b
  %res = udiv <32 x i8> %op1, %op2
  store <32 x i8> %res, <32 x i8>* %a
  ret void
}

define void @udiv_v64i8(<64 x i8>* %a, <64 x i8>* %b) #0 {
; CHECK-LABEL: udiv_v64i8:

; FULL VECTOR:
; VBITS_EQ_512: ptrue [[PG1:p[0-9]+]].b, vl64
; VBITS_EQ_512-NEXT: ptrue [[PG2:p[0-9]+]].s, vl16
; VBITS_EQ_512-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_EQ_512-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_EQ_512-NEXT: uunpkhi [[OP2_HI:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_512-NEXT: uunpkhi [[OP1_HI:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_512-NEXT: uunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_512-NEXT: uunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_512-NEXT: uunpkhi [[OP2_HI_HI:z[0-9]+]].s, [[OP2_HI]].h
; VBITS_EQ_512-NEXT: uunpkhi [[OP1_HI_HI:z[0-9]+]].s, [[OP1_HI]].h
; VBITS_EQ_512-NEXT: uunpklo [[OP2_HI_LO:z[0-9]+]].s, [[OP2_HI]].h
; VBITS_EQ_512-NEXT: uunpklo [[OP1_HI_LO:z[0-9]+]].s, [[OP1_HI]].h
; VBITS_EQ_512-NEXT: udivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI_HI]].s, [[OP1_HI_HI]].s
; VBITS_EQ_512-NEXT: uunpkhi [[OP2_LO_HI:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_512-NEXT: udivr [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI_LO]].s, [[OP1_HI_LO]].s
; VBITS_EQ_512-NEXT: uunpkhi [[OP1_LO_HI:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_512-NEXT: uunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_512-NEXT: uunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_512-NEXT: udiv [[DIV3:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_HI]].s, [[OP2_LO_HI]].s
; VBITS_EQ_512-NEXT: udiv [[DIV4:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_LO]].s, [[OP2_LO_LO]].s
; VBITS_EQ_512-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_512-NEXT: uzp1 [[UZP2:z[0-9]+]].h, [[DIV4]].h, [[DIV3]].h
; VBITS_EQ_512-NEXT: uzp1 [[UZP3:z[0-9]+]].b, [[UZP2]].b, [[UZP1]].b
; VBITS_EQ_512-NEXT: st1b { [[UZP3:z[0-9]+]].b }, [[PG1]], [x0]

; HALF VECTOR:
; VBITS_EQ_1024: ptrue [[PG1:p[0-9]+]].b, vl64
; VBITS_EQ_1024-NEXT: ptrue [[PG2:p[0-9]+]].s, vl32
; VBITS_EQ_1024-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_EQ_1024-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_EQ_1024-NEXT: uunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_1024-NEXT: uunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_1024-NEXT: uunpkhi [[OP2_LO_HI:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_1024-NEXT: uunpkhi [[OP1_LO_HI:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_1024-NEXT: uunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_1024-NEXT: uunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_1024-NEXT: udivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_LO_HI]].s, [[OP1_LO_HI]].s
; VBITS_EQ_1024-NEXT: udiv [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_LO]].s, [[OP2_LO_LO]].s
; VBITS_EQ_1024-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_1024-NEXT: uzp1 [[UZP2:z[0-9]+]].b, [[UZP1]].b, [[UZP1]].b
; VBITS_EQ_1024-NEXT: st1b { [[UZP2:z[0-9]+]].b }, [[PG1]], [x0]

; QUARTER VECTOR OR SMALLER:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].b, vl64
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_GE_2048-NEXT: uunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_GE_2048-NEXT: uunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_GE_2048-NEXT: uunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_GE_2048-NEXT: uunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_GE_2048-NEXT: udiv [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_LO]].s, [[OP2_LO_LO]].s
; VBITS_GE_2048-NEXT: uzp1 [[RES1:z[0-9]+]].h, [[DIV1]].h, [[DIV1]].h
; VBITS_GE_2048-NEXT: uzp1 [[RES2:z[0-9]+]].b, [[RES1]].b, [[RES1]].b
; VBITS_GE_2048-NEXT: st1b { [[RES2]].b }, [[PG1]], [x0]
; CHECK: ret
  %op1 = load <64 x i8>, <64 x i8>* %a
  %op2 = load <64 x i8>, <64 x i8>* %b
  %res = udiv <64 x i8> %op1, %op2
  store <64 x i8> %res, <64 x i8>* %a
  ret void
}

define void @udiv_v128i8(<128 x i8>* %a, <128 x i8>* %b) #0 {
; CHECK-LABEL: udiv_v128i8:

; FULL VECTOR:
; VBITS_EQ_1024: ptrue [[PG1:p[0-9]+]].b, vl128
; VBITS_EQ_1024-NEXT: ptrue [[PG2:p[0-9]+]].s, vl32
; VBITS_EQ_1024-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_EQ_1024-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_EQ_1024-NEXT: uunpkhi [[OP2_HI:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_1024-NEXT: uunpkhi [[OP1_HI:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_1024-NEXT: uunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_1024-NEXT: uunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_1024-NEXT: uunpkhi [[OP2_HI_HI:z[0-9]+]].s, [[OP2_HI]].h
; VBITS_EQ_1024-NEXT: uunpkhi [[OP1_HI_HI:z[0-9]+]].s, [[OP1_HI]].h
; VBITS_EQ_1024-NEXT: uunpklo [[OP2_HI_LO:z[0-9]+]].s, [[OP2_HI]].h
; VBITS_EQ_1024-NEXT: uunpklo [[OP1_HI_LO:z[0-9]+]].s, [[OP1_HI]].h
; VBITS_EQ_1024-NEXT: udivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI_HI]].s, [[OP1_HI_HI]].s
; VBITS_EQ_1024-NEXT: uunpkhi [[OP2_LO_HI:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_1024-NEXT: udivr [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI_LO]].s, [[OP1_HI_LO]].s
; VBITS_EQ_1024-NEXT: uunpkhi [[OP1_LO_HI:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_1024-NEXT: uunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_1024-NEXT: uunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_1024-NEXT: udiv [[DIV3:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_HI]].s, [[OP2_LO_HI]].s
; VBITS_EQ_1024-NEXT: udiv [[DIV4:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_LO]].s, [[OP2_LO_LO]].s
; VBITS_EQ_1024-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_1024-NEXT: uzp1 [[UZP2:z[0-9]+]].h, [[DIV4]].h, [[DIV3]].h
; VBITS_EQ_1024-NEXT: uzp1 [[UZP3:z[0-9]+]].b, [[UZP2]].b, [[UZP1]].b
; VBITS_EQ_1024-NEXT: st1b { [[UZP3:z[0-9]+]].b }, [[PG1]], [x0]

; HALF VECTOR:
; VBITS_EQ_2048: ptrue [[PG1:p[0-9]+]].b, vl128
; VBITS_EQ_2048-NEXT: ptrue [[PG2:p[0-9]+]].s, vl64
; VBITS_EQ_2048-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_EQ_2048-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_EQ_2048-NEXT: uunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_2048-NEXT: uunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_2048-NEXT: uunpkhi [[OP2_LO_HI:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_2048-NEXT: uunpkhi [[OP1_LO_HI:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_2048-NEXT: uunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_2048-NEXT: uunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_2048-NEXT: udivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_LO_HI]].s, [[OP1_LO_HI]].s
; VBITS_EQ_2048-NEXT: udiv [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_LO]].s, [[OP2_LO_LO]].s
; VBITS_EQ_2048-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_2048-NEXT: uzp1 [[UZP2:z[0-9]+]].b, [[UZP1]].b, [[UZP1]].b
; VBITS_EQ_2048-NEXT: st1b { [[UZP2:z[0-9]+]].b }, [[PG1]], [x0]
; CHECK: ret
  %op1 = load <128 x i8>, <128 x i8>* %a
  %op2 = load <128 x i8>, <128 x i8>* %b
  %res = udiv <128 x i8> %op1, %op2
  store <128 x i8> %res, <128 x i8>* %a
  ret void
}

define void @udiv_v256i8(<256 x i8>* %a, <256 x i8>* %b) #0 {
; CHECK-LABEL: udiv_v256i8:
; VBITS_EQ_2048: ptrue [[PG1:p[0-9]+]].b, vl256
; VBITS_EQ_2048-NEXT: ptrue [[PG2:p[0-9]+]].s, vl64
; VBITS_EQ_2048-NEXT: ld1b { [[OP1:z[0-9]+]].b }, [[PG1]]/z, [x0]
; VBITS_EQ_2048-NEXT: ld1b { [[OP2:z[0-9]+]].b }, [[PG1]]/z, [x1]
; VBITS_EQ_2048-NEXT: uunpkhi [[OP2_HI:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_2048-NEXT: uunpkhi [[OP1_HI:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_2048-NEXT: uunpklo [[OP2_LO:z[0-9]+]].h, [[OP2]].b
; VBITS_EQ_2048-NEXT: uunpklo [[OP1_LO:z[0-9]+]].h, [[OP1]].b
; VBITS_EQ_2048-NEXT: uunpkhi [[OP2_HI_HI:z[0-9]+]].s, [[OP2_HI]].h
; VBITS_EQ_2048-NEXT: uunpkhi [[OP1_HI_HI:z[0-9]+]].s, [[OP1_HI]].h
; VBITS_EQ_2048-NEXT: uunpklo [[OP2_HI_LO:z[0-9]+]].s, [[OP2_HI]].h
; VBITS_EQ_2048-NEXT: uunpklo [[OP1_HI_LO:z[0-9]+]].s, [[OP1_HI]].h
; VBITS_EQ_2048-NEXT: udivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI_HI]].s, [[OP1_HI_HI]].s
; VBITS_EQ_2048-NEXT: uunpkhi [[OP2_LO_HI:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_2048-NEXT: udivr [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI_LO]].s, [[OP1_HI_LO]].s
; VBITS_EQ_2048-NEXT: uunpkhi [[OP1_LO_HI:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_2048-NEXT: uunpklo [[OP2_LO_LO:z[0-9]+]].s, [[OP2_LO]].h
; VBITS_EQ_2048-NEXT: uunpklo [[OP1_LO_LO:z[0-9]+]].s, [[OP1_LO]].h
; VBITS_EQ_2048-NEXT: udiv [[DIV3:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_HI]].s, [[OP2_LO_HI]].s
; VBITS_EQ_2048-NEXT: udiv [[DIV4:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO_LO]].s, [[OP2_LO_LO]].s
; VBITS_EQ_2048-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_2048-NEXT: uzp1 [[UZP2:z[0-9]+]].h, [[DIV4]].h, [[DIV3]].h
; VBITS_EQ_2048-NEXT: uzp1 [[UZP3:z[0-9]+]].b, [[UZP2]].b, [[UZP1]].b
; VBITS_EQ_2048-NEXT: st1b { [[UZP3:z[0-9]+]].b }, [[PG1]], [x0]
; CHECK: ret
  %op1 = load <256 x i8>, <256 x i8>* %a
  %op2 = load <256 x i8>, <256 x i8>* %b
  %res = udiv <256 x i8> %op1, %op2
  store <256 x i8> %res, <256 x i8>* %a
  ret void
}

; Vector vXi16 udiv are not legal for NEON so use SVE when available.
define <4 x i16> @udiv_v4i16(<4 x i16> %op1, <4 x i16> %op2) #0 {
; CHECK-LABEL: udiv_v4i16:
; CHECK: ushll v1.4s, v1.4h, #0
; CHECK-NEXT: ushll v0.4s, v0.4h, #0
; CHECK-NEXT: ptrue [[PG1:p[0-9]+]].s, vl4
; CHECK-NEXT: udivr [[DIV1:z[0-9]+]].s, [[PG1]]/m, [[OP2:z[0-9]+]].s, [[OP1:z[0-9]+]].s
; CHECK-NEXT: mov w8, v1.s[1]
; CHECK-NEXT: mov w9, v1.s[2]
; CHECK-NEXT: mov v0.16b, v1.16b
; CHECK-NEXT: mov v0.h[1], w8
; CHECK-NEXT: mov w8, v1.s[3]
; CHECK-NEXT: mov v0.h[2], w9
; CHECK-NEXT: mov v0.h[3], w8
; CHECK: ret
  %res = udiv <4 x i16> %op1, %op2
  ret <4 x i16> %res
}

define <8 x i16> @udiv_v8i16(<8 x i16> %op1, <8 x i16> %op2) #0 {
; CHECK-LABEL: udiv_v8i16:
; CHECK: ptrue [[PG1:p[0-9]+]].s, vl8
; CHECK-NEXT: uunpklo [[OP2_LO:z[0-9]+]].s, [[OP2]].h
; CHECK-NEXT: uunpklo [[OP1_LO:z[0-9]+]].s, [[OP1]].h
; CHECK-NEXT: udiv [[DIV1:z[0-9]+]].s, [[PG1]]/m, [[OP1_LO]].s, [[OP2_LO]].s
; CHECK-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV1]].h, [[DIV1]].h
; CHECK: ret
  %res = udiv <8 x i16> %op1, %op2
  ret <8 x i16> %res
}

define void @udiv_v16i16(<16 x i16>* %a, <16 x i16>* %b) #0 {
; CHECK-LABEL: udiv_v16i16:

; FULL VECTOR:
; VBITS_EQ_256: ptrue [[PG1:p[0-9]+]].h, vl16
; VBITS_EQ_256-NEXT: ptrue [[PG2:p[0-9]+]].s, vl8
; VBITS_EQ_256-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_EQ_256-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; VBITS_EQ_256-NEXT: uunpkhi [[OP2_HI:z[0-9]+]].s, [[OP2]].h
; VBITS_EQ_256-NEXT: uunpkhi [[OP1_HI:z[0-9]+]].s, [[OP1]].h
; VBITS_EQ_256-NEXT: uunpklo [[OP2_LO:z[0-9]+]].s, [[OP2]].h
; VBITS_EQ_256-NEXT: uunpklo [[OP1_LO:z[0-9]+]].s, [[OP1]].h
; VBITS_EQ_256-NEXT: udivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI]].s, [[OP1_HI]].s
; VBITS_EQ_256-NEXT: udiv [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO]].s, [[OP2_LO]].s
; VBITS_EQ_256-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_256-NEXT: st1h { [[UZP1:z[0-9]+]].h }, [[PG1]], [x0]

; HALF VECTOR OR SMALLER:
; VBITS_GE_512: ptrue [[PG1:p[0-9]+]].h, vl16
; VBITS_GE_512-NEXT: ptrue [[PG2:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_512-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; VBITS_GE_512-NEXT: uunpklo [[OP2_LO:z[0-9]+]].s, [[OP2]].h
; VBITS_GE_512-NEXT: uunpklo [[OP1_LO:z[0-9]+]].s, [[OP1]].h
; VBITS_GE_512-NEXT: udiv [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO]].s, [[OP2_LO]].s
; VBITS_GE_512-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV1]].h, [[DIV1]].h
; VBITS_GE_512-NEXT: st1h { [[UZP1:z[0-9]+]].h }, [[PG1]], [x0]
; CHECK: ret
  %op1 = load <16 x i16>, <16 x i16>* %a
  %op2 = load <16 x i16>, <16 x i16>* %b
  %res = udiv <16 x i16> %op1, %op2
  store <16 x i16> %res, <16 x i16>* %a
  ret void
}

define void @udiv_v32i16(<32 x i16>* %a, <32 x i16>* %b) #0 {
; CHECK-LABEL: udiv_v32i16:

; FULL VECTOR:
; VBITS_EQ_512: ptrue [[PG1:p[0-9]+]].h, vl32
; VBITS_EQ_512-NEXT: ptrue [[PG2:p[0-9]+]].s, vl16
; VBITS_EQ_512-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_EQ_512-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; VBITS_EQ_512-NEXT: uunpkhi [[OP2_HI:z[0-9]+]].s, [[OP2]].h
; VBITS_EQ_512-NEXT: uunpkhi [[OP1_HI:z[0-9]+]].s, [[OP1]].h
; VBITS_EQ_512-NEXT: uunpklo [[OP2_LO:z[0-9]+]].s, [[OP2]].h
; VBITS_EQ_512-NEXT: uunpklo [[OP1_LO:z[0-9]+]].s, [[OP1]].h
; VBITS_EQ_512-NEXT: udivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI]].s, [[OP1_HI]].s
; VBITS_EQ_512-NEXT: udiv [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO]].s, [[OP2_LO]].s
; VBITS_EQ_512-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_512-NEXT: st1h { [[UZP1:z[0-9]+]].h }, [[PG1]], [x0]

; HALF VECTOR OR SMALLER:
; VBITS_GE_1024: ptrue [[PG1:p[0-9]+]].h, vl32
; VBITS_GE_1024-NEXT: ptrue [[PG2:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_1024-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; VBITS_GE_1024-NEXT: uunpklo [[OP2_LO:z[0-9]+]].s, [[OP2]].h
; VBITS_GE_1024-NEXT: uunpklo [[OP1_LO:z[0-9]+]].s, [[OP1]].h
; VBITS_GE_1024-NEXT: udiv [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO]].s, [[OP2_LO]].s
; VBITS_GE_1024-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV1]].h, [[DIV1]].h
; VBITS_GE_1024-NEXT: st1h { [[UZP1:z[0-9]+]].h }, [[PG1]], [x0]
; CHECK: ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %op2 = load <32 x i16>, <32 x i16>* %b
  %res = udiv <32 x i16> %op1, %op2
  store <32 x i16> %res, <32 x i16>* %a
  ret void
}

define void @udiv_v64i16(<64 x i16>* %a, <64 x i16>* %b) #0 {
; CHECK-LABEL: udiv_v64i16:

; FULL VECTOR:
; VBITS_EQ_1024: ptrue [[PG1:p[0-9]+]].h, vl64
; VBITS_EQ_1024-NEXT: ptrue [[PG2:p[0-9]+]].s, vl32
; VBITS_EQ_1024-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_EQ_1024-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; VBITS_EQ_1024-NEXT: uunpkhi [[OP2_HI:z[0-9]+]].s, [[OP2]].h
; VBITS_EQ_1024-NEXT: uunpkhi [[OP1_HI:z[0-9]+]].s, [[OP1]].h
; VBITS_EQ_1024-NEXT: uunpklo [[OP2_LO:z[0-9]+]].s, [[OP2]].h
; VBITS_EQ_1024-NEXT: uunpklo [[OP1_LO:z[0-9]+]].s, [[OP1]].h
; VBITS_EQ_1024-NEXT: udivr [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP2_HI]].s, [[OP1_HI]].s
; VBITS_EQ_1024-NEXT: udiv [[DIV2:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO]].s, [[OP2_LO]].s
; VBITS_EQ_1024-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV2]].h, [[DIV1]].h
; VBITS_EQ_1024-NEXT: st1h { [[UZP1:z[0-9]+]].h }, [[PG1]], [x0]

; HALF VECTOR OR SMALLER:
; VBITS_GE_2048: ptrue [[PG1:p[0-9]+]].h, vl64
; VBITS_GE_2048-NEXT: ptrue [[PG2:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG1]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG1]]/z, [x1]
; VBITS_GE_2048-NEXT: uunpklo [[OP2_LO:z[0-9]+]].s, [[OP2]].h
; VBITS_GE_2048-NEXT: uunpklo [[OP1_LO:z[0-9]+]].s, [[OP1]].h
; VBITS_GE_2048-NEXT: udiv [[DIV1:z[0-9]+]].s, [[PG2]]/m, [[OP1_LO]].s, [[OP2_LO]].s
; VBITS_GE_2048-NEXT: uzp1 [[UZP1:z[0-9]+]].h, [[DIV1]].h, [[DIV1]].h
; VBITS_GE_2048-NEXT: st1h { [[UZP1:z[0-9]+]].h }, [[PG1]], [x0]
; CHECK: ret
  %op1 = load <64 x i16>, <64 x i16>* %a
  %op2 = load <64 x i16>, <64 x i16>* %b
  %res = udiv <64 x i16> %op1, %op2
  store <64 x i16> %res, <64 x i16>* %a
  ret void
}

define void @udiv_v128i16(<128 x i16>* %a, <128 x i16>* %b) #0 {
; CHECK-LABEL: udiv_v128i16:
; VBITS_EQ_2048: ptrue [[PG:p[0-9]+]].h, vl128
; VBITS_EQ_2048-NEXT: ptrue [[PG1:p[0-9]+]].s, vl64
; VBITS_EQ_2048-NEXT: ld1h { [[OP1:z[0-9]+]].h }, [[PG]]/z, [x0]
; VBITS_EQ_2048-NEXT: ld1h { [[OP2:z[0-9]+]].h }, [[PG]]/z, [x1]
; VBITS_EQ_2048-NEXT: uunpkhi [[OP2_HI:z[0-9]+]].s, [[OP2]].h
; VBITS_EQ_2048-NEXT: uunpkhi [[OP1_HI:z[0-9]+]].s, [[OP1]].h
; VBITS_EQ_2048-NEXT: uunpklo [[OP2_LO:z[0-9]+]].s, [[OP2]].h
; VBITS_EQ_2048-NEXT: uunpklo [[OP1_LO:z[0-9]+]].s, [[OP1]].h
; VBITS_EQ_2048-NEXT: udivr   [[RES_HI:z[0-9]+]].s, [[PG1]]/m, [[OP2_HI]].s, [[OP1_HI]].s
; VBITS_EQ_2048-NEXT: udiv    [[RES_LO:z[0-9]+]].s, [[PG1]]/m, [[OP1_LO]].s, [[OP2_LO]].s
; VBITS_EQ_2048-NEXT: uzp1 [[RES:z[0-9]+]].h, [[RES_LO]].h, [[RES_HI]].h
; VBITS_EQ_2048-NEXT: st1h { [[RES]].h }, [[PG]], [x0]
; VBITS_EQ_2048-NEXT: ret
  %op1 = load <128 x i16>, <128 x i16>* %a
  %op2 = load <128 x i16>, <128 x i16>* %b
  %res = udiv <128 x i16> %op1, %op2
  store <128 x i16> %res, <128 x i16>* %a
  ret void
}

; Vector v2i32 udiv are not legal for NEON so use SVE when available.
define <2 x i32> @udiv_v2i32(<2 x i32> %op1, <2 x i32> %op2) #0 {
; CHECK-LABEL: udiv_v2i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl2
; CHECK: udiv z0.s, [[PG]]/m, z0.s, z1.s
; CHECK: ret
  %res = udiv <2 x i32> %op1, %op2
  ret <2 x i32> %res
}

; Vector v4i32 udiv are not legal for NEON so use SVE when available.
define <4 x i32> @udiv_v4i32(<4 x i32> %op1, <4 x i32> %op2) #0 {
; CHECK-LABEL: udiv_v4i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl4
; CHECK: udiv z0.s, [[PG]]/m, z0.s, z1.s
; CHECK: ret
  %res = udiv <4 x i32> %op1, %op2
  ret <4 x i32> %res
}

define void @udiv_v8i32(<8 x i32>* %a, <8 x i32>* %b) #0 {
; CHECK-LABEL: udiv_v8i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-NEXT: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-NEXT: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; CHECK-NEXT: udiv [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %op2 = load <8 x i32>, <8 x i32>* %b
  %res = udiv <8 x i32> %op1, %op2
  store <8 x i32> %res, <8 x i32>* %a
  ret void
}

define void @udiv_v16i32(<16 x i32>* %a, <16 x i32>* %b) #0 {
; CHECK-LABEL: udiv_v16i32:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].s, vl16
; VBITS_GE_512-NEXT: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: udiv [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_GE_512-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %op2 = load <16 x i32>, <16 x i32>* %b
  %res = udiv <16 x i32> %op1, %op2
  store <16 x i32> %res, <16 x i32>* %a
  ret void
}

define void @udiv_v32i32(<32 x i32>* %a, <32 x i32>* %b) #0 {
; CHECK-LABEL: udiv_v32i32:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].s, vl32
; VBITS_GE_1024-NEXT: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: udiv [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_GE_1024-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %op2 = load <32 x i32>, <32 x i32>* %b
  %res = udiv <32 x i32> %op1, %op2
  store <32 x i32> %res, <32 x i32>* %a
  ret void
}

define void @udiv_v64i32(<64 x i32>* %a, <64 x i32>* %b) #0 {
; CHECK-LABEL: udiv_v64i32:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].s, vl64
; VBITS_GE_2048-NEXT: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1w { [[OP2:z[0-9]+]].s }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: udiv [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; VBITS_GE_2048-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <64 x i32>, <64 x i32>* %a
  %op2 = load <64 x i32>, <64 x i32>* %b
  %res = udiv <64 x i32> %op1, %op2
  store <64 x i32> %res, <64 x i32>* %a
  ret void
}

; Vector i64 udiv are not legal for NEON so use SVE when available.
define <1 x i64> @udiv_v1i64(<1 x i64> %op1, <1 x i64> %op2) #0 {
; CHECK-LABEL: udiv_v1i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl1
; CHECK: udiv z0.d, [[PG]]/m, z0.d, z1.d
; CHECK: ret
  %res = udiv <1 x i64> %op1, %op2
  ret <1 x i64> %res
}

; Vector i64 udiv are not legal for NEON so use SVE when available.
define <2 x i64> @udiv_v2i64(<2 x i64> %op1, <2 x i64> %op2) #0 {
; CHECK-LABEL: udiv_v2i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl2
; CHECK: udiv z0.d, [[PG]]/m, z0.d, z1.d
; CHECK: ret
  %res = udiv <2 x i64> %op1, %op2
  ret <2 x i64> %res
}

define void @udiv_v4i64(<4 x i64>* %a, <4 x i64>* %b) #0 {
; CHECK-LABEL: udiv_v4i64:
; CHECK: ptrue [[PG:p[0-9]+]].d, vl4
; CHECK-NEXT: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; CHECK-NEXT: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; CHECK-NEXT: udiv [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; CHECK-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %op2 = load <4 x i64>, <4 x i64>* %b
  %res = udiv <4 x i64> %op1, %op2
  store <4 x i64> %res, <4 x i64>* %a
  ret void
}

define void @udiv_v8i64(<8 x i64>* %a, <8 x i64>* %b) #0 {
; CHECK-LABEL: udiv_v8i64:
; VBITS_GE_512: ptrue [[PG:p[0-9]+]].d, vl8
; VBITS_GE_512-NEXT: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_512-NEXT: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_512-NEXT: udiv [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_512-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_512-NEXT: ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %op2 = load <8 x i64>, <8 x i64>* %b
  %res = udiv <8 x i64> %op1, %op2
  store <8 x i64> %res, <8 x i64>* %a
  ret void
}

define void @udiv_v16i64(<16 x i64>* %a, <16 x i64>* %b) #0 {
; CHECK-LABEL: udiv_v16i64:
; VBITS_GE_1024: ptrue [[PG:p[0-9]+]].d, vl16
; VBITS_GE_1024-NEXT: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_1024-NEXT: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_1024-NEXT: udiv [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_1024-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_1024-NEXT: ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %op2 = load <16 x i64>, <16 x i64>* %b
  %res = udiv <16 x i64> %op1, %op2
  store <16 x i64> %res, <16 x i64>* %a
  ret void
}

define void @udiv_v32i64(<32 x i64>* %a, <32 x i64>* %b) #0 {
; CHECK-LABEL: udiv_v32i64:
; VBITS_GE_2048: ptrue [[PG:p[0-9]+]].d, vl32
; VBITS_GE_2048-NEXT: ld1d { [[OP1:z[0-9]+]].d }, [[PG]]/z, [x0]
; VBITS_GE_2048-NEXT: ld1d { [[OP2:z[0-9]+]].d }, [[PG]]/z, [x1]
; VBITS_GE_2048-NEXT: udiv [[RES:z[0-9]+]].d, [[PG]]/m, [[OP1]].d, [[OP2]].d
; VBITS_GE_2048-NEXT: st1d { [[RES]].d }, [[PG]], [x0]
; VBITS_GE_2048-NEXT: ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %op2 = load <32 x i64>, <32 x i64>* %b
  %res = udiv <32 x i64> %op1, %op2
  store <32 x i64> %res, <32 x i64>* %a
  ret void
}

; This used to crash because isUnaryPredicate and BuildUDIV don't know how
; a SPLAT_VECTOR of fixed vector type should be handled.
define void @udiv_constantsplat_v8i32(<8 x i32>* %a) #1 {
; CHECK-LABEL: udiv_constantsplat_v8i32:
; CHECK: ptrue [[PG:p[0-9]+]].s, vl8
; CHECK-NEXT: mov [[OP2:z[0-9]+]].s, #95
; CHECK-NEXT: ld1w { [[OP1:z[0-9]+]].s }, [[PG]]/z, [x0]
; CHECK-NEXT: udiv [[RES:z[0-9]+]].s, [[PG]]/m, [[OP1]].s, [[OP2]].s
; CHECK-NEXT: st1w { [[RES]].s }, [[PG]], [x0]
; CHECK-NEXT: ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %res = udiv <8 x i32> %op1, <i32 95, i32 95, i32 95, i32 95, i32 95, i32 95, i32 95, i32 95>
  store <8 x i32> %res, <8 x i32>* %a
  ret void
}

attributes #0 = { "target-features"="+sve" }
attributes #1 = { "target-features"="+sve" minsize }
