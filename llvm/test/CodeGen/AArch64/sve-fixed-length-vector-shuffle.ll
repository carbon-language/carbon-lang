; RUN: llc -aarch64-sve-vector-bits-min=128  < %s | FileCheck %s -check-prefix=NO_SVE
; RUN: llc -aarch64-sve-vector-bits-min=256  < %s | FileCheck %s -check-prefixes=CHECK,VBITS_EQ_256
; RUN: llc -aarch64-sve-vector-bits-min=384  < %s | FileCheck %s -check-prefixes=CHECK
; RUN: llc -aarch64-sve-vector-bits-min=512  < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=640  < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=768  < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=896  < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512
; RUN: llc -aarch64-sve-vector-bits-min=1024 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1152 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1280 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1408 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1536 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1664 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1792 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=1920 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024
; RUN: llc -aarch64-sve-vector-bits-min=2048 < %s | FileCheck %s -check-prefixes=CHECK,VBITS_GE_512,VBITS_GE_1024,VBITS_GE_2048

target triple = "aarch64-unknown-linux-gnu"

; Don't use SVE when its registers are no bigger than NEON.
; NO_SVE-NOT: ptrue

; Don't use SVE for 64-bit vectors
define <8 x i8> @shuffle_ext_byone_v8i8(<8 x i8> %op1, <8 x i8> %op2) #0 {
; CHECK-LABEL: shuffle_ext_byone_v8i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ext v0.8b, v0.8b, v1.8b, #7
; CHECK-NEXT:    ret
  %ret = shufflevector <8 x i8> %op1, <8 x i8> %op2, <8 x i32> <i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14>
  ret <8 x i8> %ret
}

; Don't use SVE for 128-bit vectors
define <16 x i8> @shuffle_ext_byone_v16i8(<16 x i8> %op1, <16 x i8> %op2) {
; CHECK-LABEL: shuffle_ext_byone_v16i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ext v0.16b, v0.16b, v1.16b, #15
; CHECK-NEXT:    ret
  %ret = shufflevector <16 x i8> %op1, <16 x i8> %op2, <16 x i32> <i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22,
                                                                   i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30>
  ret <16 x i8> %ret
}

define void @shuffle_ext_byone_v32i8(<32 x i8>* %a, <32 x i8>* %b) #0 {
; CHECK-LABEL: shuffle_ext_byone_v32i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.b, vl32
; CHECK-NEXT:    ld1b { z0.b }, p0/z, [x0]
; CHECK-NEXT:    ld1b { z1.b }, p0/z, [x1]
; CHECK-NEXT:    mov z0.b, z0.b[31]
; CHECK-NEXT:    fmov w8, s0
; CHECK-NEXT:    insr z1.b, w8
; CHECK-NEXT:    st1b { z1.b }, p0, [x0]
; CHECK-NEXT:    ret
  %op1 = load <32 x i8>, <32 x i8>* %a
  %op2 = load <32 x i8>, <32 x i8>* %b
  %ret = shufflevector <32 x i8> %op1, <32 x i8> %op2, <32 x i32> <i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38,
                                                                   i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46,
                                                                   i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54,
                                                                   i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62>
  store <32 x i8> %ret, <32 x i8>* %a
  ret void
}

define void @shuffle_ext_byone_v64i8(<64 x i8>* %a, <64 x i8>* %b) #0 {
; Ensure sensible type legalisation.
; VBITS_EQ_256-LABEL: shuffle_ext_byone_v64i8:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov w8, #32
; VBITS_EQ_256-NEXT:    ptrue p0.b, vl32
; VBITS_EQ_256-NEXT:    ld1b { z0.b }, p0/z, [x0, x8]
; VBITS_EQ_256-NEXT:    ld1b { z1.b }, p0/z, [x1, x8]
; VBITS_EQ_256-NEXT:    ld1b { z2.b }, p0/z, [x1]
; VBITS_EQ_256-NEXT:    mov z0.b, z0.b[31]
; VBITS_EQ_256-NEXT:    mov z3.b, z2.b[31]
; VBITS_EQ_256-NEXT:    fmov w9, s0
; VBITS_EQ_256-NEXT:    fmov w10, s3
; VBITS_EQ_256-NEXT:    insr z2.b, w9
; VBITS_EQ_256-NEXT:    insr z1.b, w10
; VBITS_EQ_256-NEXT:    st1b { z2.b }, p0, [x0]
; VBITS_EQ_256-NEXT:    st1b { z1.b }, p0, [x0, x8]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: shuffle_ext_byone_v64i8:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.b, vl64
; VBITS_GE_512-NEXT:    ld1b { z0.b }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ld1b { z1.b }, p0/z, [x1]
; VBITS_GE_512-NEXT:    mov z0.b, z0.b[63]
; VBITS_GE_512-NEXT:    fmov w8, s0
; VBITS_GE_512-NEXT:    insr z1.b, w8
; VBITS_GE_512-NEXT:    st1b { z1.b }, p0, [x0]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <64 x i8>, <64 x i8>* %a
  %op2 = load <64 x i8>, <64 x i8>* %b
  %ret = shufflevector <64 x i8> %op1, <64 x i8> %op2, <64 x i32> <i32 63, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70,
                                                                   i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78,
                                                                   i32 79, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86,
                                                                   i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94,
                                                                   i32 95, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102,
                                                                   i32 103, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110,
                                                                   i32 111, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118,
                                                                   i32 119, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126>
  store <64 x i8> %ret, <64 x i8>* %a
  ret void
}

define void @shuffle_ext_byone_v128i8(<128 x i8>* %a, <128 x i8>* %b) #0 {
; VBITS_GE_1024-LABEL: shuffle_ext_byone_v128i8:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.b, vl128
; VBITS_GE_1024-NEXT:    mov w8, #127
; VBITS_GE_1024-NEXT:    ld1b { z0.b }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ld1b { z1.b }, p0/z, [x1]
; VBITS_GE_1024-NEXT:    whilels p1.b, xzr, x8
; VBITS_GE_1024-NEXT:    lastb w8, p1, z0.b
; VBITS_GE_1024-NEXT:    insr z1.b, w8
; VBITS_GE_1024-NEXT:    st1b { z1.b }, p0, [x0]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <128 x i8>, <128 x i8>* %a
  %op2 = load <128 x i8>, <128 x i8>* %b
  %ret = shufflevector <128 x i8> %op1, <128 x i8> %op2, <128 x i32> <i32 127,  i32 128,  i32 129,  i32 130,  i32 131,  i32 132,  i32 133,  i32 134,
                                                                      i32 135,  i32 136,  i32 137,  i32 138,  i32 139,  i32 140,  i32 141,  i32 142,
                                                                      i32 143,  i32 144,  i32 145,  i32 146,  i32 147,  i32 148,  i32 149,  i32 150,
                                                                      i32 151,  i32 152,  i32 153,  i32 154,  i32 155,  i32 156,  i32 157,  i32 158,
                                                                      i32 159,  i32 160,  i32 161,  i32 162,  i32 163,  i32 164,  i32 165,  i32 166,
                                                                      i32 167,  i32 168,  i32 169,  i32 170,  i32 171,  i32 172,  i32 173,  i32 174,
                                                                      i32 175,  i32 176,  i32 177,  i32 178,  i32 179,  i32 180,  i32 181,  i32 182,
                                                                      i32 183,  i32 184,  i32 185,  i32 186,  i32 187,  i32 188,  i32 189,  i32 190,
                                                                      i32 191,  i32 192,  i32 193,  i32 194,  i32 195,  i32 196,  i32 197,  i32 198,
                                                                      i32 199,  i32 200,  i32 201,  i32 202,  i32 203,  i32 204,  i32 205,  i32 206,
                                                                      i32 207,  i32 208,  i32 209,  i32 210,  i32 211,  i32 212,  i32 213,  i32 214,
                                                                      i32 215,  i32 216,  i32 217,  i32 218,  i32 219,  i32 220,  i32 221,  i32 222,
                                                                      i32 223,  i32 224,  i32 225,  i32 226,  i32 227,  i32 228,  i32 229,  i32 230,
                                                                      i32 231,  i32 232,  i32 233,  i32 234,  i32 235,  i32 236,  i32 237,  i32 238,
                                                                      i32 239,  i32 240,  i32 241,  i32 242,  i32 243,  i32 244,  i32 245,  i32 246,
                                                                      i32 247,  i32 248,  i32 249,  i32 250,  i32 251,  i32 252,  i32 253,  i32 254>
  store <128 x i8> %ret, <128 x i8>* %a
  ret void
}

define void @shuffle_ext_byone_v256i8(<256 x i8>* %a, <256 x i8>* %b) #0 {
; VBITS_GE_2048-LABEL: shuffle_ext_byone_v256i8:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.b, vl256
; VBITS_GE_2048-NEXT:    mov w8, #255
; VBITS_GE_2048-NEXT:    ld1b { z0.b }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1b { z1.b }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    whilels p1.b, xzr, x8
; VBITS_GE_2048-NEXT:    lastb w8, p1, z0.b
; VBITS_GE_2048-NEXT:    insr z1.b, w8
; VBITS_GE_2048-NEXT:    st1b { z1.b }, p0, [x0]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <256 x i8>, <256 x i8>* %a
  %op2 = load <256 x i8>, <256 x i8>* %b
  %ret = shufflevector <256 x i8> %op1, <256 x i8> %op2, <256 x i32> <i32 255,  i32 256,  i32 257,  i32 258,  i32 259,  i32 260,  i32 261,  i32 262,
                                                                      i32 263,  i32 264,  i32 265,  i32 266,  i32 267,  i32 268,  i32 269,  i32 270,
                                                                      i32 271,  i32 272,  i32 273,  i32 274,  i32 275,  i32 276,  i32 277,  i32 278,
                                                                      i32 279,  i32 280,  i32 281,  i32 282,  i32 283,  i32 284,  i32 285,  i32 286,
                                                                      i32 287,  i32 288,  i32 289,  i32 290,  i32 291,  i32 292,  i32 293,  i32 294,
                                                                      i32 295,  i32 296,  i32 297,  i32 298,  i32 299,  i32 300,  i32 301,  i32 302,
                                                                      i32 303,  i32 304,  i32 305,  i32 306,  i32 307,  i32 308,  i32 309,  i32 310,
                                                                      i32 311,  i32 312,  i32 313,  i32 314,  i32 315,  i32 316,  i32 317,  i32 318,
                                                                      i32 319,  i32 320,  i32 321,  i32 322,  i32 323,  i32 324,  i32 325,  i32 326,
                                                                      i32 327,  i32 328,  i32 329,  i32 330,  i32 331,  i32 332,  i32 333,  i32 334,
                                                                      i32 335,  i32 336,  i32 337,  i32 338,  i32 339,  i32 340,  i32 341,  i32 342,
                                                                      i32 343,  i32 344,  i32 345,  i32 346,  i32 347,  i32 348,  i32 349,  i32 350,
                                                                      i32 351,  i32 352,  i32 353,  i32 354,  i32 355,  i32 356,  i32 357,  i32 358,
                                                                      i32 359,  i32 360,  i32 361,  i32 362,  i32 363,  i32 364,  i32 365,  i32 366,
                                                                      i32 367,  i32 368,  i32 369,  i32 370,  i32 371,  i32 372,  i32 373,  i32 374,
                                                                      i32 375,  i32 376,  i32 377,  i32 378,  i32 379,  i32 380,  i32 381,  i32 382,
                                                                      i32 383,  i32 384,  i32 385,  i32 386,  i32 387,  i32 388,  i32 389,  i32 390,
                                                                      i32 391,  i32 392,  i32 393,  i32 394,  i32 395,  i32 396,  i32 397,  i32 398,
                                                                      i32 399,  i32 400,  i32 401,  i32 402,  i32 403,  i32 404,  i32 405,  i32 406,
                                                                      i32 407,  i32 408,  i32 409,  i32 410,  i32 411,  i32 412,  i32 413,  i32 414,
                                                                      i32 415,  i32 416,  i32 417,  i32 418,  i32 419,  i32 420,  i32 421,  i32 422,
                                                                      i32 423,  i32 424,  i32 425,  i32 426,  i32 427,  i32 428,  i32 429,  i32 430,
                                                                      i32 431,  i32 432,  i32 433,  i32 434,  i32 435,  i32 436,  i32 437,  i32 438,
                                                                      i32 439,  i32 440,  i32 441,  i32 442,  i32 443,  i32 444,  i32 445,  i32 446,
                                                                      i32 447,  i32 448,  i32 449,  i32 450,  i32 451,  i32 452,  i32 453,  i32 454,
                                                                      i32 455,  i32 456,  i32 457,  i32 458,  i32 459,  i32 460,  i32 461,  i32 462,
                                                                      i32 463,  i32 464,  i32 465,  i32 466,  i32 467,  i32 468,  i32 469,  i32 470,
                                                                      i32 471,  i32 472,  i32 473,  i32 474,  i32 475,  i32 476,  i32 477,  i32 478,
                                                                      i32 479,  i32 480,  i32 481,  i32 482,  i32 483,  i32 484,  i32 485,  i32 486,
                                                                      i32 487,  i32 488,  i32 489,  i32 490,  i32 491,  i32 492,  i32 493,  i32 494,
                                                                      i32 495,  i32 496,  i32 497,  i32 498,  i32 499,  i32 500,  i32 501,  i32 502,
                                                                      i32 503,  i32 504,  i32 505,  i32 506,  i32 507,  i32 508,  i32 509,  i32 510>
  store <256 x i8> %ret, <256 x i8>* %a
  ret void
}

; Don't use SVE for 64-bit vectors
define <4 x i16> @shuffle_ext_byone_v4i16(<4 x i16> %op1, <4 x i16> %op2) #0 {
; CHECK-LABEL: shuffle_ext_byone_v4i16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ext v0.8b, v0.8b, v1.8b, #6
; CHECK-NEXT:    ret
  %ret = shufflevector <4 x i16> %op1, <4 x i16> %op2, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
  ret <4 x i16> %ret
}

; Don't use SVE for 128-bit vectors
define <8 x i16> @shuffle_ext_byone_v8i16(<8 x i16> %op1, <8 x i16> %op2) #0 {
; CHECK-LABEL: shuffle_ext_byone_v8i16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ext v0.16b, v0.16b, v1.16b, #14
; CHECK-NEXT:    ret
  %ret = shufflevector <8 x i16> %op1, <8 x i16> %op2, <8 x i32> <i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14>
  ret <8 x i16> %ret
}

define void @shuffle_ext_byone_v16i16(<16 x i16>* %a, <16 x i16>* %b) #0 {
; CHECK-LABEL: shuffle_ext_byone_v16i16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.h, vl16
; CHECK-NEXT:    ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT:    ld1h { z1.h }, p0/z, [x1]
; CHECK-NEXT:    mov z0.h, z0.h[15]
; CHECK-NEXT:    fmov w8, s0
; CHECK-NEXT:    insr z1.h, w8
; CHECK-NEXT:    st1h { z1.h }, p0, [x0]
; CHECK-NEXT:    ret
  %op1 = load <16 x i16>, <16 x i16>* %a
  %op2 = load <16 x i16>, <16 x i16>* %b
  %ret = shufflevector <16 x i16> %op1, <16 x i16> %op2, <16 x i32> <i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22,
                                                                     i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30>
  store <16 x i16> %ret, <16 x i16>* %a
  ret void
}

define void @shuffle_ext_byone_v32i16(<32 x i16>* %a, <32 x i16>* %b) #0 {
; Ensure sensible type legalisation.
; VBITS_EQ_256-LABEL: shuffle_ext_byone_v32i16:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #16
; VBITS_EQ_256-NEXT:    ptrue p0.h, vl16
; VBITS_EQ_256-NEXT:    ld1h { z0.h }, p0/z, [x0, x8, lsl #1]
; VBITS_EQ_256-NEXT:    ld1h { z1.h }, p0/z, [x1, x8, lsl #1]
; VBITS_EQ_256-NEXT:    ld1h { z2.h }, p0/z, [x1]
; VBITS_EQ_256-NEXT:    mov z0.h, z0.h[15]
; VBITS_EQ_256-NEXT:    mov z3.h, z2.h[15]
; VBITS_EQ_256-NEXT:    fmov w9, s0
; VBITS_EQ_256-NEXT:    fmov w10, s3
; VBITS_EQ_256-NEXT:    insr z2.h, w9
; VBITS_EQ_256-NEXT:    insr z1.h, w10
; VBITS_EQ_256-NEXT:    st1h { z2.h }, p0, [x0]
; VBITS_EQ_256-NEXT:    st1h { z1.h }, p0, [x0, x8, lsl #1]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: shuffle_ext_byone_v32i16:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.h, vl32
; VBITS_GE_512-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ld1h { z1.h }, p0/z, [x1]
; VBITS_GE_512-NEXT:    mov z0.h, z0.h[31]
; VBITS_GE_512-NEXT:    fmov w8, s0
; VBITS_GE_512-NEXT:    insr z1.h, w8
; VBITS_GE_512-NEXT:    st1h { z1.h }, p0, [x0]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <32 x i16>, <32 x i16>* %a
  %op2 = load <32 x i16>, <32 x i16>* %b
  %ret = shufflevector <32 x i16> %op1, <32 x i16> %op2, <32 x i32> <i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38,
                                                                     i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46,
                                                                     i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54,
                                                                     i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62>
  store <32 x i16> %ret, <32 x i16>* %a
  ret void
}

define void @shuffle_ext_byone_v64i16(<64 x i16>* %a, <64 x i16>* %b) #0 {
; VBITS_GE_1024-LABEL: shuffle_ext_byone_v64i16:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.h, vl64
; VBITS_GE_1024-NEXT:    mov w8, #63
; VBITS_GE_1024-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ld1h { z1.h }, p0/z, [x1]
; VBITS_GE_1024-NEXT:    whilels p1.h, xzr, x8
; VBITS_GE_1024-NEXT:    lastb w8, p1, z0.h
; VBITS_GE_1024-NEXT:    insr z1.h, w8
; VBITS_GE_1024-NEXT:    st1h { z1.h }, p0, [x0]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <64 x i16>, <64 x i16>* %a
  %op2 = load <64 x i16>, <64 x i16>* %b
  %ret = shufflevector <64 x i16> %op1, <64 x i16> %op2, <64 x i32> <i32 63, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70,
                                                                     i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78,
                                                                     i32 79, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86,
                                                                     i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94,
                                                                     i32 95, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102,
                                                                     i32 103, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110,
                                                                     i32 111, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118,
                                                                     i32 119, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126>
  store <64 x i16> %ret, <64 x i16>* %a
  ret void
}

define void @shuffle_ext_byone_v128i16(<128 x i16>* %a, <128 x i16>* %b) #0 {
; VBITS_GE_2048-LABEL: shuffle_ext_byone_v128i16:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.h, vl128
; VBITS_GE_2048-NEXT:    mov w8, #127
; VBITS_GE_2048-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1h { z1.h }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    whilels p1.h, xzr, x8
; VBITS_GE_2048-NEXT:    lastb w8, p1, z0.h
; VBITS_GE_2048-NEXT:    insr z1.h, w8
; VBITS_GE_2048-NEXT:    st1h { z1.h }, p0, [x0]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <128 x i16>, <128 x i16>* %a
  %op2 = load <128 x i16>, <128 x i16>* %b
  %ret = shufflevector <128 x i16> %op1, <128 x i16> %op2, <128 x i32> <i32 127,  i32 128,  i32 129,  i32 130,  i32 131,  i32 132,  i32 133,  i32 134,
                                                                        i32 135,  i32 136,  i32 137,  i32 138,  i32 139,  i32 140,  i32 141,  i32 142,
                                                                        i32 143,  i32 144,  i32 145,  i32 146,  i32 147,  i32 148,  i32 149,  i32 150,
                                                                        i32 151,  i32 152,  i32 153,  i32 154,  i32 155,  i32 156,  i32 157,  i32 158,
                                                                        i32 159,  i32 160,  i32 161,  i32 162,  i32 163,  i32 164,  i32 165,  i32 166,
                                                                        i32 167,  i32 168,  i32 169,  i32 170,  i32 171,  i32 172,  i32 173,  i32 174,
                                                                        i32 175,  i32 176,  i32 177,  i32 178,  i32 179,  i32 180,  i32 181,  i32 182,
                                                                        i32 183,  i32 184,  i32 185,  i32 186,  i32 187,  i32 188,  i32 189,  i32 190,
                                                                        i32 191,  i32 192,  i32 193,  i32 194,  i32 195,  i32 196,  i32 197,  i32 198,
                                                                        i32 199,  i32 200,  i32 201,  i32 202,  i32 203,  i32 204,  i32 205,  i32 206,
                                                                        i32 207,  i32 208,  i32 209,  i32 210,  i32 211,  i32 212,  i32 213,  i32 214,
                                                                        i32 215,  i32 216,  i32 217,  i32 218,  i32 219,  i32 220,  i32 221,  i32 222,
                                                                        i32 223,  i32 224,  i32 225,  i32 226,  i32 227,  i32 228,  i32 229,  i32 230,
                                                                        i32 231,  i32 232,  i32 233,  i32 234,  i32 235,  i32 236,  i32 237,  i32 238,
                                                                        i32 239,  i32 240,  i32 241,  i32 242,  i32 243,  i32 244,  i32 245,  i32 246,
                                                                        i32 247,  i32 248,  i32 249,  i32 250,  i32 251,  i32 252,  i32 253,  i32 254>
  store <128 x i16> %ret, <128 x i16>* %a
  ret void
}

; Don't use SVE for 64-bit vectors
define <2 x i32> @shuffle_ext_byone_v2i32(<2 x i32> %op1, <2 x i32> %op2) #0 {
; CHECK-LABEL: shuffle_ext_byone_v2i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ext v0.8b, v0.8b, v1.8b, #4
; CHECK-NEXT:    ret
  %ret = shufflevector <2 x i32> %op1, <2 x i32> %op2, <2 x i32> <i32 1, i32 2>
  ret <2 x i32> %ret
}

; Don't use SVE for 128-bit vectors
define <4 x i32> @shuffle_ext_byone_v4i32(<4 x i32> %op1, <4 x i32> %op2) #0 {
; CHECK-LABEL: shuffle_ext_byone_v4i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ext v0.16b, v0.16b, v1.16b, #12
; CHECK-NEXT:    ret
  %ret = shufflevector <4 x i32> %op1, <4 x i32> %op2, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
  ret <4 x i32> %ret
}

define void @shuffle_ext_byone_v8i32(<8 x i32>* %a, <8 x i32>* %b) #0 {
; CHECK-LABEL: shuffle_ext_byone_v8i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.s, vl8
; CHECK-NEXT:    ld1w { z0.s }, p0/z, [x0]
; CHECK-NEXT:    ld1w { z1.s }, p0/z, [x1]
; CHECK-NEXT:    mov z0.s, z0.s[7]
; CHECK-NEXT:    fmov w8, s0
; CHECK-NEXT:    insr z1.s, w8
; CHECK-NEXT:    st1w { z1.s }, p0, [x0]
; CHECK-NEXT:    ret
  %op1 = load <8 x i32>, <8 x i32>* %a
  %op2 = load <8 x i32>, <8 x i32>* %b
  %ret = shufflevector <8 x i32> %op1, <8 x i32> %op2, <8 x i32> <i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14>
  store <8 x i32> %ret, <8 x i32>* %a
  ret void
}

define void @shuffle_ext_byone_v16i32(<16 x i32>* %a, <16 x i32>* %b) #0 {
; Ensure sensible type legalisation.
; VBITS_EQ_256-LABEL: shuffle_ext_byone_v16i32:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #8
; VBITS_EQ_256-NEXT:    ptrue p0.s, vl8
; VBITS_EQ_256-NEXT:    ld1w { z0.s }, p0/z, [x0, x8, lsl #2]
; VBITS_EQ_256-NEXT:    ld1w { z1.s }, p0/z, [x1, x8, lsl #2]
; VBITS_EQ_256-NEXT:    ld1w { z2.s }, p0/z, [x1]
; VBITS_EQ_256-NEXT:    mov z0.s, z0.s[7]
; VBITS_EQ_256-NEXT:    mov z3.s, z2.s[7]
; VBITS_EQ_256-NEXT:    fmov w9, s0
; VBITS_EQ_256-NEXT:    fmov w10, s3
; VBITS_EQ_256-NEXT:    insr z2.s, w9
; VBITS_EQ_256-NEXT:    insr z1.s, w10
; VBITS_EQ_256-NEXT:    st1w { z2.s }, p0, [x0]
; VBITS_EQ_256-NEXT:    st1w { z1.s }, p0, [x0, x8, lsl #2]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: shuffle_ext_byone_v16i32:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.s, vl16
; VBITS_GE_512-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ld1w { z1.s }, p0/z, [x1]
; VBITS_GE_512-NEXT:    mov z0.s, z0.s[15]
; VBITS_GE_512-NEXT:    fmov w8, s0
; VBITS_GE_512-NEXT:    insr z1.s, w8
; VBITS_GE_512-NEXT:    st1w { z1.s }, p0, [x0]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <16 x i32>, <16 x i32>* %a
  %op2 = load <16 x i32>, <16 x i32>* %b
  %ret = shufflevector <16 x i32> %op1, <16 x i32> %op2, <16 x i32> <i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22,
                                                                     i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30>
  store <16 x i32> %ret, <16 x i32>* %a
  ret void
}

define void @shuffle_ext_byone_v32i32(<32 x i32>* %a, <32 x i32>* %b) #0 {
; VBITS_GE_1024-LABEL: shuffle_ext_byone_v32i32:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.s, vl32
; VBITS_GE_1024-NEXT:    mov w8, #31
; VBITS_GE_1024-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ld1w { z1.s }, p0/z, [x1]
; VBITS_GE_1024-NEXT:    whilels p1.s, xzr, x8
; VBITS_GE_1024-NEXT:    lastb w8, p1, z0.s
; VBITS_GE_1024-NEXT:    insr z1.s, w8
; VBITS_GE_1024-NEXT:    st1w { z1.s }, p0, [x0]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <32 x i32>, <32 x i32>* %a
  %op2 = load <32 x i32>, <32 x i32>* %b
  %ret = shufflevector <32 x i32> %op1, <32 x i32> %op2, <32 x i32> <i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38,
                                                                     i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46,
                                                                     i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54,
                                                                     i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62>
  store <32 x i32> %ret, <32 x i32>* %a
  ret void
}

define void @shuffle_ext_byone_v64i32(<64 x i32>* %a, <64 x i32>* %b) #0 {
; VBITS_GE_2048-LABEL: shuffle_ext_byone_v64i32:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl64
; VBITS_GE_2048-NEXT:    mov w8, #63
; VBITS_GE_2048-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1w { z1.s }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    whilels p1.s, xzr, x8
; VBITS_GE_2048-NEXT:    lastb w8, p1, z0.s
; VBITS_GE_2048-NEXT:    insr z1.s, w8
; VBITS_GE_2048-NEXT:    st1w { z1.s }, p0, [x0]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <64 x i32>, <64 x i32>* %a
  %op2 = load <64 x i32>, <64 x i32>* %b
  %ret = shufflevector <64 x i32> %op1, <64 x i32> %op2, <64 x i32> <i32 63, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70,
                                                                     i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78,
                                                                     i32 79, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86,
                                                                     i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94,
                                                                     i32 95, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102,
                                                                     i32 103, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110,
                                                                     i32 111, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118,
                                                                     i32 119, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126>
  store <64 x i32> %ret, <64 x i32>* %a
  ret void
}

; Don't use SVE for 128-bit vectors
define <2 x i64> @shuffle_ext_byone_v2i64(<2 x i64> %op1, <2 x i64> %op2) #0 {
; CHECK-LABEL: shuffle_ext_byone_v2i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ext v0.16b, v0.16b, v1.16b, #8
; CHECK-NEXT:    ret
  %ret = shufflevector <2 x i64> %op1, <2 x i64> %op2, <2 x i32> <i32 1, i32 2>
  ret <2 x i64> %ret
}

define void @shuffle_ext_byone_v4i64(<4 x i64>* %a, <4 x i64>* %b) #0 {
; CHECK-LABEL: shuffle_ext_byone_v4i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ld1d { z1.d }, p0/z, [x1]
; CHECK-NEXT:    mov z0.d, z0.d[3]
; CHECK-NEXT:    fmov x8, d0
; CHECK-NEXT:    insr z1.d, x8
; CHECK-NEXT:    st1d { z1.d }, p0, [x0]
; CHECK-NEXT:    ret
  %op1 = load <4 x i64>, <4 x i64>* %a
  %op2 = load <4 x i64>, <4 x i64>* %b
  %ret = shufflevector <4 x i64> %op1, <4 x i64> %op2, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
  store <4 x i64> %ret, <4 x i64>* %a
  ret void
}

define void @shuffle_ext_byone_v8i64(<8 x i64>* %a, <8 x i64>* %b) #0 {
; Ensure sensible type legalisation.
; VBITS_EQ_256-LABEL: shuffle_ext_byone_v8i64:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #4
; VBITS_EQ_256-NEXT:    ptrue p0.d, vl4
; VBITS_EQ_256-NEXT:    ld1d { z0.d }, p0/z, [x0, x8, lsl #3]
; VBITS_EQ_256-NEXT:    ld1d { z1.d }, p0/z, [x1, x8, lsl #3]
; VBITS_EQ_256-NEXT:    ld1d { z2.d }, p0/z, [x1]
; VBITS_EQ_256-NEXT:    mov z0.d, z0.d[3]
; VBITS_EQ_256-NEXT:    mov z3.d, z2.d[3]
; VBITS_EQ_256-NEXT:    fmov x9, d0
; VBITS_EQ_256-NEXT:    fmov x10, d3
; VBITS_EQ_256-NEXT:    insr z2.d, x9
; VBITS_EQ_256-NEXT:    insr z1.d, x10
; VBITS_EQ_256-NEXT:    st1d { z2.d }, p0, [x0]
; VBITS_EQ_256-NEXT:    st1d { z1.d }, p0, [x0, x8, lsl #3]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: shuffle_ext_byone_v8i64:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ld1d { z1.d }, p0/z, [x1]
; VBITS_GE_512-NEXT:    mov z0.d, z0.d[7]
; VBITS_GE_512-NEXT:    fmov x8, d0
; VBITS_GE_512-NEXT:    insr z1.d, x8
; VBITS_GE_512-NEXT:    st1d { z1.d }, p0, [x0]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <8 x i64>, <8 x i64>* %a
  %op2 = load <8 x i64>, <8 x i64>* %b
  %ret = shufflevector <8 x i64> %op1, <8 x i64> %op2, <8 x i32> <i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14>
  store <8 x i64> %ret, <8 x i64>* %a
  ret void
}

define void @shuffle_ext_byone_v16i64(<16 x i64>* %a, <16 x i64>* %b) #0 {
; VBITS_GE_1024-LABEL: shuffle_ext_byone_v16i64:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.d, vl16
; VBITS_GE_1024-NEXT:    mov w8, #15
; VBITS_GE_1024-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ld1d { z1.d }, p0/z, [x1]
; VBITS_GE_1024-NEXT:    whilels p1.d, xzr, x8
; VBITS_GE_1024-NEXT:    lastb x8, p1, z0.d
; VBITS_GE_1024-NEXT:    insr z1.d, x8
; VBITS_GE_1024-NEXT:    st1d { z1.d }, p0, [x0]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <16 x i64>, <16 x i64>* %a
  %op2 = load <16 x i64>, <16 x i64>* %b
  %ret = shufflevector <16 x i64> %op1, <16 x i64> %op2, <16 x i32> <i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22,
                                                                     i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30>
  store <16 x i64> %ret, <16 x i64>* %a
  ret void
}

define void @shuffle_ext_byone_v32i64(<32 x i64>* %a, <32 x i64>* %b) #0 {
; VBITS_GE_2048-LABEL: shuffle_ext_byone_v32i64:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.d, vl32
; VBITS_GE_2048-NEXT:    mov w8, #31
; VBITS_GE_2048-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1d { z1.d }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    whilels p1.d, xzr, x8
; VBITS_GE_2048-NEXT:    lastb x8, p1, z0.d
; VBITS_GE_2048-NEXT:    insr z1.d, x8
; VBITS_GE_2048-NEXT:    st1d { z1.d }, p0, [x0]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <32 x i64>, <32 x i64>* %a
  %op2 = load <32 x i64>, <32 x i64>* %b
  %ret = shufflevector <32 x i64> %op1, <32 x i64> %op2, <32 x i32> <i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38,
                                                                     i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46,
                                                                     i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54,
                                                                     i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62>
  store <32 x i64> %ret, <32 x i64>* %a
  ret void
}

; Don't use SVE for 64-bit vectors
define <4 x half> @shuffle_ext_byone_v4f16(<4 x half> %op1, <4 x half> %op2) #0 {
; CHECK-LABEL: shuffle_ext_byone_v4f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ext v0.8b, v0.8b, v1.8b, #6
; CHECK-NEXT:    ret
  %ret = shufflevector <4 x half> %op1, <4 x half> %op2, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
  ret <4 x half> %ret
}

; Don't use SVE for 128-bit vectors
define <8 x half> @shuffle_ext_byone_v8f16(<8 x half> %op1, <8 x half> %op2) #0 {
; CHECK-LABEL: shuffle_ext_byone_v8f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ext v0.16b, v0.16b, v1.16b, #14
; CHECK-NEXT:    ret
  %ret = shufflevector <8 x half> %op1, <8 x half> %op2, <8 x i32> <i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14>
  ret <8 x half> %ret
}

define void @shuffle_ext_byone_v16f16(<16 x half>* %a, <16 x half>* %b) #0 {
; CHECK-LABEL: shuffle_ext_byone_v16f16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.h, vl16
; CHECK-NEXT:    ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT:    ld1h { z1.h }, p0/z, [x1]
; CHECK-NEXT:    mov z0.h, z0.h[15]
; CHECK-NEXT:    insr z1.h, h0
; CHECK-NEXT:    st1h { z1.h }, p0, [x0]
; CHECK-NEXT:    ret
  %op1 = load <16 x half>, <16 x half>* %a
  %op2 = load <16 x half>, <16 x half>* %b
  %ret = shufflevector <16 x half> %op1, <16 x half> %op2, <16 x i32> <i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22,
                                                                       i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30>
  store <16 x half> %ret, <16 x half>* %a
  ret void
}

define void @shuffle_ext_byone_v32f16(<32 x half>* %a, <32 x half>* %b) #0 {
; Ensure sensible type legalisation.
; VBITS_EQ_256-LABEL: shuffle_ext_byone_v32f16:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #16
; VBITS_EQ_256-NEXT:    ptrue p0.h, vl16
; VBITS_EQ_256-NEXT:    ld1h { z0.h }, p0/z, [x0, x8, lsl #1]
; VBITS_EQ_256-NEXT:    ld1h { z1.h }, p0/z, [x1, x8, lsl #1]
; VBITS_EQ_256-NEXT:    ld1h { z2.h }, p0/z, [x1]
; VBITS_EQ_256-NEXT:    mov z0.h, z0.h[15]
; VBITS_EQ_256-NEXT:    mov z3.h, z2.h[15]
; VBITS_EQ_256-NEXT:    insr z2.h, h0
; VBITS_EQ_256-NEXT:    insr z1.h, h3
; VBITS_EQ_256-NEXT:    st1h { z2.h }, p0, [x0]
; VBITS_EQ_256-NEXT:    st1h { z1.h }, p0, [x0, x8, lsl #1]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: shuffle_ext_byone_v32f16:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.h, vl32
; VBITS_GE_512-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ld1h { z1.h }, p0/z, [x1]
; VBITS_GE_512-NEXT:    mov z0.h, z0.h[31]
; VBITS_GE_512-NEXT:    insr z1.h, h0
; VBITS_GE_512-NEXT:    st1h { z1.h }, p0, [x0]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <32 x half>, <32 x half>* %a
  %op2 = load <32 x half>, <32 x half>* %b
  %ret = shufflevector <32 x half> %op1, <32 x half> %op2, <32 x i32> <i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38,
                                                                       i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46,
                                                                       i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54,
                                                                       i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62>
  store <32 x half> %ret, <32 x half>* %a
  ret void
}

define void @shuffle_ext_byone_v64f16(<64 x half>* %a, <64 x half>* %b) #0 {
; VBITS_GE_1024-LABEL: shuffle_ext_byone_v64f16:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.h, vl64
; VBITS_GE_1024-NEXT:    mov w8, #63
; VBITS_GE_1024-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ld1h { z1.h }, p0/z, [x1]
; VBITS_GE_1024-NEXT:    whilels p1.h, xzr, x8
; VBITS_GE_1024-NEXT:    lastb h0, p1, z0.h
; VBITS_GE_1024-NEXT:    insr z1.h, h0
; VBITS_GE_1024-NEXT:    st1h { z1.h }, p0, [x0]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <64 x half>, <64 x half>* %a
  %op2 = load <64 x half>, <64 x half>* %b
  %ret = shufflevector <64 x half> %op1, <64 x half> %op2, <64 x i32> <i32 63, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70,
                                                                       i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78,
                                                                       i32 79, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86,
                                                                       i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94,
                                                                       i32 95, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102,
                                                                       i32 103, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110,
                                                                       i32 111, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118,
                                                                       i32 119, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126>
  store <64 x half> %ret, <64 x half>* %a
  ret void
}

define void @shuffle_ext_byone_v128f16(<128 x half>* %a, <128 x half>* %b) #0 {
; VBITS_GE_2048-LABEL: shuffle_ext_byone_v128f16:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.h, vl128
; VBITS_GE_2048-NEXT:    mov w8, #127
; VBITS_GE_2048-NEXT:    ld1h { z0.h }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1h { z1.h }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    whilels p1.h, xzr, x8
; VBITS_GE_2048-NEXT:    lastb h0, p1, z0.h
; VBITS_GE_2048-NEXT:    insr z1.h, h0
; VBITS_GE_2048-NEXT:    st1h { z1.h }, p0, [x0]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <128 x half>, <128 x half>* %a
  %op2 = load <128 x half>, <128 x half>* %b
  %ret = shufflevector <128 x half> %op1, <128 x half> %op2, <128 x i32> <i32 127,  i32 128,  i32 129,  i32 130,  i32 131,  i32 132,  i32 133,  i32 134,
                                                                          i32 135,  i32 136,  i32 137,  i32 138,  i32 139,  i32 140,  i32 141,  i32 142,
                                                                          i32 143,  i32 144,  i32 145,  i32 146,  i32 147,  i32 148,  i32 149,  i32 150,
                                                                          i32 151,  i32 152,  i32 153,  i32 154,  i32 155,  i32 156,  i32 157,  i32 158,
                                                                          i32 159,  i32 160,  i32 161,  i32 162,  i32 163,  i32 164,  i32 165,  i32 166,
                                                                          i32 167,  i32 168,  i32 169,  i32 170,  i32 171,  i32 172,  i32 173,  i32 174,
                                                                          i32 175,  i32 176,  i32 177,  i32 178,  i32 179,  i32 180,  i32 181,  i32 182,
                                                                          i32 183,  i32 184,  i32 185,  i32 186,  i32 187,  i32 188,  i32 189,  i32 190,
                                                                          i32 191,  i32 192,  i32 193,  i32 194,  i32 195,  i32 196,  i32 197,  i32 198,
                                                                          i32 199,  i32 200,  i32 201,  i32 202,  i32 203,  i32 204,  i32 205,  i32 206,
                                                                          i32 207,  i32 208,  i32 209,  i32 210,  i32 211,  i32 212,  i32 213,  i32 214,
                                                                          i32 215,  i32 216,  i32 217,  i32 218,  i32 219,  i32 220,  i32 221,  i32 222,
                                                                          i32 223,  i32 224,  i32 225,  i32 226,  i32 227,  i32 228,  i32 229,  i32 230,
                                                                          i32 231,  i32 232,  i32 233,  i32 234,  i32 235,  i32 236,  i32 237,  i32 238,
                                                                          i32 239,  i32 240,  i32 241,  i32 242,  i32 243,  i32 244,  i32 245,  i32 246,
                                                                          i32 247,  i32 248,  i32 249,  i32 250,  i32 251,  i32 252,  i32 253,  i32 254>
  store <128 x half> %ret, <128 x half>* %a
  ret void
}

; Don't use SVE for 64-bit vectors
define <2 x float> @shuffle_ext_byone_v2f32(<2 x float> %op1, <2 x float> %op2) #0 {
; CHECK-LABEL: shuffle_ext_byone_v2f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ext v0.8b, v0.8b, v1.8b, #4
; CHECK-NEXT:    ret
  %ret = shufflevector <2 x float> %op1, <2 x float> %op2, <2 x i32> <i32 1, i32 2>
  ret <2 x float> %ret
}

; Don't use SVE for 128-bit vectors
define <4 x float> @shuffle_ext_byone_v4f32(<4 x float> %op1, <4 x float> %op2) #0 {
; CHECK-LABEL: shuffle_ext_byone_v4f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ext v0.16b, v0.16b, v1.16b, #12
; CHECK-NEXT:    ret
  %ret = shufflevector <4 x float> %op1, <4 x float> %op2, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
  ret <4 x float> %ret
}

define void @shuffle_ext_byone_v8f32(<8 x float>* %a, <8 x float>* %b) #0 {
; CHECK-LABEL: shuffle_ext_byone_v8f32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.s, vl8
; CHECK-NEXT:    ld1w { z0.s }, p0/z, [x0]
; CHECK-NEXT:    ld1w { z1.s }, p0/z, [x1]
; CHECK-NEXT:    mov z0.s, z0.s[7]
; CHECK-NEXT:    insr z1.s, s0
; CHECK-NEXT:    st1w { z1.s }, p0, [x0]
; CHECK-NEXT:    ret
  %op1 = load <8 x float>, <8 x float>* %a
  %op2 = load <8 x float>, <8 x float>* %b
  %ret = shufflevector <8 x float> %op1, <8 x float> %op2, <8 x i32> <i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14>
  store <8 x float> %ret, <8 x float>* %a
  ret void
}

define void @shuffle_ext_byone_v16f32(<16 x float>* %a, <16 x float>* %b) #0 {
; Ensure sensible type legalisation.
; VBITS_EQ_256-LABEL: shuffle_ext_byone_v16f32:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #8
; VBITS_EQ_256-NEXT:    ptrue p0.s, vl8
; VBITS_EQ_256-NEXT:    ld1w { z0.s }, p0/z, [x0, x8, lsl #2]
; VBITS_EQ_256-NEXT:    ld1w { z1.s }, p0/z, [x1, x8, lsl #2]
; VBITS_EQ_256-NEXT:    ld1w { z2.s }, p0/z, [x1]
; VBITS_EQ_256-NEXT:    mov z0.s, z0.s[7]
; VBITS_EQ_256-NEXT:    mov z3.s, z2.s[7]
; VBITS_EQ_256-NEXT:    insr z2.s, s0
; VBITS_EQ_256-NEXT:    insr z1.s, s3
; VBITS_EQ_256-NEXT:    st1w { z2.s }, p0, [x0]
; VBITS_EQ_256-NEXT:    st1w { z1.s }, p0, [x0, x8, lsl #2]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: shuffle_ext_byone_v16f32:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.s, vl16
; VBITS_GE_512-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ld1w { z1.s }, p0/z, [x1]
; VBITS_GE_512-NEXT:    mov z0.s, z0.s[15]
; VBITS_GE_512-NEXT:    insr z1.s, s0
; VBITS_GE_512-NEXT:    st1w { z1.s }, p0, [x0]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <16 x float>, <16 x float>* %a
  %op2 = load <16 x float>, <16 x float>* %b
  %ret = shufflevector <16 x float> %op1, <16 x float> %op2, <16 x i32> <i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22,
                                                                         i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30>
  store <16 x float> %ret, <16 x float>* %a
  ret void
}

define void @shuffle_ext_byone_v32f32(<32 x float>* %a, <32 x float>* %b) #0 {
; VBITS_GE_1024-LABEL: shuffle_ext_byone_v32f32:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.s, vl32
; VBITS_GE_1024-NEXT:    mov w8, #31
; VBITS_GE_1024-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ld1w { z1.s }, p0/z, [x1]
; VBITS_GE_1024-NEXT:    whilels p1.s, xzr, x8
; VBITS_GE_1024-NEXT:    lastb s0, p1, z0.s
; VBITS_GE_1024-NEXT:    insr z1.s, s0
; VBITS_GE_1024-NEXT:    st1w { z1.s }, p0, [x0]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <32 x float>, <32 x float>* %a
  %op2 = load <32 x float>, <32 x float>* %b
  %ret = shufflevector <32 x float> %op1, <32 x float> %op2, <32 x i32> <i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38,
                                                                         i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46,
                                                                         i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54,
                                                                         i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62>
  store <32 x float> %ret, <32 x float>* %a
  ret void
}

define void @shuffle_ext_byone_v64f32(<64 x float>* %a, <64 x float>* %b) #0 {
; VBITS_GE_2048-LABEL: shuffle_ext_byone_v64f32:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.s, vl64
; VBITS_GE_2048-NEXT:    mov w8, #63
; VBITS_GE_2048-NEXT:    ld1w { z0.s }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1w { z1.s }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    whilels p1.s, xzr, x8
; VBITS_GE_2048-NEXT:    lastb s0, p1, z0.s
; VBITS_GE_2048-NEXT:    insr z1.s, s0
; VBITS_GE_2048-NEXT:    st1w { z1.s }, p0, [x0]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <64 x float>, <64 x float>* %a
  %op2 = load <64 x float>, <64 x float>* %b
  %ret = shufflevector <64 x float> %op1, <64 x float> %op2, <64 x i32> <i32 63, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70,
                                                                         i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78,
                                                                         i32 79, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86,
                                                                         i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94,
                                                                         i32 95, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102,
                                                                         i32 103, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110,
                                                                         i32 111, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118,
                                                                         i32 119, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126>
  store <64 x float> %ret, <64 x float>* %a
  ret void
}

; Don't use SVE for 128-bit vectors
define <2 x double> @shuffle_ext_byone_v2f64(<2 x double> %op1, <2 x double> %op2) #0 {
; CHECK-LABEL: shuffle_ext_byone_v2f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ext v0.16b, v0.16b, v1.16b, #8
; CHECK-NEXT:    ret
  %ret = shufflevector <2 x double> %op1, <2 x double> %op2, <2 x i32> <i32 1, i32 2>
  ret <2 x double> %ret
}

define void @shuffle_ext_byone_v4f64(<4 x double>* %a, <4 x double>* %b) #0 {
; CHECK-LABEL: shuffle_ext_byone_v4f64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ld1d { z1.d }, p0/z, [x1]
; CHECK-NEXT:    mov z0.d, z0.d[3]
; CHECK-NEXT:    insr z1.d, d0
; CHECK-NEXT:    st1d { z1.d }, p0, [x0]
; CHECK-NEXT:    ret
  %op1 = load <4 x double>, <4 x double>* %a
  %op2 = load <4 x double>, <4 x double>* %b
  %ret = shufflevector <4 x double> %op1, <4 x double> %op2, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
  store <4 x double> %ret, <4 x double>* %a
  ret void
}

define void @shuffle_ext_byone_v8f64(<8 x double>* %a, <8 x double>* %b) #0 {
; Ensure sensible type legalisation.
; VBITS_EQ_256-LABEL: shuffle_ext_byone_v8f64:
; VBITS_EQ_256:       // %bb.0:
; VBITS_EQ_256-NEXT:    mov x8, #4
; VBITS_EQ_256-NEXT:    ptrue p0.d, vl4
; VBITS_EQ_256-NEXT:    ld1d { z0.d }, p0/z, [x0, x8, lsl #3]
; VBITS_EQ_256-NEXT:    ld1d { z1.d }, p0/z, [x1, x8, lsl #3]
; VBITS_EQ_256-NEXT:    ld1d { z2.d }, p0/z, [x1]
; VBITS_EQ_256-NEXT:    mov z0.d, z0.d[3]
; VBITS_EQ_256-NEXT:    mov z3.d, z2.d[3]
; VBITS_EQ_256-NEXT:    insr z2.d, d0
; VBITS_EQ_256-NEXT:    insr z1.d, d3
; VBITS_EQ_256-NEXT:    st1d { z2.d }, p0, [x0]
; VBITS_EQ_256-NEXT:    st1d { z1.d }, p0, [x0, x8, lsl #3]
; VBITS_EQ_256-NEXT:    ret
;
; VBITS_GE_512-LABEL: shuffle_ext_byone_v8f64:
; VBITS_GE_512:       // %bb.0:
; VBITS_GE_512-NEXT:    ptrue p0.d, vl8
; VBITS_GE_512-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_512-NEXT:    ld1d { z1.d }, p0/z, [x1]
; VBITS_GE_512-NEXT:    mov z0.d, z0.d[7]
; VBITS_GE_512-NEXT:    insr z1.d, d0
; VBITS_GE_512-NEXT:    st1d { z1.d }, p0, [x0]
; VBITS_GE_512-NEXT:    ret
  %op1 = load <8 x double>, <8 x double>* %a
  %op2 = load <8 x double>, <8 x double>* %b
  %ret = shufflevector <8 x double> %op1, <8 x double> %op2, <8 x i32> <i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14>
  store <8 x double> %ret, <8 x double>* %a
  ret void
}

define void @shuffle_ext_byone_v16f64(<16 x double>* %a, <16 x double>* %b) #0 {
; VBITS_GE_1024-LABEL: shuffle_ext_byone_v16f64:
; VBITS_GE_1024:       // %bb.0:
; VBITS_GE_1024-NEXT:    ptrue p0.d, vl16
; VBITS_GE_1024-NEXT:    mov w8, #15
; VBITS_GE_1024-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_1024-NEXT:    ld1d { z1.d }, p0/z, [x1]
; VBITS_GE_1024-NEXT:    whilels p1.d, xzr, x8
; VBITS_GE_1024-NEXT:    lastb d0, p1, z0.d
; VBITS_GE_1024-NEXT:    insr z1.d, d0
; VBITS_GE_1024-NEXT:    st1d { z1.d }, p0, [x0]
; VBITS_GE_1024-NEXT:    ret
  %op1 = load <16 x double>, <16 x double>* %a
  %op2 = load <16 x double>, <16 x double>* %b
  %ret = shufflevector <16 x double> %op1, <16 x double> %op2, <16 x i32> <i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22,
                                                                           i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30>
  store <16 x double> %ret, <16 x double>* %a
  ret void
}

define void @shuffle_ext_byone_v32f64(<32 x double>* %a, <32 x double>* %b) #0 {
; VBITS_GE_2048-LABEL: shuffle_ext_byone_v32f64:
; VBITS_GE_2048:       // %bb.0:
; VBITS_GE_2048-NEXT:    ptrue p0.d, vl32
; VBITS_GE_2048-NEXT:    mov w8, #31
; VBITS_GE_2048-NEXT:    ld1d { z0.d }, p0/z, [x0]
; VBITS_GE_2048-NEXT:    ld1d { z1.d }, p0/z, [x1]
; VBITS_GE_2048-NEXT:    whilels p1.d, xzr, x8
; VBITS_GE_2048-NEXT:    lastb d0, p1, z0.d
; VBITS_GE_2048-NEXT:    insr z1.d, d0
; VBITS_GE_2048-NEXT:    st1d { z1.d }, p0, [x0]
; VBITS_GE_2048-NEXT:    ret
  %op1 = load <32 x double>, <32 x double>* %a
  %op2 = load <32 x double>, <32 x double>* %b
  %ret = shufflevector <32 x double> %op1, <32 x double> %op2, <32 x i32> <i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38,
                                                                           i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46,
                                                                           i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54,
                                                                           i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62>
  store <32 x double> %ret, <32 x double>* %a
  ret void
}

define void @shuffle_ext_byone_reverse(<4 x double>* %a, <4 x double>* %b) #0 {
; CHECK-LABEL: shuffle_ext_byone_reverse:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ld1d { z1.d }, p0/z, [x1]
; CHECK-NEXT:    mov z1.d, z1.d[3]
; CHECK-NEXT:    insr z0.d, d1
; CHECK-NEXT:    st1d { z0.d }, p0, [x0]
; CHECK-NEXT:    ret
  %op1 = load <4 x double>, <4 x double>* %a
  %op2 = load <4 x double>, <4 x double>* %b
  %ret = shufflevector <4 x double> %op1, <4 x double> %op2, <4 x i32> <i32 7, i32 0, i32 1, i32 2>
  store <4 x double> %ret, <4 x double>* %a
  ret void
}

define void @shuffle_ext_invalid(<4 x double>* %a, <4 x double>* %b) #0 {
; CHECK-LABEL: shuffle_ext_invalid:
; CHECK:       // %bb.0:
; CHECK-NEXT:    stp x29, x30, [sp, #-16]! // 16-byte Folded Spill
; CHECK-NEXT:    sub x9, sp, #48
; CHECK-NEXT:    mov x29, sp
; CHECK-NEXT:    and sp, x9, #0xffffffffffffffe0
; CHECK-NEXT:    .cfi_def_cfa w29, 16
; CHECK-NEXT:    .cfi_offset w30, -8
; CHECK-NEXT:    .cfi_offset w29, -16
; CHECK-NEXT:    ptrue p0.d, vl4
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ld1d { z1.d }, p0/z, [x1]
; CHECK-NEXT:    mov z2.d, z1.d[1]
; CHECK-NEXT:    stp d1, d2, [sp, #16]
; CHECK-NEXT:    mov z1.d, z0.d[3]
; CHECK-NEXT:    mov z0.d, z0.d[2]
; CHECK-NEXT:    stp d0, d1, [sp]
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [sp]
; CHECK-NEXT:    st1d { z0.d }, p0, [x0]
; CHECK-NEXT:    mov sp, x29
; CHECK-NEXT:    ldp x29, x30, [sp], #16 // 16-byte Folded Reload
; CHECK-NEXT:    ret
  %op1 = load <4 x double>, <4 x double>* %a
  %op2 = load <4 x double>, <4 x double>* %b
  %ret = shufflevector <4 x double> %op1, <4 x double> %op2, <4 x i32> <i32 2, i32 3, i32 4, i32 5>
  store <4 x double> %ret, <4 x double>* %a
  ret void
}

attributes #0 = { "target-features"="+sve" }
