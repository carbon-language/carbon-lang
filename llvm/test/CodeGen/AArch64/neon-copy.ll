; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon -fp-contract=fast | FileCheck %s


define <16 x i8> @ins16bw(<16 x i8> %tmp1, i8 %tmp2) {
;CHECK: ins {{v[0-31]+}}.b[15], {{w[0-31]+}}
  %tmp3 = insertelement <16 x i8> %tmp1, i8 %tmp2, i32 15
  ret <16 x i8> %tmp3
}

define <8 x i16> @ins8hw(<8 x i16> %tmp1, i16 %tmp2) {
;CHECK: ins {{v[0-31]+}}.h[6], {{w[0-31]+}}
  %tmp3 = insertelement <8 x i16> %tmp1, i16 %tmp2, i32 6
  ret <8 x i16> %tmp3
}

define <4 x i32> @ins4sw(<4 x i32> %tmp1, i32 %tmp2) {
;CHECK: ins {{v[0-31]+}}.s[2], {{w[0-31]+}}
  %tmp3 = insertelement <4 x i32> %tmp1, i32 %tmp2, i32 2
  ret <4 x i32> %tmp3
}

define <2 x i64> @ins2dw(<2 x i64> %tmp1, i64 %tmp2) {
;CHECK: ins {{v[0-31]+}}.d[1], {{x[0-31]+}}
  %tmp3 = insertelement <2 x i64> %tmp1, i64 %tmp2, i32 1
  ret <2 x i64> %tmp3
}

define <8 x i8> @ins8bw(<8 x i8> %tmp1, i8 %tmp2) {
;CHECK: ins {{v[0-31]+}}.b[5], {{w[0-31]+}}
  %tmp3 = insertelement <8 x i8> %tmp1, i8 %tmp2, i32 5
  ret <8 x i8> %tmp3
}

define <4 x i16> @ins4hw(<4 x i16> %tmp1, i16 %tmp2) {
;CHECK: ins {{v[0-31]+}}.h[3], {{w[0-31]+}}
  %tmp3 = insertelement <4 x i16> %tmp1, i16 %tmp2, i32 3
  ret <4 x i16> %tmp3
}

define <2 x i32> @ins2sw(<2 x i32> %tmp1, i32 %tmp2) {
;CHECK: ins {{v[0-31]+}}.s[1], {{w[0-31]+}}
  %tmp3 = insertelement <2 x i32> %tmp1, i32 %tmp2, i32 1
  ret <2 x i32> %tmp3
}

define <16 x i8> @ins16b16(<16 x i8> %tmp1, <16 x i8> %tmp2) {
;CHECK: ins {{v[0-31]+}}.b[15], {{v[0-31]+}}.b[2]
  %tmp3 = extractelement <16 x i8> %tmp1, i32 2
  %tmp4 = insertelement <16 x i8> %tmp2, i8 %tmp3, i32 15
  ret <16 x i8> %tmp4
}

define <8 x i16> @ins8h8(<8 x i16> %tmp1, <8 x i16> %tmp2) {
;CHECK: ins {{v[0-31]+}}.h[7], {{v[0-31]+}}.h[2]
  %tmp3 = extractelement <8 x i16> %tmp1, i32 2
  %tmp4 = insertelement <8 x i16> %tmp2, i16 %tmp3, i32 7
  ret <8 x i16> %tmp4
}

define <4 x i32> @ins4s4(<4 x i32> %tmp1, <4 x i32> %tmp2) {
;CHECK: ins {{v[0-31]+}}.s[1], {{v[0-31]+}}.s[2]
  %tmp3 = extractelement <4 x i32> %tmp1, i32 2
  %tmp4 = insertelement <4 x i32> %tmp2, i32 %tmp3, i32 1
  ret <4 x i32> %tmp4
}

define <2 x i64> @ins2d2(<2 x i64> %tmp1, <2 x i64> %tmp2) {
;CHECK: ins {{v[0-31]+}}.d[1], {{v[0-31]+}}.d[0]
  %tmp3 = extractelement <2 x i64> %tmp1, i32 0
  %tmp4 = insertelement <2 x i64> %tmp2, i64 %tmp3, i32 1
  ret <2 x i64> %tmp4
}

define <8 x i8> @ins8b8(<8 x i8> %tmp1, <8 x i8> %tmp2) {
;CHECK: ins {{v[0-31]+}}.b[4], {{v[0-31]+}}.b[2]
  %tmp3 = extractelement <8 x i8> %tmp1, i32 2
  %tmp4 = insertelement <8 x i8> %tmp2, i8 %tmp3, i32 4
  ret <8 x i8> %tmp4
}

define <4 x i16> @ins4h4(<4 x i16> %tmp1, <4 x i16> %tmp2) {
;CHECK: ins {{v[0-31]+}}.h[3], {{v[0-31]+}}.h[2]
  %tmp3 = extractelement <4 x i16> %tmp1, i32 2
  %tmp4 = insertelement <4 x i16> %tmp2, i16 %tmp3, i32 3
  ret <4 x i16> %tmp4
}

define <2 x i32> @ins2s2(<2 x i32> %tmp1, <2 x i32> %tmp2) {
;CHECK: ins {{v[0-31]+}}.s[1], {{v[0-31]+}}.s[0]
  %tmp3 = extractelement <2 x i32> %tmp1, i32 0
  %tmp4 = insertelement <2 x i32> %tmp2, i32 %tmp3, i32 1
  ret <2 x i32> %tmp4
}

define <1 x i64> @ins1d1(<1 x i64> %tmp1, <1 x i64> %tmp2) {
;CHECK: ins {{v[0-31]+}}.d[0], {{v[0-31]+}}.d[0]
  %tmp3 = extractelement <1 x i64> %tmp1, i32 0
  %tmp4 = insertelement <1 x i64> %tmp2, i64 %tmp3, i32 0
  ret <1 x i64> %tmp4
}

define i32 @umovw16b(<16 x i8> %tmp1) {
;CHECK: umov {{w[0-31]+}}, {{v[0-31]+}}.b[8]
  %tmp3 = extractelement <16 x i8> %tmp1, i32 8
  %tmp4 = zext i8 %tmp3 to i32
  ret i32 %tmp4
}

define i32 @umovw8h(<8 x i16> %tmp1) {
;CHECK: umov {{w[0-31]+}}, {{v[0-31]+}}.h[2]
  %tmp3 = extractelement <8 x i16> %tmp1, i32 2
  %tmp4 = zext i16 %tmp3 to i32
  ret i32 %tmp4
}

define i32 @umovw4s(<4 x i32> %tmp1) {
;CHECK: umov {{w[0-31]+}}, {{v[0-31]+}}.s[2]
  %tmp3 = extractelement <4 x i32> %tmp1, i32 2
  ret i32 %tmp3
}

define i64 @umovx2d(<2 x i64> %tmp1) {
;CHECK: umov {{x[0-31]+}}, {{v[0-31]+}}.d[0]
  %tmp3 = extractelement <2 x i64> %tmp1, i32 0
  ret i64 %tmp3
}

define i32 @umovw8b(<8 x i8> %tmp1) {
;CHECK: umov {{w[0-31]+}}, {{v[0-31]+}}.b[7]
  %tmp3 = extractelement <8 x i8> %tmp1, i32 7
  %tmp4 = zext i8 %tmp3 to i32
  ret i32 %tmp4
}

define i32 @umovw4h(<4 x i16> %tmp1) {
;CHECK: umov {{w[0-31]+}}, {{v[0-31]+}}.h[2]
  %tmp3 = extractelement <4 x i16> %tmp1, i32 2
  %tmp4 = zext i16 %tmp3 to i32
  ret i32 %tmp4
}

define i32 @umovw2s(<2 x i32> %tmp1) {
;CHECK: umov {{w[0-31]+}}, {{v[0-31]+}}.s[1]
  %tmp3 = extractelement <2 x i32> %tmp1, i32 1
  ret i32 %tmp3
}

define i64 @umovx1d(<1 x i64> %tmp1) {
;CHECK: fmov {{x[0-31]+}}, {{d[0-31]+}}
  %tmp3 = extractelement <1 x i64> %tmp1, i32 0
  ret i64 %tmp3
}

define i32 @smovw16b(<16 x i8> %tmp1) {
;CHECK: smov {{w[0-31]+}}, {{v[0-31]+}}.b[8]
  %tmp3 = extractelement <16 x i8> %tmp1, i32 8
  %tmp4 = sext i8 %tmp3 to i32
  %tmp5 = add i32 5, %tmp4
  ret i32 %tmp5
}

define i32 @smovw8h(<8 x i16> %tmp1) {
;CHECK: smov {{w[0-31]+}}, {{v[0-31]+}}.h[2]
  %tmp3 = extractelement <8 x i16> %tmp1, i32 2
  %tmp4 = sext i16 %tmp3 to i32
  %tmp5 = add i32 5, %tmp4
  ret i32 %tmp5
}

define i32 @smovx16b(<16 x i8> %tmp1) {
;CHECK: smov {{x[0-31]+}}, {{v[0-31]+}}.b[8]
  %tmp3 = extractelement <16 x i8> %tmp1, i32 8
  %tmp4 = sext i8 %tmp3 to i32
  ret i32 %tmp4
}

define i32 @smovx8h(<8 x i16> %tmp1) {
;CHECK: smov {{x[0-31]+}}, {{v[0-31]+}}.h[2]
  %tmp3 = extractelement <8 x i16> %tmp1, i32 2
  %tmp4 = sext i16 %tmp3 to i32
  ret i32 %tmp4
}

define i64 @smovx4s(<4 x i32> %tmp1) {
;CHECK: smov {{x[0-31]+}}, {{v[0-31]+}}.s[2]
  %tmp3 = extractelement <4 x i32> %tmp1, i32 2
  %tmp4 = sext i32 %tmp3 to i64
  ret i64 %tmp4
}

define i32 @smovw8b(<8 x i8> %tmp1) {
;CHECK: smov {{w[0-31]+}}, {{v[0-31]+}}.b[4]
  %tmp3 = extractelement <8 x i8> %tmp1, i32 4
  %tmp4 = sext i8 %tmp3 to i32
  %tmp5 = add i32 5, %tmp4
  ret i32 %tmp5
}

define i32 @smovw4h(<4 x i16> %tmp1) {
;CHECK: smov {{w[0-31]+}}, {{v[0-31]+}}.h[2]
  %tmp3 = extractelement <4 x i16> %tmp1, i32 2
  %tmp4 = sext i16 %tmp3 to i32
  %tmp5 = add i32 5, %tmp4
  ret i32 %tmp5
}

define i32 @smovx8b(<8 x i8> %tmp1) {
;CHECK: smov {{x[0-31]+}}, {{v[0-31]+}}.b[6]
  %tmp3 = extractelement <8 x i8> %tmp1, i32 6
  %tmp4 = sext i8 %tmp3 to i32
  ret i32 %tmp4
}

define i32 @smovx4h(<4 x i16> %tmp1) {
;CHECK: smov {{x[0-31]+}}, {{v[0-31]+}}.h[2]
  %tmp3 = extractelement <4 x i16> %tmp1, i32 2
  %tmp4 = sext i16 %tmp3 to i32
  ret i32 %tmp4
}

define i64 @smovx2s(<2 x i32> %tmp1) {
;CHECK: smov {{x[0-31]+}}, {{v[0-31]+}}.s[1]
  %tmp3 = extractelement <2 x i32> %tmp1, i32 1
  %tmp4 = sext i32 %tmp3 to i64
  ret i64 %tmp4
}






