; RUN: opt -mtriple=x86_64-apple-darwin -mcpu=core2 -cost-model -analyze < %s | FileCheck --check-prefix=SSE2 %s

define <2 x double> @sitofpv2i8v2double(<2 x i8> %a) {
  ; SSE2: sitofpv2i8v2double
  ; SSE2: cost of 20 {{.*}} sitofp
  %1 = sitofp <2 x i8> %a to <2 x double>
  ret <2 x double> %1
}

define <4 x double> @sitofpv4i8v4double(<4 x i8> %a) {
  ; SSE2: sitofpv4i8v4double
  ; SSE2: cost of 40 {{.*}} sitofp
  %1 = sitofp <4 x i8> %a to <4 x double>
  ret <4 x double> %1
}

define <8 x double> @sitofpv8i8v8double(<8 x i8> %a) {
  ; SSE2: sitofpv8i8v8double
  ; SSE2: cost of 80 {{.*}} sitofp
%1 = sitofp <8 x i8> %a to <8 x double>
  ret <8 x double> %1
}

define <16 x double> @sitofpv16i8v16double(<16 x i8> %a) {
  ; SSE2: sitofpv16i8v16double
  ; SSE2: cost of 160 {{.*}} sitofp
  %1 = sitofp <16 x i8> %a to <16 x double>
  ret <16 x double> %1
}

define <32 x double> @sitofpv32i8v32double(<32 x i8> %a) {
  ; SSE2: sitofpv32i8v32double
  ; SSE2: cost of 320 {{.*}} sitofp
  %1 = sitofp <32 x i8> %a to <32 x double>
  ret <32 x double> %1
}

define <2 x double> @sitofpv2i16v2double(<2 x i16> %a) {
  ; SSE2: sitofpv2i16v2double
  ; SSE2: cost of 20 {{.*}} sitofp
  %1 = sitofp <2 x i16> %a to <2 x double>
  ret <2 x double> %1
}

define <4 x double> @sitofpv4i16v4double(<4 x i16> %a) {
  ; SSE2: sitofpv4i16v4double
  ; SSE2: cost of 40 {{.*}} sitofp
  %1 = sitofp <4 x i16> %a to <4 x double>
  ret <4 x double> %1
}

define <8 x double> @sitofpv8i16v8double(<8 x i16> %a) {
  ; SSE2: sitofpv8i16v8double
  ; SSE2: cost of 80 {{.*}} sitofp
  %1 = sitofp <8 x i16> %a to <8 x double>
  ret <8 x double> %1
}

define <16 x double> @sitofpv16i16v16double(<16 x i16> %a) {
  ; SSE2: sitofpv16i16v16double
  ; SSE2: cost of 160 {{.*}} sitofp
  %1 = sitofp <16 x i16> %a to <16 x double>
  ret <16 x double> %1
}

define <32 x double> @sitofpv32i16v32double(<32 x i16> %a) {
  ; SSE2: sitofpv32i16v32double
  ; SSE2: cost of 320 {{.*}} sitofp
  %1 = sitofp <32 x i16> %a to <32 x double>
  ret <32 x double> %1
}

define <2 x double> @sitofpv2i32v2double(<2 x i32> %a) {
  ; SSE2: sitofpv2i32v2double
  ; SSE2: cost of 20 {{.*}} sitofp
  %1 = sitofp <2 x i32> %a to <2 x double>
  ret <2 x double> %1
}

define <4 x double> @sitofpv4i32v4double(<4 x i32> %a) {
  ; SSE2: sitofpv4i32v4double
  ; SSE2: cost of 40 {{.*}} sitofp
  %1 = sitofp <4 x i32> %a to <4 x double>
  ret <4 x double> %1
}

define <8 x double> @sitofpv8i32v8double(<8 x i32> %a) {
  ; SSE2: sitofpv8i32v8double
  ; SSE2: cost of 80 {{.*}} sitofp
  %1 = sitofp <8 x i32> %a to <8 x double>
  ret <8 x double> %1
}

define <16 x double> @sitofpv16i32v16double(<16 x i32> %a) {
  ; SSE2: sitofpv16i32v16double
  ; SSE2: cost of 160 {{.*}} sitofp
  %1 = sitofp <16 x i32> %a to <16 x double>
  ret <16 x double> %1
}

define <32 x double> @sitofpv32i32v32double(<32 x i32> %a) {
  ; SSE2: sitofpv32i32v32double
  ; SSE2: cost of 320 {{.*}} sitofp
  %1 = sitofp <32 x i32> %a to <32 x double>
  ret <32 x double> %1
}

define <2 x double> @sitofpv2i64v2double(<2 x i64> %a) {
  ; SSE2: sitofpv2i64v2double
  ; SSE2: cost of 20 {{.*}} sitofp
  %1 = sitofp <2 x i64> %a to <2 x double>
  ret <2 x double> %1
}

define <4 x double> @sitofpv4i64v4double(<4 x i64> %a) {
  ; SSE2: sitofpv4i64v4double
  ; SSE2: cost of 40 {{.*}} sitofp
  %1 = sitofp <4 x i64> %a to <4 x double>
  ret <4 x double> %1
}

define <8 x double> @sitofpv8i64v8double(<8 x i64> %a) {
  %1 = sitofp <8 x i64> %a to <8 x double>
  ; SSE2: sitofpv8i64v8double
  ; SSE2: cost of 80 {{.*}} sitofp
  ret <8 x double> %1
}

define <16 x double> @sitofpv16i64v16double(<16 x i64> %a) {
  ; SSE2: sitofpv16i64v16double
  ; SSE2: cost of 160 {{.*}} sitofp
  %1 = sitofp <16 x i64> %a to <16 x double>
  ret <16 x double> %1
}

define <32 x double> @sitofpv32i64v32double(<32 x i64> %a) {
  ; SSE2: sitofpv32i64v32double
  ; SSE2: cost of 320 {{.*}} sitofp
  %1 = sitofp <32 x i64> %a to <32 x double>
  ret <32 x double> %1
}

define <2 x float> @sitofpv2i8v2float(<2 x i8> %a) {
  ; SSE2: sitofpv2i8v2float
  ; SSE2: cost of 15 {{.*}} sitofp
  %1 = sitofp <2 x i8> %a to <2 x float>
  ret <2 x float> %1
}

define <4 x float> @sitofpv4i8v4float(<4 x i8> %a) {
  ; SSE2: sitofpv4i8v4float
  ; SSE2: cost of 15 {{.*}} sitofp
  %1 = sitofp <4 x i8> %a to <4 x float>
  ret <4 x float> %1
}

define <8 x float> @sitofpv8i8v8float(<8 x i8> %a) {
  ; SSE2: sitofpv8i8v8float
  ; SSE2: cost of 15 {{.*}} sitofp
  %1 = sitofp <8 x i8> %a to <8 x float>
  ret <8 x float> %1
}

define <16 x float> @sitofpv16i8v16float(<16 x i8> %a) {
  ; SSE2: sitofpv16i8v16float
  ; SSE2: cost of 8 {{.*}} sitofp
  %1 = sitofp <16 x i8> %a to <16 x float>
  ret <16 x float> %1
}

define <32 x float> @sitofpv32i8v32float(<32 x i8> %a) {
  ; SSE2: sitofpv32i8v32float
  ; SSE2: cost of 16 {{.*}} sitofp
  %1 = sitofp <32 x i8> %a to <32 x float>
  ret <32 x float> %1
}

define <2 x float> @sitofpv2i16v2float(<2 x i16> %a) {
  ; SSE2: sitofpv2i16v2float
  ; SSE2: cost of 15 {{.*}} sitofp
  %1 = sitofp <2 x i16> %a to <2 x float>
  ret <2 x float> %1
}

define <4 x float> @sitofpv4i16v4float(<4 x i16> %a) {
  ; SSE2: sitofpv4i16v4float
  ; SSE2: cost of 15 {{.*}} sitofp
  %1 = sitofp <4 x i16> %a to <4 x float>
  ret <4 x float> %1
}

define <8 x float> @sitofpv8i16v8float(<8 x i16> %a) {
  ; SSE2: sitofpv8i16v8float
  ; SSE2: cost of 15 {{.*}} sitofp
  %1 = sitofp <8 x i16> %a to <8 x float>
  ret <8 x float> %1
}

define <16 x float> @sitofpv16i16v16float(<16 x i16> %a) {
  ; SSE2: sitofpv16i16v16float
  ; SSE2: cost of 30 {{.*}} sitofp
  %1 = sitofp <16 x i16> %a to <16 x float>
  ret <16 x float> %1
}

define <32 x float> @sitofpv32i16v32float(<32 x i16> %a) {
  ; SSE2: sitofpv32i16v32float
  ; SSE2: cost of 60 {{.*}} sitofp
  %1 = sitofp <32 x i16> %a to <32 x float>
  ret <32 x float> %1
}

define <2 x float> @sitofpv2i32v2float(<2 x i32> %a) {
  ; SSE2: sitofpv2i32v2float
  ; SSE2: cost of 15 {{.*}} sitofp
  %1 = sitofp <2 x i32> %a to <2 x float>
  ret <2 x float> %1
}

define <4 x float> @sitofpv4i32v4float(<4 x i32> %a) {
  ; SSE2: sitofpv4i32v4float
  ; SSE2: cost of 15 {{.*}} sitofp
  %1 = sitofp <4 x i32> %a to <4 x float>
  ret <4 x float> %1
}

define <8 x float> @sitofpv8i32v8float(<8 x i32> %a) {
  ; SSE2: sitofpv8i32v8float
  ; SSE2: cost of 30 {{.*}} sitofp
  %1 = sitofp <8 x i32> %a to <8 x float>
  ret <8 x float> %1
}

define <16 x float> @sitofpv16i32v16float(<16 x i32> %a) {
  ; SSE2: sitofpv16i32v16float
  ; SSE2: cost of 60 {{.*}} sitofp
  %1 = sitofp <16 x i32> %a to <16 x float>
  ret <16 x float> %1
}

define <32 x float> @sitofpv32i32v32float(<32 x i32> %a) {
  ; SSE2: sitofpv32i32v32float
  ; SSE2: cost of 120 {{.*}} sitofp
  %1 = sitofp <32 x i32> %a to <32 x float>
  ret <32 x float> %1
}

define <2 x float> @sitofpv2i64v2float(<2 x i64> %a) {
  ; SSE2: sitofpv2i64v2float
  ; SSE2: cost of 15 {{.*}} sitofp
  %1 = sitofp <2 x i64> %a to <2 x float>
  ret <2 x float> %1
}

define <4 x float> @sitofpv4i64v4float(<4 x i64> %a) {
  ; SSE2: sitofpv4i64v4float
  ; SSE2: cost of 30 {{.*}} sitofp
  %1 = sitofp <4 x i64> %a to <4 x float>
  ret <4 x float> %1
}

define <8 x float> @sitofpv8i64v8float(<8 x i64> %a) {
  ; SSE2: sitofpv8i64v8float
  ; SSE2: cost of 60 {{.*}} sitofp
  %1 = sitofp <8 x i64> %a to <8 x float>
  ret <8 x float> %1
}

define <16 x float> @sitofpv16i64v16float(<16 x i64> %a) {
  ; SSE2: sitofpv16i64v16float
  ; SSE2: cost of 120 {{.*}} sitofp
  %1 = sitofp <16 x i64> %a to <16 x float>
  ret <16 x float> %1
}

define <32 x float> @sitofpv32i64v32float(<32 x i64> %a) {
  ; SSE2: sitofpv32i64v32float
  ; SSE2: cost of 240 {{.*}} sitofp
  %1 = sitofp <32 x i64> %a to <32 x float>
  ret <32 x float> %1
}
