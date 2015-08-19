; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 | FileCheck %s
; RUN: llc < %s -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 | FileCheck %s -check-prefix=CHECK-LE

; The build[csilf] functions simply test the scalar_to_vector handling with
; direct moves. This corresponds to the "insertelement" instruction. Subsequent
; to this, there will be a splat corresponding to the shufflevector.

; Function Attrs: nounwind
define <16 x i8> @buildc(i8 zeroext %a) {
entry:
  %a.addr = alloca i8, align 1
  store i8 %a, i8* %a.addr, align 1
  %0 = load i8, i8* %a.addr, align 1
  %splat.splatinsert = insertelement <16 x i8> undef, i8 %0, i32 0
  %splat.splat = shufflevector <16 x i8> %splat.splatinsert, <16 x i8> undef, <16 x i32> zeroinitializer
  ret <16 x i8> %splat.splat
; CHECK: sldi [[REG1:[0-9]+]], 3, 56
; CHECK: mtvsrd {{[0-9]+}}, [[REG1]]
; CHECK-LE: mtvsrd [[REG1:[0-9]+]], 3
; CHECK-LE: xxswapd {{[0-9]+}}, [[REG1]]
}

; Function Attrs: nounwind
define <8 x i16> @builds(i16 zeroext %a) {
entry:
  %a.addr = alloca i16, align 2
  store i16 %a, i16* %a.addr, align 2
  %0 = load i16, i16* %a.addr, align 2
  %splat.splatinsert = insertelement <8 x i16> undef, i16 %0, i32 0
  %splat.splat = shufflevector <8 x i16> %splat.splatinsert, <8 x i16> undef, <8 x i32> zeroinitializer
  ret <8 x i16> %splat.splat
; CHECK: sldi [[REG1:[0-9]+]], 3, 48
; CHECK: mtvsrd {{[0-9]+}}, [[REG1]]
; CHECK-LE: mtvsrd [[REG1:[0-9]+]], 3
; CHECK-LE: xxswapd {{[0-9]+}}, [[REG1]]
}

; Function Attrs: nounwind
define <4 x i32> @buildi(i32 zeroext %a) {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  %0 = load i32, i32* %a.addr, align 4
  %splat.splatinsert = insertelement <4 x i32> undef, i32 %0, i32 0
  %splat.splat = shufflevector <4 x i32> %splat.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %splat.splat
; CHECK: sldi [[REG1:[0-9]+]], 3, 32
; CHECK: mtvsrd {{[0-9]+}}, [[REG1]]
; CHECK-LE: mtvsrd [[REG1:[0-9]+]], 3
; CHECK-LE: xxswapd {{[0-9]+}}, [[REG1]]
}

; Function Attrs: nounwind
define <2 x i64> @buildl(i64 %a) {
entry:
  %a.addr = alloca i64, align 8
  store i64 %a, i64* %a.addr, align 8
  %0 = load i64, i64* %a.addr, align 8
  %splat.splatinsert = insertelement <2 x i64> undef, i64 %0, i32 0
  %splat.splat = shufflevector <2 x i64> %splat.splatinsert, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %splat.splat
; FIXME-CHECK: mtvsrd {{[0-9]+}}, 3
; FIXME-CHECK-LE: mtvsrd [[REG1:[0-9]+]], 3
; FIXME-CHECK-LE: xxswapd {{[0-9]+}}, [[REG1]]
}

; Function Attrs: nounwind
define <4 x float> @buildf(float %a) {
entry:
  %a.addr = alloca float, align 4
  store float %a, float* %a.addr, align 4
  %0 = load float, float* %a.addr, align 4
  %splat.splatinsert = insertelement <4 x float> undef, float %0, i32 0
  %splat.splat = shufflevector <4 x float> %splat.splatinsert, <4 x float> undef, <4 x i32> zeroinitializer
  ret <4 x float> %splat.splat
; CHECK: xscvdpspn {{[0-9]+}}, 1
; CHECK-LE: xscvdpspn [[REG1:[0-9]+]], 1
; CHECK-LE: xxsldwi {{[0-9]+}}, [[REG1]], [[REG1]], 1
}
