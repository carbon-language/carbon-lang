; RUN: llc < %s -mtriple=x86_64-apple-darwin -march=x86 -mcpu=corei7-avx | FileCheck %s


define <2 x double> @floor_v2f64(<2 x double> %p)
{
  ; CHECK: floor_v2f64
  ; CHECK: vroundpd
  %t = call <2 x double> @llvm.floor.v2f64(<2 x double> %p)
  ret <2 x double> %t
}
declare <2 x double> @llvm.floor.v2f64(<2 x double> %p)

define <4 x float> @floor_v4f32(<4 x float> %p)
{
  ; CHECK: floor_v4f32
  ; CHECK: vroundps
  %t = call <4 x float> @llvm.floor.v4f32(<4 x float> %p)
  ret <4 x float> %t
}
declare <4 x float> @llvm.floor.v4f32(<4 x float> %p)

define <4 x double> @floor_v4f64(<4 x double> %p)
{
  ; CHECK: floor_v4f64
  ; CHECK: vroundpd
  %t = call <4 x double> @llvm.floor.v4f64(<4 x double> %p)
  ret <4 x double> %t
}
declare <4 x double> @llvm.floor.v4f64(<4 x double> %p)

define <8 x float> @floor_v8f32(<8 x float> %p)
{
  ; CHECK: floor_v8f32
  ; CHECK: vroundps
  %t = call <8 x float> @llvm.floor.v8f32(<8 x float> %p)
  ret <8 x float> %t
}
declare <8 x float> @llvm.floor.v8f32(<8 x float> %p)

define <2 x double> @ceil_v2f64(<2 x double> %p)
{
  ; CHECK: ceil_v2f64
  ; CHECK: vroundpd
  %t = call <2 x double> @llvm.ceil.v2f64(<2 x double> %p)
  ret <2 x double> %t
}
declare <2 x double> @llvm.ceil.v2f64(<2 x double> %p)

define <4 x float> @ceil_v4f32(<4 x float> %p)
{
  ; CHECK: ceil_v4f32
  ; CHECK: vroundps
  %t = call <4 x float> @llvm.ceil.v4f32(<4 x float> %p)
  ret <4 x float> %t
}
declare <4 x float> @llvm.ceil.v4f32(<4 x float> %p)

define <4 x double> @ceil_v4f64(<4 x double> %p)
{
  ; CHECK: ceil_v4f64
  ; CHECK: vroundpd
  %t = call <4 x double> @llvm.ceil.v4f64(<4 x double> %p)
  ret <4 x double> %t
}
declare <4 x double> @llvm.ceil.v4f64(<4 x double> %p)

define <8 x float> @ceil_v8f32(<8 x float> %p)
{
  ; CHECK: ceil_v8f32
  ; CHECK: vroundps
  %t = call <8 x float> @llvm.ceil.v8f32(<8 x float> %p)
  ret <8 x float> %t
}
declare <8 x float> @llvm.ceil.v8f32(<8 x float> %p)

define <2 x double> @trunc_v2f64(<2 x double> %p)
{
  ; CHECK: trunc_v2f64
  ; CHECK: vroundpd
  %t = call <2 x double> @llvm.trunc.v2f64(<2 x double> %p)
  ret <2 x double> %t
}
declare <2 x double> @llvm.trunc.v2f64(<2 x double> %p)

define <4 x float> @trunc_v4f32(<4 x float> %p)
{
  ; CHECK: trunc_v4f32
  ; CHECK: vroundps
  %t = call <4 x float> @llvm.trunc.v4f32(<4 x float> %p)
  ret <4 x float> %t
}
declare <4 x float> @llvm.trunc.v4f32(<4 x float> %p)

define <4 x double> @trunc_v4f64(<4 x double> %p)
{
  ; CHECK: trunc_v4f64
  ; CHECK: vroundpd
  %t = call <4 x double> @llvm.trunc.v4f64(<4 x double> %p)
  ret <4 x double> %t
}
declare <4 x double> @llvm.trunc.v4f64(<4 x double> %p)

define <8 x float> @trunc_v8f32(<8 x float> %p)
{
  ; CHECK: trunc_v8f32
  ; CHECK: vroundps
  %t = call <8 x float> @llvm.trunc.v8f32(<8 x float> %p)
  ret <8 x float> %t
}
declare <8 x float> @llvm.trunc.v8f32(<8 x float> %p)

define <2 x double> @rint_v2f64(<2 x double> %p)
{
  ; CHECK: rint_v2f64
  ; CHECK: vroundpd
  %t = call <2 x double> @llvm.rint.v2f64(<2 x double> %p)
  ret <2 x double> %t
}
declare <2 x double> @llvm.rint.v2f64(<2 x double> %p)

define <4 x float> @rint_v4f32(<4 x float> %p)
{
  ; CHECK: rint_v4f32
  ; CHECK: vroundps
  %t = call <4 x float> @llvm.rint.v4f32(<4 x float> %p)
  ret <4 x float> %t
}
declare <4 x float> @llvm.rint.v4f32(<4 x float> %p)

define <4 x double> @rint_v4f64(<4 x double> %p)
{
  ; CHECK: rint_v4f64
  ; CHECK: vroundpd
  %t = call <4 x double> @llvm.rint.v4f64(<4 x double> %p)
  ret <4 x double> %t
}
declare <4 x double> @llvm.rint.v4f64(<4 x double> %p)

define <8 x float> @rint_v8f32(<8 x float> %p)
{
  ; CHECK: rint_v8f32
  ; CHECK: vroundps
  %t = call <8 x float> @llvm.rint.v8f32(<8 x float> %p)
  ret <8 x float> %t
}
declare <8 x float> @llvm.rint.v8f32(<8 x float> %p)

define <2 x double> @nearbyint_v2f64(<2 x double> %p)
{
  ; CHECK: nearbyint_v2f64
  ; CHECK: vroundpd
  %t = call <2 x double> @llvm.nearbyint.v2f64(<2 x double> %p)
  ret <2 x double> %t
}
declare <2 x double> @llvm.nearbyint.v2f64(<2 x double> %p)

define <4 x float> @nearbyint_v4f32(<4 x float> %p)
{
  ; CHECK: nearbyint_v4f32
  ; CHECK: vroundps
  %t = call <4 x float> @llvm.nearbyint.v4f32(<4 x float> %p)
  ret <4 x float> %t
}
declare <4 x float> @llvm.nearbyint.v4f32(<4 x float> %p)

define <4 x double> @nearbyint_v4f64(<4 x double> %p)
{
  ; CHECK: nearbyint_v4f64
  ; CHECK: vroundpd
  %t = call <4 x double> @llvm.nearbyint.v4f64(<4 x double> %p)
  ret <4 x double> %t
}
declare <4 x double> @llvm.nearbyint.v4f64(<4 x double> %p)

define <8 x float> @nearbyint_v8f32(<8 x float> %p)
{
  ; CHECK: nearbyint_v8f32
  ; CHECK: vroundps
  %t = call <8 x float> @llvm.nearbyint.v8f32(<8 x float> %p)
  ret <8 x float> %t
}
declare <8 x float> @llvm.nearbyint.v8f32(<8 x float> %p)
