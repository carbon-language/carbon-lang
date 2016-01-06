; Round trip constant sequences through bitcode
; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; CHECK: @array.i8  = constant [3 x i8] c"\00\01\00"
@array.i8  = constant [3 x i8] [i8 -0, i8 1, i8 0]
; CHECK: @array.i16 = constant [3 x i16] [i16 0, i16 1, i16 0]
@array.i16 = constant [3 x i16] [i16 -0, i16 1, i16 0]
; CHECK: @array.i32 = constant [3 x i32] [i32 0, i32 1, i32 0]
@array.i32 = constant [3 x i32] [i32 -0, i32 1, i32 0]
; CHECK: @array.i64 = constant [3 x i64] [i64 0, i64 1, i64 0]
@array.i64 = constant [3 x i64] [i64 -0, i64 1, i64 0]
; CHECK: @array.f16 = constant [3 x half] [half 0xH8000, half 0xH3C00, half 0xH0000]
@array.f16 = constant [3 x half] [half -0.0, half 1.0, half 0.0]
; CHECK: @array.f32 = constant [3 x float] [float -0.000000e+00, float 1.000000e+00, float 0.000000e+00]
@array.f32 = constant [3 x float] [float -0.0, float 1.0, float 0.0]
; CHECK: @array.f64 = constant [3 x double] [double -0.000000e+00, double 1.000000e+00, double 0.000000e+00]
@array.f64 = constant [3 x double] [double -0.0, double 1.0, double 0.0]

; CHECK: @vector.i8  = constant <3 x i8>  <i8 0, i8 1, i8 0>
@vector.i8  = constant <3 x i8>  <i8 -0, i8 1, i8 0>
; CHECK: @vector.i16 = constant <3 x i16> <i16 0, i16 1, i16 0>
@vector.i16 = constant <3 x i16> <i16 -0, i16 1, i16 0>
; CHECK: @vector.i32 = constant <3 x i32> <i32 0, i32 1, i32 0>
@vector.i32 = constant <3 x i32> <i32 -0, i32 1, i32 0>
; CHECK: @vector.i64 = constant <3 x i64> <i64 0, i64 1, i64 0>
@vector.i64 = constant <3 x i64> <i64 -0, i64 1, i64 0>
; CHECK: @vector.f16 = constant <3 x half> <half 0xH8000, half 0xH3C00, half 0xH0000>
@vector.f16 = constant <3 x half> <half -0.0, half 1.0, half 0.0>
; CHECK: @vector.f32 = constant <3 x float> <float -0.000000e+00, float 1.000000e+00, float 0.000000e+00>
@vector.f32 = constant <3 x float> <float -0.0, float 1.0, float 0.0>
; CHECK: @vector.f64 = constant <3 x double> <double -0.000000e+00, double 1.000000e+00, double 0.000000e+00>
@vector.f64 = constant <3 x double> <double -0.0, double 1.0, double 0.0>
