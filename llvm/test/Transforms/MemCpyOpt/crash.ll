; RUN: opt < %s -memcpyopt -disable-output
; PR4882

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "armv7-eabi"

%struct.qw = type { [4 x float] }
%struct.bar = type { %struct.qw, %struct.qw, %struct.qw, %struct.qw, %struct.qw, float, float}

define arm_aapcs_vfpcc void @test1(%struct.bar* %this) {
entry:
  %0 = getelementptr inbounds %struct.bar* %this, i32 0, i32 0, i32 0, i32 0
  store float 0.000000e+00, float* %0, align 4
  %1 = getelementptr inbounds %struct.bar* %this, i32 0, i32 0, i32 0, i32 1
  store float 0.000000e+00, float* %1, align 4
  %2 = getelementptr inbounds %struct.bar* %this, i32 0, i32 0, i32 0, i32 2
  store float 0.000000e+00, float* %2, align 4
  %3 = getelementptr inbounds %struct.bar* %this, i32 0, i32 0, i32 0, i32 3
  store float 0.000000e+00, float* %3, align 4
  %4 = getelementptr inbounds %struct.bar* %this, i32 0, i32 1, i32 0, i32 0
  store float 0.000000e+00, float* %4, align 4
  %5 = getelementptr inbounds %struct.bar* %this, i32 0, i32 1, i32 0, i32 1
  store float 0.000000e+00, float* %5, align 4
  %6 = getelementptr inbounds %struct.bar* %this, i32 0, i32 1, i32 0, i32 2
  store float 0.000000e+00, float* %6, align 4
  %7 = getelementptr inbounds %struct.bar* %this, i32 0, i32 1, i32 0, i32 3
  store float 0.000000e+00, float* %7, align 4
  %8 = getelementptr inbounds %struct.bar* %this, i32 0, i32 3, i32 0, i32 1
  store float 0.000000e+00, float* %8, align 4
  %9 = getelementptr inbounds %struct.bar* %this, i32 0, i32 3, i32 0, i32 2
  store float 0.000000e+00, float* %9, align 4
  %10 = getelementptr inbounds %struct.bar* %this, i32 0, i32 3, i32 0, i32 3
  store float 0.000000e+00, float* %10, align 4
  %11 = getelementptr inbounds %struct.bar* %this, i32 0, i32 4, i32 0, i32 0
  store float 0.000000e+00, float* %11, align 4
  %12 = getelementptr inbounds %struct.bar* %this, i32 0, i32 4, i32 0, i32 1
  store float 0.000000e+00, float* %12, align 4
  %13 = getelementptr inbounds %struct.bar* %this, i32 0, i32 4, i32 0, i32 2
  store float 0.000000e+00, float* %13, align 4
  %14 = getelementptr inbounds %struct.bar* %this, i32 0, i32 4, i32 0, i32 3
  store float 0.000000e+00, float* %14, align 4
  %15 = getelementptr inbounds %struct.bar* %this, i32 0, i32 5
  store float 0.000000e+00, float* %15, align 4
  unreachable
}
