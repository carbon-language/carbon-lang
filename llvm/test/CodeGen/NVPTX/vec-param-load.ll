; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"


define <16 x float> @foo(<16 x float> %a) {
; Make sure we index into vectors properly
; CHECK: ld.param.v4.f32         {%f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}}, [foo_param_0];
; CHECK: ld.param.v4.f32         {%f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}}, [foo_param_0+16];
; CHECK: ld.param.v4.f32         {%f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}}, [foo_param_0+32];
; CHECK: ld.param.v4.f32         {%f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}}, [foo_param_0+48];
  ret <16 x float> %a
}
