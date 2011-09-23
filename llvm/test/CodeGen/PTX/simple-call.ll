; RUN: llc < %s -march=ptx32 -mattr=sm20 | FileCheck %s

define ptx_device void @test_add(float %x, float %y) {
; CHECK: ret;
	%z = fadd float %x, %y
	ret void
}

define ptx_device float @test_call(float %x, float %y) {
  %a = fadd float %x, %y
; CHECK: call.uni test_add, (__localparam_{{[0-9]+}}, __localparam_{{[0-9]+}});
  call void @test_add(float %a, float %y)
  ret float %a
}

define ptx_device float @test_compute(float %x, float %y) {
; CHECK: ret;
  %z = fadd float %x, %y
  ret float %z
}

define ptx_device float @test_call_compute(float %x, float %y) {
; CHECK: call.uni (__localparam_{{[0-9]+}}), test_compute, (__localparam_{{[0-9]+}}, __localparam_{{[0-9]+}})
  %z = call float @test_compute(float %x, float %y)
  ret float %z
}

