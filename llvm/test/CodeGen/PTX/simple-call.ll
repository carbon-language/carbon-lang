; RUN: llc < %s -march=ptx32 -mattr=sm20 | FileCheck %s
; XFAIL: *

define ptx_device void @test_add(float %x, float %y) {
; CHECK: ret;
	%z = fadd float %x, %y
	ret void
}

define ptx_device float @test_call(float %x, float %y) {
  %a = fadd float %x, %y
; CHECK: call.uni test_add, (__ret_{{[0-9]+}}, __ret_{{[0-9]+}});
  call void @test_add(float %a, float %y)
  ret float %a
}
