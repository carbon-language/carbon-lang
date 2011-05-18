; RUN: llc < %s -march=ptx32 -mattr=+ptx20,+sm20 | grep "mad"
; RUN: llc < %s -march=ptx32 -mattr=+ptx20,+sm20,+no-fma | grep -v "mad"

define ptx_device float @test_mul_add_f(float %x, float %y, float %z) {
entry:
  %a = fmul float %x, %y
  %b = fadd float %a, %z
  ret float %b
}

define ptx_device double @test_mul_add_d(double %x, double %y, double %z) {
entry:
  %a = fmul double %x, %y
  %b = fadd double %a, %z
  ret double %b
}
