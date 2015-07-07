; RUN: llc < %s -mtriple=arm64-apple-ios

define void @test(float * %p1, i32 %v1) {
entry:
  %v2 = extractelement <3 x float> <float 0.000000e+00, float 2.000000e+00, float 0.000000e+00>, i32 %v1
  store float %v2, float* %p1, align 4
  ret void
}
