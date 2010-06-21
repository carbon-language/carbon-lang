; RUN: llc --march=cellspu < %s | FileCheck %s

define <4 x float> @shuffle(<4 x float> %param1, <4 x float> %param2) {
  ; CHECK: cwd {{\$.}}, 0($sp)
  ; CHECK: shufb {{\$., \$4, \$3, \$.}}
  %val= shufflevector <4 x float> %param1, <4 x float> %param2, <4 x i32> <i32 4,i32 1,i32 2,i32 3>
  ret <4 x float> %val
}
 
define <4 x float> @splat(float %param1) {
  ; CHECK: lqa
  ; CHECK: shufb $3
  ; CHECK: bi
  %vec = insertelement <1 x float> undef, float %param1, i32 0
  %val= shufflevector <1 x float> %vec, <1 x float> undef, <4 x i32> <i32 0,i32 0,i32 0,i32 0>
  ret <4 x float> %val  
}

