; RUN: llc --march=cellspu < %s | FileCheck %s

define <4 x float> @shuffle(<4 x float> %param1, <4 x float> %param2) {
  ; CHECK: cwd {{\$.}}, 0($sp)
  ; CHECK: shufb {{\$., \$4, \$3, \$.}}
  %val= shufflevector <4 x float> %param1, <4 x float> %param2, <4 x i32> <i32 4,i32 1,i32 2,i32 3>
  ret <4 x float> %val
  
}
 
