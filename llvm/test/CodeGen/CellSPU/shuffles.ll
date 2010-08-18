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

define void @test_insert( <2 x float>* %ptr, float %val1, float %val2 ) {
  %sl2_17_tmp1 = insertelement <2 x float> zeroinitializer, float %val1, i32 0
;CHECK:	lqa	$6,
;CHECK:	shufb	$4, $4, $5, $6
  %sl2_17 = insertelement <2 x float> %sl2_17_tmp1, float %val2, i32 1

;CHECK: cdd	$5, 0($3)
;CHECK: lqd	$6, 0($3)
;CHECK: shufb	$4, $4, $6, $5
;CHECK: stqd	$4, 0($3)
;CHECK:	bi	$lr
  store <2 x float> %sl2_17, <2 x float>* %ptr
  ret void 
}

