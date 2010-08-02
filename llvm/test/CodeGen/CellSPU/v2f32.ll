;RUN: llc --march=cellspu %s -o - | FileCheck %s
%vec = type <2 x float>

define %vec @test_ret(%vec %param)
{
;CHECK: bi $lr
 ret %vec %param
}

define %vec @test_add(%vec %param)
{
;CHECK: fa $3, $3, $3
 %1 = fadd %vec %param, %param
;CHECK: bi $lr
 ret %vec %1
}

define %vec @test_sub(%vec %param)
{
;CHECK: fs $3, $3, $3
 %1 = fsub %vec %param, %param

;CHECK: bi $lr
 ret %vec %1
}

define %vec @test_mul(%vec %param)
{
;CHECK: fm $3, $3, $3
 %1 = fmul %vec %param, %param

;CHECK: bi $lr
 ret %vec %1
}

define %vec @test_splat(float %param ) {
;CHECK: lqa
;CHECK: shufb
  %sv = insertelement <1 x float> undef, float %param, i32 0 
  %rv = shufflevector <1 x float> %sv, <1 x float> undef, <2 x i32> zeroinitializer 
;CHECK: bi $lr
  ret %vec %rv
}


