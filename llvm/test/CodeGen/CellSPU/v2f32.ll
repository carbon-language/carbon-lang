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

