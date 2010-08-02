;RUN: llc --march=cellspu %s -o - | FileCheck %s
%vec = type <2 x i32>

define %vec @test_ret(%vec %param)
{
;CHECK:	bi	$lr
  ret %vec %param
}

define %vec @test_add(%vec %param)
{
;CHECK: a $3, $3, $3
  %1 = add %vec %param, %param
;CHECK: bi $lr
  ret %vec %1
}

define %vec @test_sub(%vec %param)
{
;CHECK: sf $3, $4, $3
  %1 = sub %vec %param, <i32 1, i32 1>

;CHECK: bi $lr
  ret %vec %1
}

define %vec @test_mul(%vec %param)
{
;CHECK: mpyu
;CHECK: mpyh
;CHECK: a
;CHECK: a $3
  %1 = mul %vec %param, %param

;CHECK: bi $lr
  ret %vec %1
}

define <2 x i32> @test_splat(i32 %param ) {
;TODO insertelement transforms to a PREFSLOT2VEC, that trasforms to the 
;     somewhat redundant: 
;CHECK-NOT or $3, $3, $3
;CHECK: lqa
;CHECK: shufb
  %sv = insertelement <1 x i32> undef, i32 %param, i32 0 
  %rv = shufflevector <1 x i32> %sv, <1 x i32> undef, <2 x i32> zeroinitializer 
;CHECK: bi $lr
  ret <2 x i32> %rv
}

define i32 @test_extract() {
;CHECK: shufb $3
  %rv = extractelement <2 x i32> zeroinitializer, i32 undef ; <i32> [#uses=1]
;CHECK: bi $lr
  ret i32 %rv
}

