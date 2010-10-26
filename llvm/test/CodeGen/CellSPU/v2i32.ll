;RUN: llc --march=cellspu %s -o - | FileCheck %s
%vec = type <2 x i32>

define %vec @test_ret(%vec %param)
{
;CHECK:	bi	$lr
  ret %vec %param
}

define %vec @test_add(%vec %param)
{
;CHECK: a {{\$.}}, $3, $3
  %1 = add %vec %param, %param
;CHECK: bi $lr
  ret %vec %1
}

define %vec @test_sub(%vec %param)
{
;CHECK: sf {{\$.}}, $4, $3
  %1 = sub %vec %param, <i32 1, i32 1>

;CHECK: bi $lr
  ret %vec %1
}

define %vec @test_mul(%vec %param)
{
;CHECK: mpyu
;CHECK: mpyh
;CHECK: a {{\$., \$., \$.}}
;CHECK: a {{\$., \$., \$.}}
  %1 = mul %vec %param, %param

;CHECK: bi $lr
  ret %vec %1
}

define <2 x i32> @test_splat(i32 %param ) {
;see svn log for why this is here...
;CHECK-NOT: or $3, $3, $3
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

define void @test_store( %vec %val, %vec* %ptr)
{
;CHECK: stqd $3, 0(${{.}})
;CHECK: bi $lr
  store %vec %val, %vec* %ptr
  ret void
}

;Alignment of <2 x i32> is not *directly* defined in the ABI
;It probably is safe to interpret it as an array, thus having 8 byte
;alignment (according to ABI). This tests that the size of
;[2 x <2 x i32>] is 16 bytes, i.e. there is no padding between the
;two arrays
define <2 x i32>* @test_alignment( [2 x <2 x i32>]* %ptr)
{
; CHECK-NOT:	ai	$3, $3, 16
; CHECK:	ai	$3, $3, 8
; CHECK:	bi	$lr
   %rv = getelementptr [2 x <2 x i32>]* %ptr, i32 0, i32 1
   ret <2 x i32>* %rv
}
