; Test that vectors are scalarized/lowered correctly
; (with both legalization methods).
; RUN: llc -march=x86 -promote-elements < %s
; RUN: llc -march=x86                   < %s

; A simple test to check copyToParts and copyFromParts

define <4 x i64> @test_param_0(<4 x i64> %A, <2 x i32> %B, <4 x i8> %C)  {
   ret <4 x i64> %A
}

define <2 x i32> @test_param_1(<4 x i64> %A, <2 x i32> %B, <4 x i8> %C)  {
   ret <2 x i32> %B
}

define <4 x i8> @test_param_2(<4 x i64> %A, <2 x i32> %B, <4 x i8> %C)  {
   ret <4 x i8> %C
}


