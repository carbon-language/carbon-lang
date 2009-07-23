; RUN: llvm-as < %s | opt -instcombine | llvm-dis | FileCheck %s

; This turns into a&1 != 0
define <2 x i1> @test1(<2 x i64> %a) {
  %t = trunc <2 x i64> %a to <2 x i1>
  ret <2 x i1> %t

; CHECK: @test1
; CHECK:   and <2 x i64> %a, <i64 1, i64 1>
; CHECK:   icmp ne <2 x i64> %tmp, zeroinitializer
}

; The ashr turns into an lshr.
define <2 x i64> @test2(<2 x i64> %a) {
  %b = and <2 x i64> %a, <i64 65535, i64 65535>
  %t = ashr <2 x i64> %b, <i64 1, i64 1>
  ret <2 x i64> %t

; CHECK: @test2
; CHECK:   and <2 x i64> %a, <i64 65535, i64 65535>
; CHECK:   lshr <2 x i64> %b, <i64 1, i64 1>
}



define <2 x i64> @test3(<4 x float> %a, <4 x float> %b) nounwind readnone {
entry:
	%cmp = fcmp ord <4 x float> %a, zeroinitializer	
	%sext = sext <4 x i1> %cmp to <4 x i32>	
	%cmp4 = fcmp ord <4 x float> %b, zeroinitializer
	%sext5 = sext <4 x i1> %cmp4 to <4 x i32>
	%and = and <4 x i32> %sext, %sext5
	%conv = bitcast <4 x i32> %and to <2 x i64>
	ret <2 x i64> %conv
        
; CHECK: @test3
; CHECK:   fcmp ord <4 x float> %a, %b
}

define <2 x i64> @test4(<4 x float> %a, <4 x float> %b) nounwind readnone {
entry:
	%cmp = fcmp uno <4 x float> %a, zeroinitializer
	%sext = sext <4 x i1> %cmp to <4 x i32>
	%cmp4 = fcmp uno <4 x float> %b, zeroinitializer
	%sext5 = sext <4 x i1> %cmp4 to <4 x i32>
	%or = or <4 x i32> %sext, %sext5
	%conv = bitcast <4 x i32> %or to <2 x i64>
	ret <2 x i64> %conv
; CHECK: @test4
; CHECK:   fcmp uno <4 x float> %a, %b
}
