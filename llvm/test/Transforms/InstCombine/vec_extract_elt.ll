; RUN: opt < %s -instcombine -S | not grep extractelement

define i32 @test(float %f) {
        %tmp7 = insertelement <4 x float> undef, float %f, i32 0                ; <<4 x float>> [#uses=1]
        %tmp17 = bitcast <4 x float> %tmp7 to <4 x i32>         ; <<4 x i32>> [#uses=1]
        %tmp19 = extractelement <4 x i32> %tmp17, i32 0         ; <i32> [#uses=1]
        ret i32 %tmp19
}

define i64 @test2(i64 %in) {
  %vec = insertelement <8 x i64> undef, i64 %in, i32 0
  %splat = shufflevector <8 x i64> %vec, <8 x i64> undef, <8 x i32> zeroinitializer
  %add = add <8 x i64> %splat, <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7>
  %scl1 = extractelement <8 x i64> %add, i32 0
  %scl2 = extractelement <8 x i64> %add, i32 0
  %r = add i64 %scl1, %scl2
  ret i64 %r
}

