; RUN: llvm-as < %s | opt -gcse -instcombine | \
; RUN:   llvm-dis | not grep sub

define i32 @test_extractelement(<4 x i32> %V) {
        %R = extractelement <4 x i32> %V, i32 1         ; <i32> [#uses=1]
        %R2 = extractelement <4 x i32> %V, i32 1                ; <i32> [#uses=1]
        %V.upgrd.1 = sub i32 %R, %R2            ; <i32> [#uses=1]
        ret i32 %V.upgrd.1
}

define <4 x i32> @test_insertelement(<4 x i32> %V) {
        %R = insertelement <4 x i32> %V, i32 0, i32 0           ; <<4 x i32>> [#uses=1]
        %R2 = insertelement <4 x i32> %V, i32 0, i32 0          ; <<4 x i32>> [#uses=1]
        %x = sub <4 x i32> %R, %R2              ; <<4 x i32>> [#uses=1]
        ret <4 x i32> %x
}

define <4 x i32> @test_shufflevector(<4 x i32> %V) {
        %R = shufflevector <4 x i32> %V, <4 x i32> %V, <4 x i32> < i32 1, i32 undef, i32 7, i32 2 >             ; <<4 x i32>> [#uses=1]
        %R2 = shufflevector <4 x i32> %V, <4 x i32> %V, <4 x i32> < i32 1, i32 undef, i32 7, i32 2 >            ; <<4 x i32>> [#uses=1]
        %x = sub <4 x i32> %R, %R2              ; <<4 x i32>> [#uses=1]
        ret <4 x i32> %x
}

