; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 -realign-stack=0 | not grep and

define <4 x float> @test(<4 x float>* %v1) {
        %tmp = load <4 x float>* %v1            ; <<4 x float>> [#uses=1]
        %tmp15 = bitcast <4 x float> %tmp to <2 x i64>          ; <<2 x i64>> [#uses=1]
        %tmp24 = and <2 x i64> %tmp15, bitcast (<4 x i32> < i32 0, i32 0, i32 -1, i32 -1 > to <2 x i64>)              ; <<2 x i64>> [#uses=1]
        %tmp31 = bitcast <2 x i64> %tmp24 to <4 x float>                ; <<4 x float>> [#uses=1]
        ret <4 x float> %tmp31
}

