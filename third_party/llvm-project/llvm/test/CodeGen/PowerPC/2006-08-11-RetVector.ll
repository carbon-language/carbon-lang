; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- -mcpu=g5 -ppc-disable-perfect-shuffle=false | grep vsldoi
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- -mcpu=g5 -ppc-disable-perfect-shuffle=false | not grep vor

; TODO: Fix this case when disabling perfect shuffle

define <4 x float> @func(<4 x float> %fp0, <4 x float> %fp1) {
        %tmp76 = shufflevector <4 x float> %fp0, <4 x float> %fp1, <4 x i32> < i32 0, i32 1, i32 2, i32 7 >     ; <<4 x float>> [#uses=1]
        ret <4 x float> %tmp76
}

