; RUN: llvm-as -o - %s | llc -march=cellspu > %t1.s
; RUN: grep selb   %t1.s | count 160
; RUN: grep and    %t1.s | count 2
; RUN: grep xsbh   %t1.s | count 1
; RUN: grep xshw   %t1.s | count 2
target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128-s0:128:128"
target triple = "spu"

define <16 x i8> @selb_v16i8_1(<16 x i8> %arg1, <16 x i8> %arg2, <16 x i8> %arg3) {
	%A = xor <16 x i8> %arg3, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1,
                                    i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1,
                                    i8 -1, i8 -1, i8 -1, i8 -1 >
	%B = and <16 x i8> %A, %arg1		; <<16 x i8>> [#uses=1]
	%C = and <16 x i8> %arg2, %arg3		; <<16 x i8>> [#uses=1]
	%D = or <16 x i8> %B, %C		; <<16 x i8>> [#uses=1]
	ret <16 x i8> %D
}

define <16 x i8> @selb_v16i8_11(<16 x i8> %arg1, <16 x i8> %arg2, <16 x i8> %arg3) {
	%A = xor <16 x i8> %arg3, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1,
                                    i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1,
                                    i8 -1, i8 -1, i8 -1, i8 -1 >
	%B = and <16 x i8> %arg1, %A		; <<16 x i8>> [#uses=1]
	%C = and <16 x i8> %arg3, %arg2		; <<16 x i8>> [#uses=1]
	%D = or <16 x i8> %B, %C		; <<16 x i8>> [#uses=1]
	ret <16 x i8> %D
}

define <16 x i8> @selb_v16i8_12(<16 x i8> %arg1, <16 x i8> %arg2, <16 x i8> %arg3) {
	%A = xor <16 x i8> %arg3, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1,
                                    i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1,
                                    i8 -1, i8 -1, i8 -1, i8 -1 >
	%B = and <16 x i8> %arg1, %A		; <<16 x i8>> [#uses=1]
	%C = and <16 x i8> %arg2, %arg3		; <<16 x i8>> [#uses=1]
	%D = or <16 x i8> %B, %C		; <<16 x i8>> [#uses=1]
	ret <16 x i8> %D
}

define <16 x i8> @selb_v16i8_13(<16 x i8> %arg1, <16 x i8> %arg2, <16 x i8> %arg3) {
	%A = xor <16 x i8> %arg3, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1,
                                    i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1,
                                    i8 -1, i8 -1, i8 -1, i8 -1 >
	%B = and <16 x i8> %A, %arg1		; <<16 x i8>> [#uses=1]
	%C = and <16 x i8> %arg2, %arg3		; <<16 x i8>> [#uses=1]
	%D = or <16 x i8> %B, %C		; <<16 x i8>> [#uses=1]
	ret <16 x i8> %D
}

define <16 x i8> @selb_v16i8_2(<16 x i8> %arg1, <16 x i8> %arg2, <16 x i8> %arg3) {
	%A = xor <16 x i8> %arg1, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1,
                                    i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1,
                                    i8 -1, i8 -1, i8 -1, i8 -1 >
	%B = and <16 x i8> %A, %arg2		; <<16 x i8>> [#uses=1]
	%C = and <16 x i8> %arg3, %arg1		; <<16 x i8>> [#uses=1]
	%D = or <16 x i8> %B, %C		; <<16 x i8>> [#uses=1]
	ret <16 x i8> %D
}

define <16 x i8> @selb_v16i8_21(<16 x i8> %arg1, <16 x i8> %arg2, <16 x i8> %arg3) {
	%A = xor <16 x i8> %arg1, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1,
                                    i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1,
                                    i8 -1, i8 -1, i8 -1, i8 -1 >
	%B = and <16 x i8> %arg2, %A		; <<16 x i8>> [#uses=1]
	%C = and <16 x i8> %arg3, %arg1		; <<16 x i8>> [#uses=1]
	%D = or <16 x i8> %B, %C		; <<16 x i8>> [#uses=1]
	ret <16 x i8> %D
}

define <16 x i8> @selb_v16i8_3(<16 x i8> %arg1, <16 x i8> %arg2, <16 x i8> %arg3) {
	%A = xor <16 x i8> %arg2, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1,
                                    i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1,
                                    i8 -1, i8 -1, i8 -1, i8 -1 >
	%B = and <16 x i8> %A, %arg1		; <<16 x i8>> [#uses=1]
	%C = and <16 x i8> %arg3, %arg2		; <<16 x i8>> [#uses=1]
	%D = or <16 x i8> %B, %C		; <<16 x i8>> [#uses=1]
	ret <16 x i8> %D
}

define <16 x i8> @selb_v16i8_4(<16 x i8> %arg1, <16 x i8> %arg2, <16 x i8> %arg3) {
	%C = and <16 x i8> %arg3, %arg2		; <<16 x i8>> [#uses=1]
	%A = xor <16 x i8> %arg2, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1,
                                    i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1,
                                    i8 -1, i8 -1, i8 -1, i8 -1 >
	%B = and <16 x i8> %A, %arg1		; <<16 x i8>> [#uses=1]
	%D = or <16 x i8> %B, %C		; <<16 x i8>> [#uses=1]
	ret <16 x i8> %D
}

define <16 x i8> @selb_v16i8_41(<16 x i8> %arg1, <16 x i8> %arg2, <16 x i8> %arg3) {
	%C = and <16 x i8> %arg2, %arg3		; <<16 x i8>> [#uses=1]
	%A = xor <16 x i8> %arg2, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1,
                                    i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1,
                                    i8 -1, i8 -1, i8 -1, i8 -1 >
	%B = and <16 x i8> %arg1, %A		; <<16 x i8>> [#uses=1]
	%D = or <16 x i8> %C, %B		; <<16 x i8>> [#uses=1]
	ret <16 x i8> %D
}

define <16 x i8> @selb_v16i8_42(<16 x i8> %arg1, <16 x i8> %arg2, <16 x i8> %arg3) {
	%C = and <16 x i8> %arg2, %arg3		; <<16 x i8>> [#uses=1]
	%A = xor <16 x i8> %arg2, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1,
                                    i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1,
                                    i8 -1, i8 -1, i8 -1, i8 -1 >
	%B = and <16 x i8> %A, %arg1		; <<16 x i8>> [#uses=1]
	%D = or <16 x i8> %C, %B		; <<16 x i8>> [#uses=1]
	ret <16 x i8> %D
}

define <16 x i8> @selb_v16i8_5(<16 x i8> %arg1, <16 x i8> %arg2, <16 x i8> %arg3) {
	%C = and <16 x i8> %arg2, %arg1		; <<16 x i8>> [#uses=1]
	%A = xor <16 x i8> %arg1, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1,
                                    i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1,
                                    i8 -1, i8 -1, i8 -1, i8 -1 >
	%B = and <16 x i8> %A, %arg3		; <<16 x i8>> [#uses=1]
	%D = or <16 x i8> %B, %C		; <<16 x i8>> [#uses=1]
	ret <16 x i8> %D
}

define <8 x i16> @selb_v8i16_1(<8 x i16> %arg1, <8 x i16> %arg2, <8 x i16> %arg3) {
	%A = xor <8 x i16> %arg3, < i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1,
                                    i16 -1, i16 -1 >
	%B = and <8 x i16> %A, %arg1		; <<8 x i16>> [#uses=1]
	%C = and <8 x i16> %arg2, %arg3		; <<8 x i16>> [#uses=1]
	%D = or <8 x i16> %B, %C		; <<8 x i16>> [#uses=1]
	ret <8 x i16> %D
}

define <8 x i16> @selb_v8i16_11(<8 x i16> %arg1, <8 x i16> %arg2, <8 x i16> %arg3) {
	%A = xor <8 x i16> %arg3, < i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1,
                                    i16 -1, i16 -1 >
	%B = and <8 x i16> %arg1, %A		; <<8 x i16>> [#uses=1]
	%C = and <8 x i16> %arg3, %arg2		; <<8 x i16>> [#uses=1]
	%D = or <8 x i16> %B, %C		; <<8 x i16>> [#uses=1]
	ret <8 x i16> %D
}

define <8 x i16> @selb_v8i16_12(<8 x i16> %arg1, <8 x i16> %arg2, <8 x i16> %arg3) {
	%A = xor <8 x i16> %arg3, < i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1,
                                    i16 -1, i16 -1 >
	%B = and <8 x i16> %arg1, %A		; <<8 x i16>> [#uses=1]
	%C = and <8 x i16> %arg2, %arg3		; <<8 x i16>> [#uses=1]
	%D = or <8 x i16> %B, %C		; <<8 x i16>> [#uses=1]
	ret <8 x i16> %D
}

define <8 x i16> @selb_v8i16_13(<8 x i16> %arg1, <8 x i16> %arg2, <8 x i16> %arg3) {
	%A = xor <8 x i16> %arg3, < i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1,
                                    i16 -1, i16 -1 >
	%B = and <8 x i16> %A, %arg1		; <<8 x i16>> [#uses=1]
	%C = and <8 x i16> %arg2, %arg3		; <<8 x i16>> [#uses=1]
	%D = or <8 x i16> %B, %C		; <<8 x i16>> [#uses=1]
	ret <8 x i16> %D
}

define <8 x i16> @selb_v8i16_2(<8 x i16> %arg1, <8 x i16> %arg2, <8 x i16> %arg3) {
	%A = xor <8 x i16> %arg1, < i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1,
                                    i16 -1, i16 -1 >
	%B = and <8 x i16> %A, %arg2		; <<8 x i16>> [#uses=1]
	%C = and <8 x i16> %arg3, %arg1		; <<8 x i16>> [#uses=1]
	%D = or <8 x i16> %B, %C		; <<8 x i16>> [#uses=1]
	ret <8 x i16> %D
}

define <8 x i16> @selb_v8i16_21(<8 x i16> %arg1, <8 x i16> %arg2, <8 x i16> %arg3) {
	%A = xor <8 x i16> %arg1, < i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1,
                                    i16 -1, i16 -1 >
	%B = and <8 x i16> %arg2, %A		; <<8 x i16>> [#uses=1]
	%C = and <8 x i16> %arg3, %arg1		; <<8 x i16>> [#uses=1]
	%D = or <8 x i16> %B, %C		; <<8 x i16>> [#uses=1]
	ret <8 x i16> %D
}

define <8 x i16> @selb_v8i16_3(<8 x i16> %arg1, <8 x i16> %arg2, <8 x i16> %arg3) {
	%A = xor <8 x i16> %arg2, < i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1,
                                    i16 -1, i16 -1 >
	%B = and <8 x i16> %A, %arg1		; <<8 x i16>> [#uses=1]
	%C = and <8 x i16> %arg3, %arg2		; <<8 x i16>> [#uses=1]
	%D = or <8 x i16> %B, %C		; <<8 x i16>> [#uses=1]
	ret <8 x i16> %D
}

define <8 x i16> @selb_v8i16_4(<8 x i16> %arg1, <8 x i16> %arg2, <8 x i16> %arg3) {
	%C = and <8 x i16> %arg3, %arg2		; <<8 x i16>> [#uses=1]
	%A = xor <8 x i16> %arg2, < i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1,
                                    i16 -1, i16 -1 >
	%B = and <8 x i16> %A, %arg1		; <<8 x i16>> [#uses=1]
	%D = or <8 x i16> %B, %C		; <<8 x i16>> [#uses=1]
	ret <8 x i16> %D
}

define <8 x i16> @selb_v8i16_41(<8 x i16> %arg1, <8 x i16> %arg2, <8 x i16> %arg3) {
	%C = and <8 x i16> %arg2, %arg3		; <<8 x i16>> [#uses=1]
	%A = xor <8 x i16> %arg2, < i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1,
                                    i16 -1, i16 -1 >
	%B = and <8 x i16> %arg1, %A		; <<8 x i16>> [#uses=1]
	%D = or <8 x i16> %C, %B		; <<8 x i16>> [#uses=1]
	ret <8 x i16> %D
}

define <8 x i16> @selb_v8i16_42(<8 x i16> %arg1, <8 x i16> %arg2, <8 x i16> %arg3) {
	%C = and <8 x i16> %arg2, %arg3		; <<8 x i16>> [#uses=1]
	%A = xor <8 x i16> %arg2, < i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1,
                                    i16 -1, i16 -1 >
	%B = and <8 x i16> %A, %arg1		; <<8 x i16>> [#uses=1]
	%D = or <8 x i16> %C, %B		; <<8 x i16>> [#uses=1]
	ret <8 x i16> %D
}

define <8 x i16> @selb_v8i16_5(<8 x i16> %arg1, <8 x i16> %arg2, <8 x i16> %arg3) {
	%C = and <8 x i16> %arg2, %arg1		; <<8 x i16>> [#uses=1]
	%A = xor <8 x i16> %arg1, < i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1,
                                    i16 -1, i16 -1 >
	%B = and <8 x i16> %A, %arg3		; <<8 x i16>> [#uses=1]
	%D = or <8 x i16> %B, %C		; <<8 x i16>> [#uses=1]
	ret <8 x i16> %D
}

define <4 x i32> @selb_v4i32_1(<4 x i32> %arg1, <4 x i32> %arg2, <4 x i32> %arg3) {
	%tmpnot = xor <4 x i32> %arg3, < i32 -1, i32 -1, i32 -1, i32 -1 >		; <<4 x i32>> [#uses=1]
	%tmp2 = and <4 x i32> %tmpnot, %arg1		; <<4 x i32>> [#uses=1]
	%tmp5 = and <4 x i32> %arg2, %arg3		; <<4 x i32>> [#uses=1]
	%tmp6 = or <4 x i32> %tmp2, %tmp5		; <<4 x i32>> [#uses=1]
	ret <4 x i32> %tmp6
}

define <4 x i32> @selb_v4i32_2(<4 x i32> %arg1, <4 x i32> %arg2, <4 x i32> %arg3) {
	%tmpnot = xor <4 x i32> %arg3, < i32 -1, i32 -1, i32 -1, i32 -1 >		; <<4 x i32>> [#uses=1]
	%tmp2 = and <4 x i32> %tmpnot, %arg1		; <<4 x i32>> [#uses=1]
	%tmp5 = and <4 x i32> %arg2, %arg3		; <<4 x i32>> [#uses=1]
	%tmp6 = or <4 x i32> %tmp2, %tmp5		; <<4 x i32>> [#uses=1]
	ret <4 x i32> %tmp6
}

define <4 x i32> @selb_v4i32_3(<4 x i32> %arg1, <4 x i32> %arg2, <4 x i32> %arg3) {
	%tmpnot = xor <4 x i32> %arg3, < i32 -1, i32 -1, i32 -1, i32 -1 >		; <<4 x i32>> [#uses=1]
	%tmp2 = and <4 x i32> %tmpnot, %arg1		; <<4 x i32>> [#uses=1]
	%tmp5 = and <4 x i32> %arg3, %arg2		; <<4 x i32>> [#uses=1]
	%tmp6 = or <4 x i32> %tmp2, %tmp5		; <<4 x i32>> [#uses=1]
	ret <4 x i32> %tmp6
}

define <4 x i32> @selb_v4i32_4(<4 x i32> %arg1, <4 x i32> %arg2, <4 x i32> %arg3) {
	%tmp2 = and <4 x i32> %arg3, %arg2		; <<4 x i32>> [#uses=1]
	%tmp3not = xor <4 x i32> %arg3, < i32 -1, i32 -1, i32 -1, i32 -1 >		; <<4 x i32>> [#uses=1]
	%tmp5 = and <4 x i32> %tmp3not, %arg1		; <<4 x i32>> [#uses=1]
	%tmp6 = or <4 x i32> %tmp2, %tmp5		; <<4 x i32>> [#uses=1]
	ret <4 x i32> %tmp6
}

define <4 x i32> @selb_v4i32_5(<4 x i32> %arg1, <4 x i32> %arg2, <4 x i32> %arg3) {
	%tmp2 = and <4 x i32> %arg3, %arg2		; <<4 x i32>> [#uses=1]
	%tmp3not = xor <4 x i32> %arg3, < i32 -1, i32 -1, i32 -1, i32 -1 >		; <<4 x i32>> [#uses=1]
	%tmp5 = and <4 x i32> %tmp3not, %arg1		; <<4 x i32>> [#uses=1]
	%tmp6 = or <4 x i32> %tmp2, %tmp5		; <<4 x i32>> [#uses=1]
	ret <4 x i32> %tmp6
}

define i32 @selb_i32(i32 %arg1, i32 %arg2, i32 %arg3) {
	%tmp1not = xor i32 %arg3, -1		; <i32> [#uses=1]
	%tmp3 = and i32 %tmp1not, %arg1		; <i32> [#uses=1]
	%tmp6 = and i32 %arg3, %arg2		; <i32> [#uses=1]
	%tmp7 = or i32 %tmp3, %tmp6		; <i32> [#uses=1]
	ret i32 %tmp7
}

define i16 @selb_i16(i16 signext  %arg1, i16 signext  %arg2, i16 signext  %arg3) signext  {
	%tmp3 = and i16 %arg3, %arg1		; <i16> [#uses=1]
	%tmp4not = xor i16 %arg3, -1		; <i16> [#uses=1]
	%tmp6 = and i16 %tmp4not, %arg2		; <i16> [#uses=1]
	%retval1011 = or i16 %tmp3, %tmp6		; <i16> [#uses=1]
	ret i16 %retval1011
}

define i16 @selb_i16u(i16 zeroext  %arg1, i16 zeroext  %arg2, i16 zeroext  %arg3) zeroext  {
	%tmp3 = and i16 %arg3, %arg1		; <i16> [#uses=1]
	%tmp4not = xor i16 %arg3, -1		; <i16> [#uses=1]
	%tmp6 = and i16 %tmp4not, %arg2		; <i16> [#uses=1]
	%retval1011 = or i16 %tmp3, %tmp6		; <i16> [#uses=1]
	ret i16 %retval1011
}

define i8 @selb_i8u(i8 zeroext  %arg1, i8 zeroext  %arg2, i8 zeroext  %arg3) zeroext  {
	%tmp3 = and i8 %arg3, %arg1		; <i8> [#uses=1]
	%tmp4not = xor i8 %arg3, -1		; <i8> [#uses=1]
	%tmp6 = and i8 %tmp4not, %arg2		; <i8> [#uses=1]
	%retval1011 = or i8 %tmp3, %tmp6		; <i8> [#uses=1]
	ret i8 %retval1011
}

define i8 @selb_i8(i8 signext  %arg1, i8 signext  %arg2, i8 signext  %arg3) signext  {
	%tmp3 = and i8 %arg3, %arg1		; <i8> [#uses=1]
	%tmp4not = xor i8 %arg3, -1		; <i8> [#uses=1]
	%tmp6 = and i8 %tmp4not, %arg2		; <i8> [#uses=1]
	%retval1011 = or i8 %tmp3, %tmp6		; <i8> [#uses=1]
	ret i8 %retval1011
}
