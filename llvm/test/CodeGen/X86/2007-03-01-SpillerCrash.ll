; RUN: llvm-as < %s | llc -mtriple=x86_64-apple-darwin8 -mattr=+sse2

define void @test() {
test.exit:
	mul <4 x float> zeroinitializer, zeroinitializer		; <<4 x float>>:0 [#uses=4]
	load <4 x float>* null		; <<4 x float>>:1 [#uses=1]
	shufflevector <4 x float> %1, <4 x float> undef, <4 x i32> < i32 3, i32 3, i32 3, i32 3 >		; <<4 x float>>:2 [#uses=1]
	mul <4 x float> %0, %2		; <<4 x float>>:3 [#uses=1]
	sub <4 x float> zeroinitializer, %3		; <<4 x float>>:4 [#uses=1]
	mul <4 x float> %4, zeroinitializer		; <<4 x float>>:5 [#uses=2]
	bitcast <4 x float> zeroinitializer to <4 x i32>		; <<4 x i32>>:0 [#uses=1]
	and <4 x i32> %0, < i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647 >		; <<4 x i32>>:1 [#uses=1]
	bitcast <4 x i32> %1 to <4 x float>		; <<4 x float>>:6 [#uses=2]
	extractelement <4 x float> %6, i32 0		; <float>:0 [#uses=1]
	extractelement <4 x float> %6, i32 1		; <float>:1 [#uses=2]
	br i1 false, label %0, label %5

; <label>:0		; preds = %test.exit
	br i1 false, label %3, label %1

; <label>:1		; preds = %0
	br i1 false, label %5, label %2

; <label>:2		; preds = %1
	sub float -0.000000e+00, 0.000000e+00		; <float>:2 [#uses=1]
	%tmp207 = extractelement <4 x float> zeroinitializer, i32 0		; <float> [#uses=1]
	%tmp208 = extractelement <4 x float> zeroinitializer, i32 2		; <float> [#uses=1]
	sub float -0.000000e+00, %tmp208		; <float>:3 [#uses=1]
	%tmp155 = extractelement <4 x float> zeroinitializer, i32 0		; <float> [#uses=1]
	%tmp156 = extractelement <4 x float> zeroinitializer, i32 2		; <float> [#uses=1]
	sub float -0.000000e+00, %tmp156		; <float>:4 [#uses=1]
	br label %5

; <label>:3		; preds = %0
	br i1 false, label %5, label %4

; <label>:4		; preds = %3
	br label %5

; <label>:5		; preds = %4, %3, %2, %1, %test.exit
	phi i32 [ 5, %4 ], [ 3, %2 ], [ 1, %test.exit ], [ 2, %1 ], [ 4, %3 ]		; <i32>:0 [#uses=0]
	phi float [ 0.000000e+00, %4 ], [ %4, %2 ], [ 0.000000e+00, %test.exit ], [ 0.000000e+00, %1 ], [ 0.000000e+00, %3 ]		; <float>:5 [#uses=1]
	phi float [ 0.000000e+00, %4 ], [ %tmp155, %2 ], [ 0.000000e+00, %test.exit ], [ 0.000000e+00, %1 ], [ 0.000000e+00, %3 ]		; <float>:6 [#uses=1]
	phi float [ 0.000000e+00, %4 ], [ %3, %2 ], [ 0.000000e+00, %test.exit ], [ 0.000000e+00, %1 ], [ 0.000000e+00, %3 ]		; <float>:7 [#uses=1]
	phi float [ 0.000000e+00, %4 ], [ %tmp207, %2 ], [ 0.000000e+00, %test.exit ], [ 0.000000e+00, %1 ], [ 0.000000e+00, %3 ]		; <float>:8 [#uses=1]
	phi float [ 0.000000e+00, %4 ], [ %1, %2 ], [ %0, %test.exit ], [ %1, %1 ], [ 0.000000e+00, %3 ]		; <float>:9 [#uses=2]
	phi float [ 0.000000e+00, %4 ], [ %2, %2 ], [ 0.000000e+00, %test.exit ], [ 0.000000e+00, %1 ], [ 0.000000e+00, %3 ]		; <float>:10 [#uses=1]
	phi float [ 0.000000e+00, %4 ], [ 0.000000e+00, %2 ], [ 0.000000e+00, %test.exit ], [ 0.000000e+00, %1 ], [ 0.000000e+00, %3 ]		; <float>:11 [#uses=1]
	insertelement <4 x float> undef, float %11, i32 0		; <<4 x float>>:7 [#uses=1]
	insertelement <4 x float> %7, float %10, i32 1		; <<4 x float>>:8 [#uses=0]
	insertelement <4 x float> undef, float %8, i32 0		; <<4 x float>>:9 [#uses=1]
	insertelement <4 x float> %9, float %7, i32 1		; <<4 x float>>:10 [#uses=1]
	insertelement <4 x float> %10, float %9, i32 2		; <<4 x float>>:11 [#uses=1]
	insertelement <4 x float> %11, float %9, i32 3		; <<4 x float>>:12 [#uses=1]
	fdiv <4 x float> %12, zeroinitializer		; <<4 x float>>:13 [#uses=1]
	mul <4 x float> %13, < float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01 >		; <<4 x float>>:14 [#uses=1]
	insertelement <4 x float> undef, float %6, i32 0		; <<4 x float>>:15 [#uses=1]
	insertelement <4 x float> %15, float %5, i32 1		; <<4 x float>>:16 [#uses=0]
	br i1 false, label %foo.exit, label %6

; <label>:6		; preds = %5
	extractelement <4 x float> %0, i32 0		; <float>:12 [#uses=1]
	fcmp ogt float %12, 0.000000e+00		; <i1>:0 [#uses=1]
	extractelement <4 x float> %0, i32 2		; <float>:13 [#uses=1]
	extractelement <4 x float> %0, i32 1		; <float>:14 [#uses=1]
	sub float -0.000000e+00, %14		; <float>:15 [#uses=2]
	%tmp189 = extractelement <4 x float> %5, i32 2		; <float> [#uses=1]
	br i1 %0, label %7, label %8

; <label>:7		; preds = %6
	sub float -0.000000e+00, %tmp189		; <float>:16 [#uses=0]
	br label %foo.exit

; <label>:8		; preds = %6
	%tmp192 = extractelement <4 x float> %5, i32 1		; <float> [#uses=1]
	sub float -0.000000e+00, %tmp192		; <float>:17 [#uses=1]
	br label %foo.exit

foo.exit:		; preds = %8, %7, %5
	phi float [ 0.000000e+00, %7 ], [ %17, %8 ], [ 0.000000e+00, %5 ]		; <float>:18 [#uses=0]
	phi float [ %15, %7 ], [ %15, %8 ], [ 0.000000e+00, %5 ]		; <float>:19 [#uses=0]
	phi float [ 0.000000e+00, %7 ], [ %13, %8 ], [ 0.000000e+00, %5 ]		; <float>:20 [#uses=0]
	shufflevector <4 x float> %14, <4 x float> zeroinitializer, <4 x i32> < i32 0, i32 4, i32 1, i32 5 >		; <<4 x float>>:17 [#uses=0]
	unreachable
}
