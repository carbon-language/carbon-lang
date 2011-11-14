; RUN: llc < %s -march=x86-64 -enable-lsr-nested -o %t
; RUN: not grep inc %t
; RUN: grep dec %t | count 2
; RUN: grep addq %t | count 12
; RUN: not grep addb %t
; RUN: not grep leal %t
; RUN: not grep movq %t

; IV users in each of the loops from other loops shouldn't cause LSR
; to insert new induction variables. Previously it would create a
; flood of new induction variables.
; Also, the loop reversal should kick in once.
;
; In this example, performing LSR on the entire loop nest,
; as opposed to only the inner loop can further reduce induction variables,
; and their related instructions and registers.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(float* %A, i32 %IA, float* %B, i32 %IB, float* nocapture %C, i32 %N) nounwind {
entry:
      %0 = xor i32 %IA, 1		; <i32> [#uses=1]
      %1 = xor i32 %IB, 1		; <i32> [#uses=1]
      %2 = or i32 %1, %0		; <i32> [#uses=1]
      %3 = icmp eq i32 %2, 0		; <i1> [#uses=1]
      br i1 %3, label %bb2, label %bb13

bb:		; preds = %bb3
      %4 = load float* %A_addr.0, align 4		; <float> [#uses=1]
      %5 = load float* %B_addr.0, align 4		; <float> [#uses=1]
      %6 = fmul float %4, %5		; <float> [#uses=1]
      %7 = fadd float %6, %Sum0.0		; <float> [#uses=1]
      %indvar.next154 = add i64 %B_addr.0.rec, 1		; <i64> [#uses=1]
      br label %bb2

bb2:		; preds = %entry, %bb
      %B_addr.0.rec = phi i64 [ %indvar.next154, %bb ], [ 0, %entry ]		; <i64> [#uses=14]
      %Sum0.0 = phi float [ %7, %bb ], [ 0.000000e+00, %entry ]		; <float> [#uses=5]
      %indvar146 = trunc i64 %B_addr.0.rec to i32		; <i32> [#uses=1]
      %N_addr.0 = sub i32 %N, %indvar146		; <i32> [#uses=6]
      %A_addr.0 = getelementptr float* %A, i64 %B_addr.0.rec		; <float*> [#uses=4]
      %B_addr.0 = getelementptr float* %B, i64 %B_addr.0.rec		; <float*> [#uses=4]
      %8 = icmp sgt i32 %N_addr.0, 0		; <i1> [#uses=1]
      br i1 %8, label %bb3, label %bb4

bb3:		; preds = %bb2
      %9 = ptrtoint float* %A_addr.0 to i64		; <i64> [#uses=1]
      %10 = and i64 %9, 15		; <i64> [#uses=1]
      %11 = icmp eq i64 %10, 0		; <i1> [#uses=1]
      br i1 %11, label %bb4, label %bb

bb4:		; preds = %bb3, %bb2
      %12 = ptrtoint float* %B_addr.0 to i64		; <i64> [#uses=1]
      %13 = and i64 %12, 15		; <i64> [#uses=1]
      %14 = icmp eq i64 %13, 0		; <i1> [#uses=1]
      %15 = icmp sgt i32 %N_addr.0, 15		; <i1> [#uses=2]
      br i1 %14, label %bb6.preheader, label %bb10.preheader

bb10.preheader:		; preds = %bb4
      br i1 %15, label %bb9, label %bb12.loopexit

bb6.preheader:		; preds = %bb4
      br i1 %15, label %bb5, label %bb8.loopexit

bb5:		; preds = %bb5, %bb6.preheader
      %indvar143 = phi i64 [ 0, %bb6.preheader ], [ %indvar.next144, %bb5 ]		; <i64> [#uses=3]
      %vSum0.072 = phi <4 x float> [ zeroinitializer, %bb6.preheader ], [ %21, %bb5 ]		; <<4 x float>> [#uses=1]
	%vSum1.070 = phi <4 x float> [ zeroinitializer, %bb6.preheader ], [ %29, %bb5 ]		; <<4 x float>> [#uses=1]
	%vSum2.069 = phi <4 x float> [ zeroinitializer, %bb6.preheader ], [ %37, %bb5 ]		; <<4 x float>> [#uses=1]
	%vSum3.067 = phi <4 x float> [ zeroinitializer, %bb6.preheader ], [ %45, %bb5 ]		; <<4 x float>> [#uses=1]
	%indvar145 = trunc i64 %indvar143 to i32		; <i32> [#uses=1]
	%tmp150 = mul i32 %indvar145, -16		; <i32> [#uses=1]
	%N_addr.268 = add i32 %tmp150, %N_addr.0		; <i32> [#uses=1]
	%A_addr.273.rec = shl i64 %indvar143, 4		; <i64> [#uses=5]
	%B_addr.0.sum180 = add i64 %B_addr.0.rec, %A_addr.273.rec		; <i64> [#uses=2]
	%B_addr.271 = getelementptr float* %B, i64 %B_addr.0.sum180		; <float*> [#uses=1]
	%A_addr.273 = getelementptr float* %A, i64 %B_addr.0.sum180		; <float*> [#uses=1]
	tail call void asm sideeffect ";# foo", "~{dirflag},~{fpsr},~{flags}"() nounwind
	%16 = bitcast float* %A_addr.273 to <4 x float>*		; <<4 x float>*> [#uses=1]
	%17 = load <4 x float>* %16, align 16		; <<4 x float>> [#uses=1]
	%18 = bitcast float* %B_addr.271 to <4 x float>*		; <<4 x float>*> [#uses=1]
	%19 = load <4 x float>* %18, align 16		; <<4 x float>> [#uses=1]
	%20 = fmul <4 x float> %17, %19		; <<4 x float>> [#uses=1]
	%21 = fadd <4 x float> %20, %vSum0.072		; <<4 x float>> [#uses=2]
	%A_addr.273.sum163 = or i64 %A_addr.273.rec, 4		; <i64> [#uses=1]
	%A_addr.0.sum175 = add i64 %B_addr.0.rec, %A_addr.273.sum163		; <i64> [#uses=2]
	%22 = getelementptr float* %A, i64 %A_addr.0.sum175		; <float*> [#uses=1]
	%23 = bitcast float* %22 to <4 x float>*		; <<4 x float>*> [#uses=1]
	%24 = load <4 x float>* %23, align 16		; <<4 x float>> [#uses=1]
	%25 = getelementptr float* %B, i64 %A_addr.0.sum175		; <float*> [#uses=1]
	%26 = bitcast float* %25 to <4 x float>*		; <<4 x float>*> [#uses=1]
	%27 = load <4 x float>* %26, align 16		; <<4 x float>> [#uses=1]
	%28 = fmul <4 x float> %24, %27		; <<4 x float>> [#uses=1]
	%29 = fadd <4 x float> %28, %vSum1.070		; <<4 x float>> [#uses=2]
	%A_addr.273.sum161 = or i64 %A_addr.273.rec, 8		; <i64> [#uses=1]
	%A_addr.0.sum174 = add i64 %B_addr.0.rec, %A_addr.273.sum161		; <i64> [#uses=2]
	%30 = getelementptr float* %A, i64 %A_addr.0.sum174		; <float*> [#uses=1]
	%31 = bitcast float* %30 to <4 x float>*		; <<4 x float>*> [#uses=1]
	%32 = load <4 x float>* %31, align 16		; <<4 x float>> [#uses=1]
	%33 = getelementptr float* %B, i64 %A_addr.0.sum174		; <float*> [#uses=1]
	%34 = bitcast float* %33 to <4 x float>*		; <<4 x float>*> [#uses=1]
	%35 = load <4 x float>* %34, align 16		; <<4 x float>> [#uses=1]
	%36 = fmul <4 x float> %32, %35		; <<4 x float>> [#uses=1]
	%37 = fadd <4 x float> %36, %vSum2.069		; <<4 x float>> [#uses=2]
	%A_addr.273.sum159 = or i64 %A_addr.273.rec, 12		; <i64> [#uses=1]
	%A_addr.0.sum173 = add i64 %B_addr.0.rec, %A_addr.273.sum159		; <i64> [#uses=2]
	%38 = getelementptr float* %A, i64 %A_addr.0.sum173		; <float*> [#uses=1]
	%39 = bitcast float* %38 to <4 x float>*		; <<4 x float>*> [#uses=1]
	%40 = load <4 x float>* %39, align 16		; <<4 x float>> [#uses=1]
	%41 = getelementptr float* %B, i64 %A_addr.0.sum173		; <float*> [#uses=1]
	%42 = bitcast float* %41 to <4 x float>*		; <<4 x float>*> [#uses=1]
	%43 = load <4 x float>* %42, align 16		; <<4 x float>> [#uses=1]
	%44 = fmul <4 x float> %40, %43		; <<4 x float>> [#uses=1]
	%45 = fadd <4 x float> %44, %vSum3.067		; <<4 x float>> [#uses=2]
	%.rec83 = add i64 %A_addr.273.rec, 16		; <i64> [#uses=1]
	%A_addr.0.sum172 = add i64 %B_addr.0.rec, %.rec83		; <i64> [#uses=2]
	%46 = getelementptr float* %A, i64 %A_addr.0.sum172		; <float*> [#uses=1]
	%47 = getelementptr float* %B, i64 %A_addr.0.sum172		; <float*> [#uses=1]
	%48 = add i32 %N_addr.268, -16		; <i32> [#uses=2]
	%49 = icmp sgt i32 %48, 15		; <i1> [#uses=1]
	%indvar.next144 = add i64 %indvar143, 1		; <i64> [#uses=1]
	br i1 %49, label %bb5, label %bb8.loopexit

bb7:		; preds = %bb7, %bb8.loopexit
	%indvar130 = phi i64 [ 0, %bb8.loopexit ], [ %indvar.next131, %bb7 ]		; <i64> [#uses=3]
	%vSum0.260 = phi <4 x float> [ %vSum0.0.lcssa, %bb8.loopexit ], [ %55, %bb7 ]		; <<4 x float>> [#uses=1]
	%indvar132 = trunc i64 %indvar130 to i32		; <i32> [#uses=1]
	%tmp133 = mul i32 %indvar132, -4		; <i32> [#uses=1]
	%N_addr.358 = add i32 %tmp133, %N_addr.2.lcssa		; <i32> [#uses=1]
	%A_addr.361.rec = shl i64 %indvar130, 2		; <i64> [#uses=3]
	%B_addr.359 = getelementptr float* %B_addr.2.lcssa, i64 %A_addr.361.rec		; <float*> [#uses=1]
	%A_addr.361 = getelementptr float* %A_addr.2.lcssa, i64 %A_addr.361.rec		; <float*> [#uses=1]
	%50 = bitcast float* %A_addr.361 to <4 x float>*		; <<4 x float>*> [#uses=1]
	%51 = load <4 x float>* %50, align 16		; <<4 x float>> [#uses=1]
	%52 = bitcast float* %B_addr.359 to <4 x float>*		; <<4 x float>*> [#uses=1]
	%53 = load <4 x float>* %52, align 16		; <<4 x float>> [#uses=1]
	%54 = fmul <4 x float> %51, %53		; <<4 x float>> [#uses=1]
	%55 = fadd <4 x float> %54, %vSum0.260		; <<4 x float>> [#uses=2]
	%.rec85 = add i64 %A_addr.361.rec, 4		; <i64> [#uses=2]
	%56 = getelementptr float* %A_addr.2.lcssa, i64 %.rec85		; <float*> [#uses=1]
	%57 = getelementptr float* %B_addr.2.lcssa, i64 %.rec85		; <float*> [#uses=1]
	%58 = add i32 %N_addr.358, -4		; <i32> [#uses=2]
	%59 = icmp sgt i32 %58, 3		; <i1> [#uses=1]
	%indvar.next131 = add i64 %indvar130, 1		; <i64> [#uses=1]
	br i1 %59, label %bb7, label %bb13

bb8.loopexit:		; preds = %bb5, %bb6.preheader
	%A_addr.2.lcssa = phi float* [ %A_addr.0, %bb6.preheader ], [ %46, %bb5 ]		; <float*> [#uses=3]
	%vSum0.0.lcssa = phi <4 x float> [ zeroinitializer, %bb6.preheader ], [ %21, %bb5 ]		; <<4 x float>> [#uses=2]
	%B_addr.2.lcssa = phi float* [ %B_addr.0, %bb6.preheader ], [ %47, %bb5 ]		; <float*> [#uses=3]
	%vSum1.0.lcssa = phi <4 x float> [ zeroinitializer, %bb6.preheader ], [ %29, %bb5 ]		; <<4 x float>> [#uses=2]
	%vSum2.0.lcssa = phi <4 x float> [ zeroinitializer, %bb6.preheader ], [ %37, %bb5 ]		; <<4 x float>> [#uses=2]
	%N_addr.2.lcssa = phi i32 [ %N_addr.0, %bb6.preheader ], [ %48, %bb5 ]		; <i32> [#uses=3]
	%vSum3.0.lcssa = phi <4 x float> [ zeroinitializer, %bb6.preheader ], [ %45, %bb5 ]		; <<4 x float>> [#uses=2]
	%60 = icmp sgt i32 %N_addr.2.lcssa, 3		; <i1> [#uses=1]
	br i1 %60, label %bb7, label %bb13

bb9:		; preds = %bb9, %bb10.preheader
	%indvar106 = phi i64 [ 0, %bb10.preheader ], [ %indvar.next107, %bb9 ]		; <i64> [#uses=3]
	%vSum0.339 = phi <4 x float> [ zeroinitializer, %bb10.preheader ], [ %75, %bb9 ]		; <<4 x float>> [#uses=1]
	%vSum1.237 = phi <4 x float> [ zeroinitializer, %bb10.preheader ], [ %80, %bb9 ]		; <<4 x float>> [#uses=1]
	%vSum2.236 = phi <4 x float> [ zeroinitializer, %bb10.preheader ], [ %85, %bb9 ]		; <<4 x float>> [#uses=1]
	%vSum3.234 = phi <4 x float> [ zeroinitializer, %bb10.preheader ], [ %90, %bb9 ]		; <<4 x float>> [#uses=1]
	%indvar108 = trunc i64 %indvar106 to i32		; <i32> [#uses=1]
	%tmp113 = mul i32 %indvar108, -16		; <i32> [#uses=1]
	%N_addr.435 = add i32 %tmp113, %N_addr.0		; <i32> [#uses=1]
	%A_addr.440.rec = shl i64 %indvar106, 4		; <i64> [#uses=5]
	%B_addr.0.sum = add i64 %B_addr.0.rec, %A_addr.440.rec		; <i64> [#uses=2]
	%B_addr.438 = getelementptr float* %B, i64 %B_addr.0.sum		; <float*> [#uses=1]
	%A_addr.440 = getelementptr float* %A, i64 %B_addr.0.sum		; <float*> [#uses=1]
	%61 = bitcast float* %B_addr.438 to <4 x float>*		; <i8*> [#uses=1]
	%62 = load <4 x float>* %61, align 1
	%B_addr.438.sum169 = or i64 %A_addr.440.rec, 4		; <i64> [#uses=1]
	%B_addr.0.sum187 = add i64 %B_addr.0.rec, %B_addr.438.sum169		; <i64> [#uses=2]
	%63 = getelementptr float* %B, i64 %B_addr.0.sum187		; <float*> [#uses=1]
	%64 = bitcast float* %63 to <4 x float>*		; <i8*> [#uses=1]
	%65 = load <4 x float>* %64, align 1
	%B_addr.438.sum168 = or i64 %A_addr.440.rec, 8		; <i64> [#uses=1]
	%B_addr.0.sum186 = add i64 %B_addr.0.rec, %B_addr.438.sum168		; <i64> [#uses=2]
	%66 = getelementptr float* %B, i64 %B_addr.0.sum186		; <float*> [#uses=1]
	%67 = bitcast float* %66 to <4 x float>*		; <i8*> [#uses=1]
	%68 = load <4 x float>* %67, align 1
	%B_addr.438.sum167 = or i64 %A_addr.440.rec, 12		; <i64> [#uses=1]
	%B_addr.0.sum185 = add i64 %B_addr.0.rec, %B_addr.438.sum167		; <i64> [#uses=2]
	%69 = getelementptr float* %B, i64 %B_addr.0.sum185		; <float*> [#uses=1]
	%70 = bitcast float* %69 to <4 x float>*		; <i8*> [#uses=1]
	%71 = load <4 x float>* %70, align 1
	%72 = bitcast float* %A_addr.440 to <4 x float>*		; <<4 x float>*> [#uses=1]
	%73 = load <4 x float>* %72, align 16		; <<4 x float>> [#uses=1]
	%74 = fmul <4 x float> %73, %62		; <<4 x float>> [#uses=1]
	%75 = fadd <4 x float> %74, %vSum0.339		; <<4 x float>> [#uses=2]
	%76 = getelementptr float* %A, i64 %B_addr.0.sum187		; <float*> [#uses=1]
	%77 = bitcast float* %76 to <4 x float>*		; <<4 x float>*> [#uses=1]
	%78 = load <4 x float>* %77, align 16		; <<4 x float>> [#uses=1]
	%79 = fmul <4 x float> %78, %65		; <<4 x float>> [#uses=1]
	%80 = fadd <4 x float> %79, %vSum1.237		; <<4 x float>> [#uses=2]
	%81 = getelementptr float* %A, i64 %B_addr.0.sum186		; <float*> [#uses=1]
	%82 = bitcast float* %81 to <4 x float>*		; <<4 x float>*> [#uses=1]
	%83 = load <4 x float>* %82, align 16		; <<4 x float>> [#uses=1]
	%84 = fmul <4 x float> %83, %68		; <<4 x float>> [#uses=1]
	%85 = fadd <4 x float> %84, %vSum2.236		; <<4 x float>> [#uses=2]
	%86 = getelementptr float* %A, i64 %B_addr.0.sum185		; <float*> [#uses=1]
	%87 = bitcast float* %86 to <4 x float>*		; <<4 x float>*> [#uses=1]
	%88 = load <4 x float>* %87, align 16		; <<4 x float>> [#uses=1]
	%89 = fmul <4 x float> %88, %71		; <<4 x float>> [#uses=1]
	%90 = fadd <4 x float> %89, %vSum3.234		; <<4 x float>> [#uses=2]
	%.rec89 = add i64 %A_addr.440.rec, 16		; <i64> [#uses=1]
	%A_addr.0.sum170 = add i64 %B_addr.0.rec, %.rec89		; <i64> [#uses=2]
	%91 = getelementptr float* %A, i64 %A_addr.0.sum170		; <float*> [#uses=1]
	%92 = getelementptr float* %B, i64 %A_addr.0.sum170		; <float*> [#uses=1]
	%93 = add i32 %N_addr.435, -16		; <i32> [#uses=2]
	%94 = icmp sgt i32 %93, 15		; <i1> [#uses=1]
	%indvar.next107 = add i64 %indvar106, 1		; <i64> [#uses=1]
	br i1 %94, label %bb9, label %bb12.loopexit

bb11:		; preds = %bb11, %bb12.loopexit
	%indvar = phi i64 [ 0, %bb12.loopexit ], [ %indvar.next, %bb11 ]		; <i64> [#uses=3]
	%vSum0.428 = phi <4 x float> [ %vSum0.3.lcssa, %bb12.loopexit ], [ %100, %bb11 ]		; <<4 x float>> [#uses=1]
	%indvar96 = trunc i64 %indvar to i32		; <i32> [#uses=1]
	%tmp = mul i32 %indvar96, -4		; <i32> [#uses=1]
	%N_addr.526 = add i32 %tmp, %N_addr.4.lcssa		; <i32> [#uses=1]
	%A_addr.529.rec = shl i64 %indvar, 2		; <i64> [#uses=3]
	%B_addr.527 = getelementptr float* %B_addr.4.lcssa, i64 %A_addr.529.rec		; <float*> [#uses=1]
	%A_addr.529 = getelementptr float* %A_addr.4.lcssa, i64 %A_addr.529.rec		; <float*> [#uses=1]
	%95 = bitcast float* %B_addr.527 to <4 x float>*		; <i8*> [#uses=1]
	%96 = load <4 x float>* %95, align 1
	%97 = bitcast float* %A_addr.529 to <4 x float>*		; <<4 x float>*> [#uses=1]
	%98 = load <4 x float>* %97, align 16		; <<4 x float>> [#uses=1]
	%99 = fmul <4 x float> %98, %96		; <<4 x float>> [#uses=1]
	%100 = fadd <4 x float> %99, %vSum0.428		; <<4 x float>> [#uses=2]
	%.rec91 = add i64 %A_addr.529.rec, 4		; <i64> [#uses=2]
	%101 = getelementptr float* %A_addr.4.lcssa, i64 %.rec91		; <float*> [#uses=1]
	%102 = getelementptr float* %B_addr.4.lcssa, i64 %.rec91		; <float*> [#uses=1]
	%103 = add i32 %N_addr.526, -4		; <i32> [#uses=2]
	%104 = icmp sgt i32 %103, 3		; <i1> [#uses=1]
	%indvar.next = add i64 %indvar, 1		; <i64> [#uses=1]
	br i1 %104, label %bb11, label %bb13

bb12.loopexit:		; preds = %bb9, %bb10.preheader
	%A_addr.4.lcssa = phi float* [ %A_addr.0, %bb10.preheader ], [ %91, %bb9 ]		; <float*> [#uses=3]
	%vSum0.3.lcssa = phi <4 x float> [ zeroinitializer, %bb10.preheader ], [ %75, %bb9 ]		; <<4 x float>> [#uses=2]
	%B_addr.4.lcssa = phi float* [ %B_addr.0, %bb10.preheader ], [ %92, %bb9 ]		; <float*> [#uses=3]
	%vSum1.2.lcssa = phi <4 x float> [ zeroinitializer, %bb10.preheader ], [ %80, %bb9 ]		; <<4 x float>> [#uses=2]
	%vSum2.2.lcssa = phi <4 x float> [ zeroinitializer, %bb10.preheader ], [ %85, %bb9 ]		; <<4 x float>> [#uses=2]
	%N_addr.4.lcssa = phi i32 [ %N_addr.0, %bb10.preheader ], [ %93, %bb9 ]		; <i32> [#uses=3]
	%vSum3.2.lcssa = phi <4 x float> [ zeroinitializer, %bb10.preheader ], [ %90, %bb9 ]		; <<4 x float>> [#uses=2]
	%105 = icmp sgt i32 %N_addr.4.lcssa, 3		; <i1> [#uses=1]
	br i1 %105, label %bb11, label %bb13

bb13:		; preds = %bb12.loopexit, %bb11, %bb8.loopexit, %bb7, %entry
	%Sum0.1 = phi float [ 0.000000e+00, %entry ], [ %Sum0.0, %bb7 ], [ %Sum0.0, %bb8.loopexit ], [ %Sum0.0, %bb11 ], [ %Sum0.0, %bb12.loopexit ]		; <float> [#uses=1]
	%vSum3.1 = phi <4 x float> [ zeroinitializer, %entry ], [ %vSum3.0.lcssa, %bb7 ], [ %vSum3.0.lcssa, %bb8.loopexit ], [ %vSum3.2.lcssa, %bb11 ], [ %vSum3.2.lcssa, %bb12.loopexit ]		; <<4 x float>> [#uses=1]
	%N_addr.1 = phi i32 [ %N, %entry ], [ %N_addr.2.lcssa, %bb8.loopexit ], [ %58, %bb7 ], [ %N_addr.4.lcssa, %bb12.loopexit ], [ %103, %bb11 ]		; <i32> [#uses=2]
	%vSum2.1 = phi <4 x float> [ zeroinitializer, %entry ], [ %vSum2.0.lcssa, %bb7 ], [ %vSum2.0.lcssa, %bb8.loopexit ], [ %vSum2.2.lcssa, %bb11 ], [ %vSum2.2.lcssa, %bb12.loopexit ]		; <<4 x float>> [#uses=1]
	%vSum1.1 = phi <4 x float> [ zeroinitializer, %entry ], [ %vSum1.0.lcssa, %bb7 ], [ %vSum1.0.lcssa, %bb8.loopexit ], [ %vSum1.2.lcssa, %bb11 ], [ %vSum1.2.lcssa, %bb12.loopexit ]		; <<4 x float>> [#uses=1]
	%B_addr.1 = phi float* [ %B, %entry ], [ %B_addr.2.lcssa, %bb8.loopexit ], [ %57, %bb7 ], [ %B_addr.4.lcssa, %bb12.loopexit ], [ %102, %bb11 ]		; <float*> [#uses=1]
	%vSum0.1 = phi <4 x float> [ zeroinitializer, %entry ], [ %vSum0.0.lcssa, %bb8.loopexit ], [ %55, %bb7 ], [ %vSum0.3.lcssa, %bb12.loopexit ], [ %100, %bb11 ]		; <<4 x float>> [#uses=1]
	%A_addr.1 = phi float* [ %A, %entry ], [ %A_addr.2.lcssa, %bb8.loopexit ], [ %56, %bb7 ], [ %A_addr.4.lcssa, %bb12.loopexit ], [ %101, %bb11 ]		; <float*> [#uses=1]
	%106 = fadd <4 x float> %vSum0.1, %vSum2.1		; <<4 x float>> [#uses=1]
	%107 = fadd <4 x float> %vSum1.1, %vSum3.1		; <<4 x float>> [#uses=1]
	%108 = fadd <4 x float> %106, %107		; <<4 x float>> [#uses=4]
	%tmp23 = extractelement <4 x float> %108, i32 0		; <float> [#uses=1]
	%tmp21 = extractelement <4 x float> %108, i32 1		; <float> [#uses=1]
	%109 = fadd float %tmp23, %tmp21		; <float> [#uses=1]
	%tmp19 = extractelement <4 x float> %108, i32 2		; <float> [#uses=1]
	%tmp17 = extractelement <4 x float> %108, i32 3		; <float> [#uses=1]
	%110 = fadd float %tmp19, %tmp17		; <float> [#uses=1]
	%111 = fadd float %109, %110		; <float> [#uses=1]
	%Sum0.254 = fadd float %111, %Sum0.1		; <float> [#uses=2]
	%112 = icmp sgt i32 %N_addr.1, 0		; <i1> [#uses=1]
	br i1 %112, label %bb.nph56, label %bb16

bb.nph56:		; preds = %bb13
	%tmp. = zext i32 %N_addr.1 to i64		; <i64> [#uses=1]
	br label %bb14

bb14:		; preds = %bb14, %bb.nph56
	%indvar117 = phi i64 [ 0, %bb.nph56 ], [ %indvar.next118, %bb14 ]		; <i64> [#uses=3]
	%Sum0.255 = phi float [ %Sum0.254, %bb.nph56 ], [ %Sum0.2, %bb14 ]		; <float> [#uses=1]
	%tmp.122 = sext i32 %IB to i64		; <i64> [#uses=1]
	%B_addr.652.rec = mul i64 %indvar117, %tmp.122		; <i64> [#uses=1]
	%tmp.124 = sext i32 %IA to i64		; <i64> [#uses=1]
	%A_addr.653.rec = mul i64 %indvar117, %tmp.124		; <i64> [#uses=1]
	%B_addr.652 = getelementptr float* %B_addr.1, i64 %B_addr.652.rec		; <float*> [#uses=1]
	%A_addr.653 = getelementptr float* %A_addr.1, i64 %A_addr.653.rec		; <float*> [#uses=1]
	%113 = load float* %A_addr.653, align 4		; <float> [#uses=1]
	%114 = load float* %B_addr.652, align 4		; <float> [#uses=1]
	%115 = fmul float %113, %114		; <float> [#uses=1]
	%Sum0.2 = fadd float %115, %Sum0.255		; <float> [#uses=2]
	%indvar.next118 = add i64 %indvar117, 1		; <i64> [#uses=2]
	%exitcond = icmp eq i64 %indvar.next118, %tmp.		; <i1> [#uses=1]
	br i1 %exitcond, label %bb16, label %bb14

bb16:		; preds = %bb14, %bb13
	%Sum0.2.lcssa = phi float [ %Sum0.254, %bb13 ], [ %Sum0.2, %bb14 ]		; <float> [#uses=1]
	store float %Sum0.2.lcssa, float* %C, align 4
	ret void
}
