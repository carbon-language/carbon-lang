; This testcase was extracted from the gzip SPEC benchmark
;
; RUN: llvm-as < %s | opt -adce

%bk = external global uint		; <uint*> [#uses=2]
%hufts = external global uint		; <uint*> [#uses=1]

implementation   ; Functions:

int %inflate() {
bb0:					;[#uses=1]
	br label %bb2

bb2:					;[#uses=2]
	%reg128 = phi uint [ %reg130, %bb6 ], [ 0, %bb0 ]		; <uint> [#uses=2]
	br bool true, label %bb4, label %bb3

bb3:					;[#uses=2]
	br label %UnifiedExitNode

bb4:					;[#uses=2]
	%reg117 = load uint* %hufts		; <uint> [#uses=2]
	%cond241 = setle uint %reg117, %reg128		; <bool> [#uses=1]
	br bool %cond241, label %bb6, label %bb5

bb5:					;[#uses=2]
	br label %bb6

bb6:					;[#uses=3]
	%reg130 = phi uint [ %reg117, %bb5 ], [ %reg128, %bb4 ]		; <uint> [#uses=1]
	br bool false, label %bb2, label %bb7

bb7:					;[#uses=1]
	%reg126 = load uint* %bk		; <uint> [#uses=1]
	%cond247 = setle uint %reg126, 7		; <bool> [#uses=1]
	br bool %cond247, label %bb9, label %bb8

bb8:					;[#uses=2]
	%reg119 = load uint* %bk		; <uint> [#uses=1]
	%cond256 = setgt uint %reg119, 7		; <bool> [#uses=1]
	br bool %cond256, label %bb8, label %bb9

bb9:					;[#uses=3]
	br label %UnifiedExitNode

UnifiedExitNode:					;[#uses=2]
	%UnifiedRetVal = phi int [ 7, %bb3 ], [ 0, %bb9 ]		; <int> [#uses=1]
	ret int %UnifiedRetVal
}
