; This testcase was extracted from the gzip SPEC benchmark
;
; RUN: as < %s | opt -adce

%inptr = internal uninitialized global uint		; <uint*> [#uses=2]
%outcnt = internal uninitialized global uint		; <uint*> [#uses=1]
%bb = internal uninitialized global ulong		; <ulong*> [#uses=1]
%bk = internal uninitialized global uint		; <uint*> [#uses=5]
%hufts = internal uninitialized global uint		; <uint*> [#uses=2]

implementation   ; Functions:

int %inflate() {
bb0:					;[#uses=0]
	%e = alloca int		; <int*> [#uses=2]
	br label %bb1

bb1:					;[#uses=2]
	store uint 0, uint* %outcnt
	store uint 0, uint* %bk
	store ulong 0, ulong* %bb
	br label %bb2

bb2:					;[#uses=2]
	%reg128 = phi uint [ %reg130, %bb6 ], [ 0, %bb1 ]		; <uint> [#uses=2]
	store uint 0, uint* %hufts
	%reg236 = call int %inflate_block( int* %e )		; <int> [#uses=2]
	%cond237 = seteq int %reg236, 0		; <bool> [#uses=1]
	br bool %cond237, label %bb4, label %bb3

bb3:					;[#uses=1]
	ret int %reg236

bb4:					;[#uses=2]
	%reg117 = load uint* %hufts		; <uint> [#uses=2]
	%cond241 = setle uint %reg117, %reg128		; <bool> [#uses=1]
	br bool %cond241, label %bb6, label %bb5

bb5:					;[#uses=2]
	br label %bb6

bb6:					;[#uses=3]
	%reg130 = phi uint [ %reg117, %bb5 ], [ %reg128, %bb4 ]		; <uint> [#uses=1]
	%reg118 = load int* %e, uint 0		; <int> [#uses=1]
	%cond244 = seteq int %reg118, 0		; <bool> [#uses=1]
	br bool %cond244, label %bb2, label %bb7

bb7:					;[#uses=1]
	%reg126 = load uint* %bk		; <uint> [#uses=1]
	%cond247 = setle uint %reg126, 7		; <bool> [#uses=1]
	br bool %cond247, label %bb9, label %bb8

bb8:					;[#uses=2]
	%reg120 = load uint* %bk		; <uint> [#uses=1]
	%reg121 = add uint %reg120, 4294967288		; <uint> [#uses=1]
	store uint %reg121, uint* %bk
	%reg122 = load uint* %inptr		; <uint> [#uses=1]
	%reg123 = add uint %reg122, 4294967295		; <uint> [#uses=1]
	store uint %reg123, uint* %inptr
	%reg119 = load uint* %bk		; <uint> [#uses=1]
	%cond256 = setgt uint %reg119, 7		; <bool> [#uses=1]
	br bool %cond256, label %bb8, label %bb9

bb9:					;[#uses=2]
	call void %flush_window( )
	ret int 0

bb10:					;[#uses=0]
	ret int 42
}

declare void %flush_window()

declare int %inflate_block(int*)
