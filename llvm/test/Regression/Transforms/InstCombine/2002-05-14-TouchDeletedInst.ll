; This testcase, obviously distilled from a large program (bzip2 from Specint2000)
; caused instcombine to fail because it got the same instruction on it's worklist
; more than once (which is ok), but then deleted the instruction.  Since the inst
; stayed on the worklist, as soon as it came back up to be processed, bad things
; happened, and opt asserted.
;
; RUN: llvm-as < %s | opt -instcombine
;

%.LC0 = internal global [21 x sbyte] c"hbMakeCodeLengths(1)\00"		; <[21 x sbyte]*> [#uses=1]
%.LC1 = internal global [21 x sbyte] c"hbMakeCodeLengths(2)\00"		; <[21 x sbyte]*> [#uses=1]

implementation   ; Functions:

void %hbMakeCodeLengths(ubyte* %len, int* %freq, int %alphaSize, int %maxLen) {
bb0:					;[#uses=0]
	%len = alloca ubyte*		; <ubyte**> [#uses=2]
	store ubyte* %len, ubyte** %len
	%freq = alloca int*		; <int**> [#uses=2]
	store int* %freq, int** %freq
	%alphaSize = alloca int		; <int*> [#uses=2]
	store int %alphaSize, int* %alphaSize
	%maxLen = alloca int		; <int*> [#uses=2]
	store int %maxLen, int* %maxLen
	%heap = alloca int, uint 260		; <int*> [#uses=27]
	%weight = alloca int, uint 516		; <int*> [#uses=18]
	%parent = alloca int, uint 516		; <int*> [#uses=7]
	br label %bb1

bb1:					;[#uses=2]
	%reg107 = load ubyte** %len		; <ubyte*> [#uses=1]
	%reg108 = load int** %freq		; <int*> [#uses=1]
	%reg109 = load int* %alphaSize		; <int> [#uses=10]
	%reg110 = load int* %maxLen		; <int> [#uses=1]
	%cond747 = setge int 0, %reg109		; <bool> [#uses=1]
	br bool %cond747, label %bb6, label %bb2

bb2:					;[#uses=2]
	%reg591 = phi int [ %reg594, %bb5 ], [ 0, %bb1 ]		; <int> [#uses=3]
	%reg591-idxcast1 = cast int %reg591 to uint		; <uint> [#uses=1]
	%reg591-idxcast1-offset = add uint %reg591-idxcast1, 1		; <uint> [#uses=1]
	%reg591-idxcast1-offset = cast uint %reg591-idxcast1-offset to long		; <long> [#uses=1]
	%reg126 = getelementptr int* %weight, long %reg591-idxcast1-offset		; <int*> [#uses=1]
	%reg591-idxcast = cast int %reg591 to long		; <long> [#uses=1]
	%reg132 = getelementptr int* %reg108, long %reg591-idxcast		; <int*> [#uses=1]
	%reg133 = load int* %reg132		; <int> [#uses=2]
	%cond748 = seteq int %reg133, 0		; <bool> [#uses=1]
	br bool %cond748, label %bb4, label %bb3

bb3:					;[#uses=2]
	%reg127 = shl int %reg133, ubyte 8		; <int> [#uses=1]
	br label %bb5

bb4:					;[#uses=2]
	br label %bb5

bb5:					;[#uses=3]
	%reg593 = phi int [ 256, %bb4 ], [ %reg127, %bb3 ]		; <int> [#uses=1]
	store int %reg593, int* %reg126
	%reg594 = add int %reg591, 1		; <int> [#uses=2]
	%cond749 = setlt int %reg594, %reg109		; <bool> [#uses=1]
	br bool %cond749, label %bb2, label %bb6

bb6:					;[#uses=6]
	store int 0, int* %heap
	store int 0, int* %weight
	store int -2, int* %parent
	%cond750 = setgt int 1, %reg109		; <bool> [#uses=1]
	br bool %cond750, label %bb11, label %bb7

bb7:					;[#uses=3]
	%reg597 = phi uint [ %reg598, %bb10 ], [ 0, %bb6 ]		; <uint> [#uses=5]
	%reg597-casted = cast uint %reg597 to int		; <int> [#uses=1]
	%reg596 = add int %reg597-casted, 1		; <int> [#uses=3]
	%reg597-offset = add uint %reg597, 1		; <uint> [#uses=1]
	%reg597-offset = cast uint %reg597-offset to long		; <long> [#uses=1]
	%reg149 = getelementptr int* %parent, long %reg597-offset		; <int*> [#uses=1]
	store int -1, int* %reg149
	%reg598 = add uint %reg597, 1		; <uint> [#uses=3]
	%reg597-offset1 = add uint %reg597, 1		; <uint> [#uses=1]
	%reg597-offset1 = cast uint %reg597-offset1 to long		; <long> [#uses=1]
	%reg157 = getelementptr int* %heap, long %reg597-offset1		; <int*> [#uses=1]
	store int %reg596, int* %reg157
	br label %bb9

bb8:					;[#uses=2]
	%reg599 = cast uint %reg599 to long		; <long> [#uses=1]
	%reg198 = getelementptr int* %heap, long %reg599		; <int*> [#uses=1]
	store int %reg182, int* %reg198
	%cast938 = cast int %reg174 to uint		; <uint> [#uses=1]
	br label %bb9

bb9:					;[#uses=2]
	%reg599 = phi uint [ %cast938, %bb8 ], [ %reg598, %bb7 ]		; <uint> [#uses=3]
	%cast807 = cast uint %reg599 to int		; <int> [#uses=1]
	%reg597-offset2 = add uint %reg597, 1		; <uint> [#uses=1]
	%reg597-offset2 = cast uint %reg597-offset2 to long		; <long> [#uses=1]
	%reg173 = getelementptr int* %weight, long %reg597-offset2		; <int*> [#uses=1]
	%reg174 = shr int %cast807, ubyte 1		; <int> [#uses=2]
	%reg174-idxcast = cast int %reg174 to uint		; <uint> [#uses=1]
	cast uint %reg174-idxcast to long		; <long>:0 [#uses=1]
	%reg181 = getelementptr int* %heap, long %0		; <int*> [#uses=1]
	%reg182 = load int* %reg181		; <int> [#uses=2]
	%reg182-idxcast = cast int %reg182 to uint		; <uint> [#uses=1]
	cast uint %reg182-idxcast to long		; <long>:1 [#uses=1]
	%reg189 = getelementptr int* %weight, long %1		; <int*> [#uses=1]
	%reg190 = load int* %reg173		; <int> [#uses=1]
	%reg191 = load int* %reg189		; <int> [#uses=1]
	%cond751 = setlt int %reg190, %reg191		; <bool> [#uses=1]
	br bool %cond751, label %bb8, label %bb10

bb10:					;[#uses=3]
	cast uint %reg599 to long		; <long>:2 [#uses=1]
	%reg214 = getelementptr int* %heap, long %2		; <int*> [#uses=1]
	store int %reg596, int* %reg214
	%reg601 = add int %reg596, 1		; <int> [#uses=1]
	%cond752 = setle int %reg601, %reg109		; <bool> [#uses=1]
	br bool %cond752, label %bb7, label %bb11

bb11:					;[#uses=2]
	%reg602 = phi uint [ %reg598, %bb10 ], [ 0, %bb6 ]		; <uint> [#uses=3]
	%cast819 = cast uint %reg602 to int		; <int> [#uses=1]
	%cast818 = cast uint %reg602 to int		; <int> [#uses=1]
	%cond753 = setle int %cast818, 259		; <bool> [#uses=1]
	br bool %cond753, label %bb13, label %bb12

bb12:					;[#uses=1]
	cast uint 0 to long		; <long>:3 [#uses=1]
	cast uint 0 to long		; <long>:4 [#uses=1]
	%cast784 = getelementptr [21 x sbyte]* %.LC0, long %3, long %4		; <sbyte*> [#uses=1]
	call void %panic( sbyte* %cast784 )
	br label %bb13

bb13:					;[#uses=4]
	%cond754 = setle int %cast819, 1		; <bool> [#uses=1]
	%cast918 = cast int %reg109 to uint		; <uint> [#uses=1]
	%cast940 = cast uint %reg602 to int		; <int> [#uses=1]
	%cast942 = cast int %reg109 to uint		; <uint> [#uses=1]
	br bool %cond754, label %bb32, label %bb14

bb14:					;[#uses=5]
	%cann-indvar1 = phi uint [ 0, %bb13 ], [ %add1-indvar1, %bb31 ]		; <uint> [#uses=3]
	%cann-indvar1-casted = cast uint %cann-indvar1 to int		; <int> [#uses=1]
	%reg603-scale = mul int %cann-indvar1-casted, -1		; <int> [#uses=1]
	%reg603 = add int %reg603-scale, %cast940		; <int> [#uses=4]
	%reg604 = add uint %cann-indvar1, %cast942		; <uint> [#uses=4]
	%add1-indvar1 = add uint %cann-indvar1, 1		; <uint> [#uses=1]
	cast uint 1 to long		; <long>:5 [#uses=1]
	%reg7551 = getelementptr int* %heap, long %5		; <int*> [#uses=1]
	%reg113 = load int* %reg7551		; <int> [#uses=2]
	%reg603-idxcast = cast int %reg603 to uint		; <uint> [#uses=1]
	cast uint %reg603-idxcast to long		; <long>:6 [#uses=1]
	%reg222 = getelementptr int* %heap, long %6		; <int*> [#uses=1]
	%reg223 = load int* %reg222		; <int> [#uses=1]
	cast uint 1 to long		; <long>:7 [#uses=1]
	%reg7561 = getelementptr int* %heap, long %7		; <int*> [#uses=1]
	store int %reg223, int* %reg7561
	%reg605 = add int %reg603, -1		; <int> [#uses=4]
	cast uint 1 to long		; <long>:8 [#uses=1]
	%reg757 = getelementptr int* %heap, long %8		; <int*> [#uses=1]
	%reg226 = load int* %reg757		; <int> [#uses=2]
	%cond758 = setgt int 2, %reg605		; <bool> [#uses=1]
	br bool %cond758, label %bb20, label %bb15

bb15:					;[#uses=3]
	%reg606 = phi int [ %reg611, %bb19 ], [ 2, %bb14 ]		; <int> [#uses=6]
	%reg607 = phi int [ %reg609, %bb19 ], [ 1, %bb14 ]		; <int> [#uses=2]
	%cond759 = setge int %reg606, %reg605		; <bool> [#uses=1]
	br bool %cond759, label %bb18, label %bb16

bb16:					;[#uses=2]
	%reg606-idxcast = cast int %reg606 to uint		; <uint> [#uses=1]
	%reg606-idxcast-offset = add uint %reg606-idxcast, 1		; <uint> [#uses=1]
	cast uint %reg606-idxcast-offset to long		; <long>:9 [#uses=1]
	%reg241 = getelementptr int* %heap, long %9		; <int*> [#uses=1]
	%reg242 = load int* %reg241		; <int> [#uses=1]
	%reg242-idxcast = cast int %reg242 to uint		; <uint> [#uses=1]
	cast uint %reg242-idxcast to long		; <long>:10 [#uses=1]
	%reg249 = getelementptr int* %weight, long %10		; <int*> [#uses=1]
	%reg606-idxcast1 = cast int %reg606 to uint		; <uint> [#uses=1]
	cast uint %reg606-idxcast1 to long		; <long>:11 [#uses=1]
	%reg256 = getelementptr int* %heap, long %11		; <int*> [#uses=1]
	%reg257 = load int* %reg256		; <int> [#uses=1]
	%reg257-idxcast = cast int %reg257 to uint		; <uint> [#uses=1]
	cast uint %reg257-idxcast to long		; <long>:12 [#uses=1]
	%reg264 = getelementptr int* %weight, long %12		; <int*> [#uses=1]
	%reg265 = load int* %reg249		; <int> [#uses=1]
	%reg266 = load int* %reg264		; <int> [#uses=1]
	%cond760 = setge int %reg265, %reg266		; <bool> [#uses=1]
	br bool %cond760, label %bb18, label %bb17

bb17:					;[#uses=2]
	%reg608 = add int %reg606, 1		; <int> [#uses=1]
	br label %bb18

bb18:					;[#uses=4]
	%reg609 = phi int [ %reg608, %bb17 ], [ %reg606, %bb16 ], [ %reg606, %bb15 ]		; <int> [#uses=4]
	%reg226-idxcast = cast int %reg226 to uint		; <uint> [#uses=1]
	cast uint %reg226-idxcast to long		; <long>:13 [#uses=1]
	%reg273 = getelementptr int* %weight, long %13		; <int*> [#uses=1]
	%reg609-idxcast = cast int %reg609 to uint		; <uint> [#uses=1]
	cast uint %reg609-idxcast to long		; <long>:14 [#uses=1]
	%reg280 = getelementptr int* %heap, long %14		; <int*> [#uses=1]
	%reg281 = load int* %reg280		; <int> [#uses=2]
	%reg281-idxcast = cast int %reg281 to uint		; <uint> [#uses=1]
	cast uint %reg281-idxcast to long		; <long>:15 [#uses=1]
	%reg288 = getelementptr int* %weight, long %15		; <int*> [#uses=1]
	%reg289 = load int* %reg273		; <int> [#uses=1]
	%reg290 = load int* %reg288		; <int> [#uses=1]
	%cond761 = setlt int %reg289, %reg290		; <bool> [#uses=1]
	br bool %cond761, label %bb20, label %bb19

bb19:					;[#uses=4]
	%reg607-idxcast = cast int %reg607 to uint		; <uint> [#uses=1]
	cast uint %reg607-idxcast to long		; <long>:16 [#uses=1]
	%reg297 = getelementptr int* %heap, long %16		; <int*> [#uses=1]
	store int %reg281, int* %reg297
	%reg611 = shl int %reg609, ubyte 1		; <int> [#uses=2]
	%cond762 = setle int %reg611, %reg605		; <bool> [#uses=1]
	br bool %cond762, label %bb15, label %bb20

bb20:					;[#uses=6]
	%reg612 = phi int [ %reg609, %bb19 ], [ %reg607, %bb18 ], [ 1, %bb14 ]		; <int> [#uses=1]
	%reg612-idxcast = cast int %reg612 to uint		; <uint> [#uses=1]
	cast uint %reg612-idxcast to long		; <long>:17 [#uses=1]
	%reg312 = getelementptr int* %heap, long %17		; <int*> [#uses=1]
	store int %reg226, int* %reg312
	cast uint 1 to long		; <long>:18 [#uses=1]
	%reg7631 = getelementptr int* %heap, long %18		; <int*> [#uses=1]
	%reg114 = load int* %reg7631		; <int> [#uses=2]
	%reg603-idxcast1 = cast int %reg603 to uint		; <uint> [#uses=1]
	%reg603-idxcast1-offset = add uint %reg603-idxcast1, 1073741823		; <uint> [#uses=1]
	cast uint %reg603-idxcast1-offset to long		; <long>:19 [#uses=1]
	%reg319 = getelementptr int* %heap, long %19		; <int*> [#uses=1]
	%reg320 = load int* %reg319		; <int> [#uses=1]
	cast uint 1 to long		; <long>:20 [#uses=1]
	%reg7641 = getelementptr int* %heap, long %20		; <int*> [#uses=1]
	store int %reg320, int* %reg7641
	%reg613 = add int %reg605, -1		; <int> [#uses=4]
	cast uint 1 to long		; <long>:21 [#uses=1]
	%reg765 = getelementptr int* %heap, long %21		; <int*> [#uses=1]
	%reg323 = load int* %reg765		; <int> [#uses=2]
	%cond766 = setgt int 2, %reg613		; <bool> [#uses=1]
	br bool %cond766, label %bb26, label %bb21

bb21:					;[#uses=3]
	%reg614 = phi int [ %reg619, %bb25 ], [ 2, %bb20 ]		; <int> [#uses=6]
	%reg615 = phi int [ %reg617, %bb25 ], [ 1, %bb20 ]		; <int> [#uses=2]
	%cond767 = setge int %reg614, %reg613		; <bool> [#uses=1]
	br bool %cond767, label %bb24, label %bb22

bb22:					;[#uses=2]
	%reg614-idxcast = cast int %reg614 to uint		; <uint> [#uses=1]
	%reg614-idxcast-offset = add uint %reg614-idxcast, 1		; <uint> [#uses=1]
	cast uint %reg614-idxcast-offset to long		; <long>:22 [#uses=1]
	%reg338 = getelementptr int* %heap, long %22		; <int*> [#uses=1]
	%reg339 = load int* %reg338		; <int> [#uses=1]
	%reg339-idxcast = cast int %reg339 to uint		; <uint> [#uses=1]
	cast uint %reg339-idxcast to long		; <long>:23 [#uses=1]
	%reg346 = getelementptr int* %weight, long %23		; <int*> [#uses=1]
	%reg614-idxcast1 = cast int %reg614 to uint		; <uint> [#uses=1]
	cast uint %reg614-idxcast1 to long		; <long>:24 [#uses=1]
	%reg353 = getelementptr int* %heap, long %24		; <int*> [#uses=1]
	%reg354 = load int* %reg353		; <int> [#uses=1]
	%reg354-idxcast = cast int %reg354 to uint		; <uint> [#uses=1]
	cast uint %reg354-idxcast to long		; <long>:25 [#uses=1]
	%reg361 = getelementptr int* %weight, long %25		; <int*> [#uses=1]
	%reg362 = load int* %reg346		; <int> [#uses=1]
	%reg363 = load int* %reg361		; <int> [#uses=1]
	%cond768 = setge int %reg362, %reg363		; <bool> [#uses=1]
	br bool %cond768, label %bb24, label %bb23

bb23:					;[#uses=2]
	%reg616 = add int %reg614, 1		; <int> [#uses=1]
	br label %bb24

bb24:					;[#uses=4]
	%reg617 = phi int [ %reg616, %bb23 ], [ %reg614, %bb22 ], [ %reg614, %bb21 ]		; <int> [#uses=4]
	%reg323-idxcast = cast int %reg323 to uint		; <uint> [#uses=1]
	cast uint %reg323-idxcast to long		; <long>:26 [#uses=1]
	%reg370 = getelementptr int* %weight, long %26		; <int*> [#uses=1]
	%reg617-idxcast = cast int %reg617 to uint		; <uint> [#uses=1]
	cast uint %reg617-idxcast to long		; <long>:27 [#uses=1]
	%reg377 = getelementptr int* %heap, long %27		; <int*> [#uses=1]
	%reg378 = load int* %reg377		; <int> [#uses=2]
	%reg378-idxcast = cast int %reg378 to uint		; <uint> [#uses=1]
	cast uint %reg378-idxcast to long		; <long>:28 [#uses=1]
	%reg385 = getelementptr int* %weight, long %28		; <int*> [#uses=1]
	%reg386 = load int* %reg370		; <int> [#uses=1]
	%reg387 = load int* %reg385		; <int> [#uses=1]
	%cond769 = setlt int %reg386, %reg387		; <bool> [#uses=1]
	br bool %cond769, label %bb26, label %bb25

bb25:					;[#uses=4]
	%reg615-idxcast = cast int %reg615 to uint		; <uint> [#uses=1]
	cast uint %reg615-idxcast to long		; <long>:29 [#uses=1]
	%reg394 = getelementptr int* %heap, long %29		; <int*> [#uses=1]
	store int %reg378, int* %reg394
	%reg619 = shl int %reg617, ubyte 1		; <int> [#uses=2]
	%cond770 = setle int %reg619, %reg613		; <bool> [#uses=1]
	br bool %cond770, label %bb21, label %bb26

bb26:					;[#uses=4]
	%reg620 = phi int [ %reg617, %bb25 ], [ %reg615, %bb24 ], [ 1, %bb20 ]		; <int> [#uses=1]
	%reg620-idxcast = cast int %reg620 to uint		; <uint> [#uses=1]
	cast uint %reg620-idxcast to long		; <long>:30 [#uses=1]
	%reg409 = getelementptr int* %heap, long %30		; <int*> [#uses=1]
	store int %reg323, int* %reg409
	%reg621 = add uint %reg604, 1		; <uint> [#uses=5]
	%reg113-idxcast = cast int %reg113 to uint		; <uint> [#uses=1]
	cast uint %reg113-idxcast to long		; <long>:31 [#uses=1]
	%reg416 = getelementptr int* %parent, long %31		; <int*> [#uses=1]
	%reg114-idxcast = cast int %reg114 to uint		; <uint> [#uses=1]
	cast uint %reg114-idxcast to long		; <long>:32 [#uses=1]
	%reg423 = getelementptr int* %parent, long %32		; <int*> [#uses=1]
	%cast889 = cast uint %reg621 to int		; <int> [#uses=1]
	store int %cast889, int* %reg423
	%cast890 = cast uint %reg621 to int		; <int> [#uses=1]
	store int %cast890, int* %reg416
	%reg604-offset = add uint %reg604, 1		; <uint> [#uses=1]
	cast uint %reg604-offset to long		; <long>:33 [#uses=1]
	%reg431 = getelementptr int* %weight, long %33		; <int*> [#uses=1]
	%reg113-idxcast2 = cast int %reg113 to uint		; <uint> [#uses=1]
	cast uint %reg113-idxcast2 to long		; <long>:34 [#uses=1]
	%reg4381 = getelementptr int* %weight, long %34		; <int*> [#uses=1]
	%reg439 = load int* %reg4381		; <int> [#uses=2]
	%reg440 = and int %reg439, -256		; <int> [#uses=1]
	%reg114-idxcast2 = cast int %reg114 to uint		; <uint> [#uses=1]
	cast uint %reg114-idxcast2 to long		; <long>:35 [#uses=1]
	%reg4471 = getelementptr int* %weight, long %35		; <int*> [#uses=1]
	%reg448 = load int* %reg4471		; <int> [#uses=2]
	%reg449 = and int %reg448, -256		; <int> [#uses=1]
	%reg450 = add int %reg440, %reg449		; <int> [#uses=1]
	%reg460 = and int %reg439, 255		; <int> [#uses=2]
	%reg451 = and int %reg448, 255		; <int> [#uses=2]
	%cond771 = setge int %reg451, %reg460		; <bool> [#uses=1]
	br bool %cond771, label %bb28, label %bb27

bb27:					;[#uses=2]
	br label %bb28

bb28:					;[#uses=3]
	%reg623 = phi int [ %reg460, %bb27 ], [ %reg451, %bb26 ]		; <int> [#uses=1]
	%reg469 = add int %reg623, 1		; <int> [#uses=1]
	%reg470 = or int %reg450, %reg469		; <int> [#uses=1]
	store int %reg470, int* %reg431
	%reg604-offset1 = add uint %reg604, 1		; <uint> [#uses=1]
	cast uint %reg604-offset1 to long		; <long>:36 [#uses=1]
	%reg4771 = getelementptr int* %parent, long %36		; <int*> [#uses=1]
	store int -1, int* %reg4771
	%reg624 = add int %reg613, 1		; <int> [#uses=2]
	%reg603-idxcast2 = cast int %reg603 to uint		; <uint> [#uses=1]
	%reg603-idxcast2-offset = add uint %reg603-idxcast2, 1073741823		; <uint> [#uses=1]
	cast uint %reg603-idxcast2-offset to long		; <long>:37 [#uses=1]
	%reg485 = getelementptr int* %heap, long %37		; <int*> [#uses=1]
	%cast902 = cast uint %reg621 to int		; <int> [#uses=1]
	store int %cast902, int* %reg485
	br label %bb30

bb29:					;[#uses=2]
	%reg625-idxcast = cast int %reg625 to uint		; <uint> [#uses=1]
	cast uint %reg625-idxcast to long		; <long>:38 [#uses=1]
	%reg526 = getelementptr int* %heap, long %38		; <int*> [#uses=1]
	store int %reg510, int* %reg526
	br label %bb30

bb30:					;[#uses=2]
	%reg625 = phi int [ %reg502, %bb29 ], [ %reg624, %bb28 ]		; <int> [#uses=3]
	%reg604-offset2 = add uint %reg604, 1		; <uint> [#uses=1]
	cast uint %reg604-offset2 to long		; <long>:39 [#uses=1]
	%reg501 = getelementptr int* %weight, long %39		; <int*> [#uses=1]
	%reg502 = shr int %reg625, ubyte 1		; <int> [#uses=2]
	%reg502-idxcast = cast int %reg502 to uint		; <uint> [#uses=1]
	cast uint %reg502-idxcast to long		; <long>:40 [#uses=1]
	%reg509 = getelementptr int* %heap, long %40		; <int*> [#uses=1]
	%reg510 = load int* %reg509		; <int> [#uses=2]
	%reg510-idxcast = cast int %reg510 to uint		; <uint> [#uses=1]
	cast uint %reg510-idxcast to long		; <long>:41 [#uses=1]
	%reg517 = getelementptr int* %weight, long %41		; <int*> [#uses=1]
	%reg518 = load int* %reg501		; <int> [#uses=1]
	%reg519 = load int* %reg517		; <int> [#uses=1]
	%cond772 = setlt int %reg518, %reg519		; <bool> [#uses=1]
	br bool %cond772, label %bb29, label %bb31

bb31:					;[#uses=3]
	%reg625-idxcast1 = cast int %reg625 to uint		; <uint> [#uses=1]
	cast uint %reg625-idxcast1 to long		; <long>:42 [#uses=1]
	%reg542 = getelementptr int* %heap, long %42		; <int*> [#uses=1]
	%cast916 = cast uint %reg621 to int		; <int> [#uses=1]
	store int %cast916, int* %reg542
	%cond773 = setgt int %reg624, 1		; <bool> [#uses=1]
	br bool %cond773, label %bb14, label %bb32

bb32:					;[#uses=2]
	%reg627 = phi uint [ %reg621, %bb31 ], [ %cast918, %bb13 ]		; <uint> [#uses=1]
	%cast919 = cast uint %reg627 to int		; <int> [#uses=1]
	%cond774 = setle int %cast919, 515		; <bool> [#uses=1]
	br bool %cond774, label %bb34, label %bb33

bb33:					;[#uses=1]
	cast uint 0 to long		; <long>:43 [#uses=1]
	cast uint 0 to long		; <long>:44 [#uses=1]
	%cast785 = getelementptr [21 x sbyte]* %.LC1, long %43, long %44		; <sbyte*> [#uses=1]
	call void %panic( sbyte* %cast785 )
	br label %bb34

bb34:					;[#uses=5]
	%cond775 = setgt int 1, %reg109		; <bool> [#uses=1]
	br bool %cond775, label %bb40, label %bb35

bb35:					;[#uses=5]
	%reg629 = phi ubyte [ %reg639, %bb39 ], [ 0, %bb34 ]		; <ubyte> [#uses=1]
	%cann-indvar = phi uint [ 0, %bb34 ], [ %add1-indvar, %bb39 ]		; <uint> [#uses=4]
	%cann-indvar-casted = cast uint %cann-indvar to int		; <int> [#uses=1]
	%reg630 = add int %cann-indvar-casted, 1		; <int> [#uses=2]
	%add1-indvar = add uint %cann-indvar, 1		; <uint> [#uses=1]
	%cann-indvar-offset1 = add uint %cann-indvar, 1		; <uint> [#uses=1]
	cast uint %cann-indvar-offset1 to long		; <long>:45 [#uses=1]
	%reg589 = getelementptr int* %parent, long %45		; <int*> [#uses=1]
	%reg590 = load int* %reg589		; <int> [#uses=1]
	%cond776 = setlt int %reg590, 0		; <bool> [#uses=1]
	%parent-idxcast = cast int* %parent to uint		; <uint> [#uses=1]
	%cast948 = cast int %reg630 to uint		; <uint> [#uses=1]
	br bool %cond776, label %bb37, label %bb36

bb36:					;[#uses=5]
	%reg632 = phi uint [ %reg634, %bb36 ], [ %cast948, %bb35 ]		; <uint> [#uses=1]
	%reg633 = phi uint [ %reg635, %bb36 ], [ 0, %bb35 ]		; <uint> [#uses=3]
	%reg633-casted = cast uint %reg633 to sbyte*		; <sbyte*> [#uses=0]
	%reg631-scale = mul uint %reg633, 0		; <uint> [#uses=1]
	%reg631-scale = cast uint %reg631-scale to sbyte*		; <sbyte*> [#uses=1]
	cast uint %parent-idxcast to long		; <long>:46 [#uses=1]
	%reg6311 = getelementptr sbyte* %reg631-scale, long %46		; <sbyte*> [#uses=2]
	%reg632-scale = mul uint %reg632, 4		; <uint> [#uses=1]
	cast uint %reg632-scale to long		; <long>:47 [#uses=1]
	%reg5581 = getelementptr sbyte* %reg6311, long %47		; <sbyte*> [#uses=1]
	%cast924 = cast sbyte* %reg5581 to uint*		; <uint*> [#uses=1]
	%reg634 = load uint* %cast924		; <uint> [#uses=2]
	%reg635 = add uint %reg633, 1		; <uint> [#uses=2]
	%reg634-scale = mul uint %reg634, 4		; <uint> [#uses=1]
	cast uint %reg634-scale to long		; <long>:48 [#uses=1]
	%reg5501 = getelementptr sbyte* %reg6311, long %48		; <sbyte*> [#uses=1]
	%cast925 = cast sbyte* %reg5501 to int*		; <int*> [#uses=1]
	%reg551 = load int* %cast925		; <int> [#uses=1]
	%cond777 = setge int %reg551, 0		; <bool> [#uses=1]
	br bool %cond777, label %bb36, label %bb37

bb37:					;[#uses=3]
	%reg637 = phi uint [ %reg635, %bb36 ], [ 0, %bb35 ]		; <uint> [#uses=2]
	%cast928 = cast uint %reg637 to int		; <int> [#uses=1]
	%cann-indvar-offset = add uint %cann-indvar, 1		; <uint> [#uses=1]
	cast uint %cann-indvar-offset to long		; <long>:49 [#uses=1]
	%reg561 = getelementptr ubyte* %reg107, long %49		; <ubyte*> [#uses=1]
	cast uint 4294967295 to long		; <long>:50 [#uses=1]
	%reg778 = getelementptr ubyte* %reg561, long %50		; <ubyte*> [#uses=1]
	%cast788 = cast uint %reg637 to ubyte		; <ubyte> [#uses=1]
	store ubyte %cast788, ubyte* %reg778
	%cond779 = setle int %cast928, %reg110		; <bool> [#uses=1]
	br bool %cond779, label %bb39, label %bb38

bb38:					;[#uses=2]
	br label %bb39

bb39:					;[#uses=5]
	%reg639 = phi ubyte [ 1, %bb38 ], [ %reg629, %bb37 ]		; <ubyte> [#uses=2]
	%reg640 = add int %reg630, 1		; <int> [#uses=1]
	%cond780 = setle int %reg640, %reg109		; <bool> [#uses=1]
	br bool %cond780, label %bb35, label %bb40

bb40:					;[#uses=2]
	%reg641 = phi ubyte [ %reg639, %bb39 ], [ 0, %bb34 ]		; <ubyte> [#uses=1]
	%cond781 = seteq ubyte %reg641, 0		; <bool> [#uses=1]
	br bool %cond781, label %bb44, label %bb41

bb41:					;[#uses=2]
	%cond782 = setge int 1, %reg109		; <bool> [#uses=1]
	br bool %cond782, label %bb6, label %bb42

bb42:					;[#uses=3]
	%cann-indvar2 = phi int [ 0, %bb41 ], [ %add1-indvar2, %bb42 ]		; <int> [#uses=3]
	%reg643 = add int %cann-indvar2, 1		; <int> [#uses=1]
	%add1-indvar2 = add int %cann-indvar2, 1		; <int> [#uses=1]
	%cann-indvar2-idxcast = cast int %cann-indvar2 to uint		; <uint> [#uses=1]
	%cann-indvar2-idxcast-offset = add uint %cann-indvar2-idxcast, 1		; <uint> [#uses=1]
	cast uint %cann-indvar2-idxcast-offset to long		; <long>:51 [#uses=1]
	%reg569 = getelementptr int* %weight, long %51		; <int*> [#uses=2]
	%reg570 = load int* %reg569		; <int> [#uses=2]
	%reg644 = shr int %reg570, ubyte 8		; <int> [#uses=1]
	%reg572 = shr int %reg570, ubyte 31		; <int> [#uses=1]
	%cast933 = cast int %reg572 to uint		; <uint> [#uses=1]
	%reg573 = shr uint %cast933, ubyte 31		; <uint> [#uses=1]
	%cast934 = cast uint %reg573 to int		; <int> [#uses=1]
	%reg574 = add int %reg644, %cast934		; <int> [#uses=1]
	%reg571 = shr int %reg574, ubyte 1		; <int> [#uses=1]
	%reg645 = add int %reg571, 1		; <int> [#uses=1]
	%reg582 = shl int %reg645, ubyte 8		; <int> [#uses=1]
	store int %reg582, int* %reg569
	%reg646 = add int %reg643, 1		; <int> [#uses=1]
	%cond783 = setlt int %reg646, %reg109		; <bool> [#uses=1]
	br bool %cond783, label %bb42, label %bb43

bb43:					;[#uses=1]
	br label %bb6

bb44:					;[#uses=1]
	ret void
}

declare void %panic(sbyte*)
