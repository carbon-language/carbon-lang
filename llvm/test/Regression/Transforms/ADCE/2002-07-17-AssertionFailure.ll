; Testcase unanalyzed, not checked in yet.
; RUN: as < %s | opt -adce	

	%FILE = type { [16 x long] }
	%arc_t = type { %node_t*, %node_t*, %arc_t*, %arc_t*, long, long, long, long }
	%network_t = type { [200 x sbyte], [200 x sbyte], long, long, long, long, long, long, long, long, long, long, long, long, long, long, long, double, long, %node_t*, %node_t*, %arc_t*, %arc_t*, %arc_t*, %arc_t*, long, long, long }
	%node_t = type { long, sbyte*, %node_t*, %node_t*, %node_t*, %node_t*, long, long, %arc_t*, %arc_t*, %arc_t*, long, long, long, long }
	%struct.__FILE = type { [16 x long] }
	%struct.arc = type { %node_t*, %node_t*, %arc_t*, %arc_t*, long, long, long, long }
	%struct.network = type { [200 x sbyte], [200 x sbyte], long, long, long, long, long, long, long, long, long, long, long, long, long, long, long, double, long, %node_t*, %node_t*, %arc_t*, %arc_t*, %arc_t*, %arc_t*, long, long, long }
	%struct.node = type { long, sbyte*, %node_t*, %node_t*, %node_t*, %node_t*, long, long, %arc_t*, %arc_t*, %arc_t*, long, long, long, long }
%.LC0 = internal global [2 x sbyte] c"r\00"		; <[2 x sbyte]*> [#uses=1]
%.LC1 = internal global [8 x sbyte] c"%ld %ld\00"		; <[8 x sbyte]*> [#uses=1]
%.LC2 = internal global [30 x sbyte] c"read_min(): not enough memory\00"		; <[30 x sbyte]*> [#uses=1]
%.LC3 = internal global [12 x sbyte] c"%ld %ld %ld\00"		; <[12 x sbyte]*> [#uses=1]

implementation   ; Functions:

long %read_min(%network_t* %net) {
bb0:					;[#uses=0]
	%instring = alloca sbyte, uint 201		; <sbyte*> [#uses=6]
	%t = alloca long		; <long*> [#uses=11]
	%h = alloca long		; <long*> [#uses=8]
	%c = alloca long		; <long*> [#uses=3]
	br label %bb1

bb1:					;[#uses=1]
	%cast518 = getelementptr %network_t* %net, uint 0, ubyte 0, uint 0		; <sbyte*> [#uses=15]
	%cast1028 = getelementptr [2 x sbyte]* %.LC0, uint 0, uint 0		; <sbyte*> [#uses=1]
	%reg520 = call %FILE* %fopen( sbyte* %cast518, sbyte* %cast1028 )		; <%FILE*> [#uses=5]
	%cond522 = setne %FILE* %reg520, null		; <bool> [#uses=1]
	br bool %cond522, label %bb3, label %bb2

bb2:					;[#uses=1]
	ret long -1

bb3:					;[#uses=1]
	%reg526 = call sbyte* %fgets( sbyte* %instring, int 200, %FILE* %reg520 )		; <sbyte*> [#uses=0]
	%cast1029 = getelementptr [8 x sbyte]* %.LC1, uint 0, uint 0		; <sbyte*> [#uses=2]
	%reg530 = call int (sbyte*, sbyte*, ...)* %sscanf( sbyte* %instring, sbyte* %cast1029, long* %t, long* %h )		; <int> [#uses=1]
	%cond534 = seteq int %reg530, 2		; <bool> [#uses=1]
	br bool %cond534, label %bb5, label %bb4

bb4:					;[#uses=1]
	ret long -1

bb5:					;[#uses=1]
	%reg129 = load long* %t, uint 0		; <long> [#uses=1]
	%reg5381 = getelementptr sbyte* %cast518, uint 408		; <sbyte*> [#uses=1]
	%cast540 = cast sbyte* %reg5381 to long*		; <long*> [#uses=12]
	store long %reg129, long* %cast540
	%reg130 = load long* %h, uint 0		; <long> [#uses=1]
	%reg5421 = getelementptr sbyte* %cast518, uint 432		; <sbyte*> [#uses=2]
	%cast544 = cast sbyte* %reg5421 to long*		; <long*> [#uses=3]
	store long %reg130, long* %cast544
	%reg131 = load long* %t, uint 0		; <long> [#uses=1]
	%reg132 = load long* %t		; <long> [#uses=1]
	%reg133 = add long %reg131, %reg132		; <long> [#uses=1]
	%reg134 = add long %reg133, 1		; <long> [#uses=1]
	%reg5481 = getelementptr sbyte* %cast518, uint 400		; <sbyte*> [#uses=3]
	%cast550 = cast sbyte* %reg5481 to long*		; <long*> [#uses=3]
	store long %reg134, long* %cast550
	%reg135 = load long* %t, uint 0		; <long> [#uses=1]
	%reg136 = load long* %t		; <long> [#uses=2]
	%reg137 = add long %reg135, %reg136		; <long> [#uses=1]
	%reg139 = add long %reg137, %reg136		; <long> [#uses=1]
	%reg140 = load long* %h		; <long> [#uses=1]
	%reg141 = add long %reg139, %reg140		; <long> [#uses=1]
	%reg5551 = getelementptr sbyte* %cast518, uint 424		; <sbyte*> [#uses=3]
	%cast557 = cast sbyte* %reg5551 to long*		; <long*> [#uses=1]
	store long %reg141, long* %cast557
	%reg5581 = getelementptr sbyte* %cast518, uint 416		; <sbyte*> [#uses=1]
	%cast560 = cast sbyte* %reg5581 to ulong*		; <ulong*> [#uses=2]
	store ulong 3014656, ulong* %cast560
	%cast563 = cast sbyte* %reg5481 to ulong*		; <ulong*> [#uses=2]
	%reg143 = load ulong* %cast563		; <ulong> [#uses=1]
	%reg144 = add ulong %reg143, 1		; <ulong> [#uses=1]
	%reg566 = call sbyte* %calloc( ulong %reg144, ulong 120 )		; <sbyte*> [#uses=1]
	%reg5681 = getelementptr sbyte* %cast518, uint 536		; <sbyte*> [#uses=2]
	%cast570 = cast sbyte* %reg5681 to sbyte**		; <sbyte**> [#uses=2]
	store sbyte* %reg566, sbyte** %cast570
	%reg148 = load ulong* %cast563		; <ulong> [#uses=1]
	%reg575 = call sbyte* %calloc( ulong %reg148, ulong 64 )		; <sbyte*> [#uses=1]
	%reg5771 = getelementptr sbyte* %cast518, uint 568		; <sbyte*> [#uses=1]
	%cast579 = cast sbyte* %reg5771 to sbyte**		; <sbyte**> [#uses=3]
	store sbyte* %reg575, sbyte** %cast579
	%reg152 = load ulong* %cast560		; <ulong> [#uses=1]
	%reg584 = call sbyte* %calloc( ulong %reg152, ulong 64 )		; <sbyte*> [#uses=1]
	%reg5861 = getelementptr sbyte* %cast518, uint 552		; <sbyte*> [#uses=3]
	%cast588 = cast sbyte* %reg5861 to sbyte**		; <sbyte**> [#uses=4]
	store sbyte* %reg584, sbyte** %cast588
	%reg156 = load sbyte** %cast570		; <sbyte*> [#uses=2]
	%cond592 = seteq sbyte* %reg156, null		; <bool> [#uses=1]
	br bool %cond592, label %bb8, label %bb6

bb6:					;[#uses=1]
	%reg157 = load sbyte** %cast588		; <sbyte*> [#uses=1]
	%cond597 = seteq sbyte* %reg157, null		; <bool> [#uses=1]
	br bool %cond597, label %bb8, label %bb7

bb7:					;[#uses=1]
	%reg158 = load sbyte** %cast579		; <sbyte*> [#uses=1]
	%cond602 = setne sbyte* %reg158, null		; <bool> [#uses=1]
	br bool %cond602, label %bb9, label %bb8

bb8:					;[#uses=3]
	%cast1030 = getelementptr [30 x sbyte]* %.LC2, uint 0, uint 0		; <sbyte*> [#uses=1]
	%reg605 = call int %puts( sbyte* %cast1030 )		; <int> [#uses=0]
	%cast607 = cast sbyte* %cast518 to %network_t*		; <%network_t*> [#uses=1]
	%reg607 = call long %getfree( %network_t* %cast607 )		; <long> [#uses=0]
	ret long -1

bb9:					;[#uses=4]
	%cast611 = cast sbyte* %reg5481 to sbyte**		; <sbyte**> [#uses=2]
	%reg162 = load sbyte** %cast611		; <sbyte*> [#uses=1]
	%reg165 = sub sbyte* null, %reg162		; <sbyte*> [#uses=0]
	%reg1701 = getelementptr sbyte* %reg156, uint 120		; <sbyte*> [#uses=1]
	%reg6181 = getelementptr sbyte* %cast518, uint 544		; <sbyte*> [#uses=1]
	%cast620 = cast sbyte* %reg6181 to sbyte**		; <sbyte**> [#uses=1]
	store sbyte* %reg1701, sbyte** %cast620
	%cast623 = cast sbyte* %reg5551 to sbyte**		; <sbyte**> [#uses=1]
	%reg171 = load sbyte** %cast623		; <sbyte*> [#uses=0]
	%reg175 = load sbyte** %cast588		; <sbyte*> [#uses=1]
	%reg6281 = getelementptr sbyte* %cast518, uint 560		; <sbyte*> [#uses=2]
	%cast630 = cast sbyte* %reg6281 to sbyte**		; <sbyte**> [#uses=3]
	store sbyte* %reg175, sbyte** %cast630
	%reg177 = load sbyte** %cast611		; <sbyte*> [#uses=0]
	%reg181 = load sbyte** %cast579		; <sbyte*> [#uses=1]
	%reg6381 = getelementptr sbyte* %cast518, uint 576		; <sbyte*> [#uses=1]
	%cast640 = cast sbyte* %reg6381 to sbyte**		; <sbyte**> [#uses=1]
	store sbyte* %reg181, sbyte** %cast640
	%cast643 = cast sbyte* %reg5681 to long*		; <long*> [#uses=1]
	%reg117 = load long* %cast643		; <long> [#uses=10]
	%cast646 = cast sbyte* %reg5861 to ulong*		; <ulong*> [#uses=1]
	%reg116 = load ulong* %cast646		; <ulong> [#uses=2]
	%reg378 = load long* %cast540		; <long> [#uses=1]
	%cond650 = setlt long %reg378, 1		; <bool> [#uses=1]
	%reg7201 = getelementptr sbyte* %cast518, uint 512		; <sbyte*> [#uses=2]
	%cast722 = cast sbyte* %reg7201 to ulong*		; <ulong*> [#uses=3]
	%cast817 = cast sbyte* %reg7201 to long*		; <long*> [#uses=1]
	br bool %cond650, label %bb16, label %bb10

bb10:					;[#uses=2]
	%cann-indvar3 = phi uint [ 0, %bb9 ], [ %add1-indvar3, %bb15 ]		; <uint> [#uses=3]
	%add1-indvar3 = add uint %cann-indvar3, 1		; <uint> [#uses=1]
	%cann-indvar3-casted = cast uint %cann-indvar3 to ulong		; <ulong> [#uses=1]
	%reg392-scale = mul ulong %cann-indvar3-casted, 192		; <ulong> [#uses=1]
	%reg392 = add ulong %reg392-scale, %reg116		; <ulong> [#uses=23]
	%cann-indvar3-casted1 = cast uint %cann-indvar3 to long		; <long> [#uses=2]
	%reg393 = add long %cann-indvar3-casted1, 1		; <long> [#uses=9]
	%reg654 = call sbyte* %fgets( sbyte* %instring, int 200, %FILE* %reg520 )		; <sbyte*> [#uses=0]
	%reg658 = call int (sbyte*, sbyte*, ...)* %sscanf( sbyte* %instring, sbyte* %cast1029, long* %t, long* %h )		; <int> [#uses=1]
	%cond662 = setne int %reg658, 2		; <bool> [#uses=1]
	br bool %cond662, label %bb12, label %bb11

bb11:					;[#uses=1]
	%reg189 = load long* %t, uint 0		; <long> [#uses=1]
	%reg190 = load long* %h, uint 0		; <long> [#uses=1]
	%cond666 = setle long %reg189, %reg190		; <bool> [#uses=1]
	br bool %cond666, label %bb13, label %bb12

bb12:					;[#uses=2]
	ret long -1

bb13:					;[#uses=2]
	%reg194 = shl long %reg393, ubyte 4		; <long> [#uses=1]
	%reg195 = sub long %reg194, %reg393		; <long> [#uses=1]
	%reg196 = shl long %reg195, ubyte 3		; <long> [#uses=1]
	%reg198 = add long %reg117, %reg196		; <long> [#uses=5]
	%reg199 = sub long 0, %reg393		; <long> [#uses=1]
	%cast674 = cast long %reg198 to long*		; <long*> [#uses=1]
	store long %reg199, long* %cast674
	%reg677 = add long %reg198, 96		; <long> [#uses=1]
	%cast679 = cast long %reg677 to ulong*		; <ulong*> [#uses=1]
	store ulong 18446744073709551615, ulong* %cast679
	%reg207 = load long* %cast540		; <long> [#uses=1]
	%reg208 = add long %reg393, %reg207		; <long> [#uses=2]
	%reg210 = shl long %reg208, ubyte 4		; <long> [#uses=1]
	%reg211 = sub long %reg210, %reg208		; <long> [#uses=1]
	%reg212 = shl long %reg211, ubyte 3		; <long> [#uses=1]
	%reg214 = add long %reg117, %reg212		; <long> [#uses=1]
	%cast685 = cast long %reg214 to long*		; <long*> [#uses=1]
	store long %reg393, long* %cast685
	%reg215 = load long* %cast540		; <long> [#uses=1]
	%reg216 = add long %reg393, %reg215		; <long> [#uses=2]
	%reg218 = shl long %reg216, ubyte 4		; <long> [#uses=1]
	%reg219 = sub long %reg218, %reg216		; <long> [#uses=1]
	%reg220 = shl long %reg219, ubyte 3		; <long> [#uses=1]
	%reg222 = add long %reg117, %reg220		; <long> [#uses=1]
	%reg691 = add long %reg222, 96		; <long> [#uses=1]
	%cast693 = cast long %reg691 to ulong*		; <ulong*> [#uses=1]
	store ulong 1, ulong* %cast693
	%reg230 = load long* %t, uint 0		; <long> [#uses=1]
	%reg697 = add long %reg198, 112		; <long> [#uses=1]
	%cast699 = cast long %reg697 to long*		; <long*> [#uses=1]
	store long %reg230, long* %cast699
	%reg231 = load long* %cast540		; <long> [#uses=1]
	%reg232 = add long %reg393, %reg231		; <long> [#uses=2]
	%reg234 = shl long %reg232, ubyte 4		; <long> [#uses=1]
	%reg235 = sub long %reg234, %reg232		; <long> [#uses=1]
	%reg236 = shl long %reg235, ubyte 3		; <long> [#uses=1]
	%reg238 = add long %reg117, %reg236		; <long> [#uses=1]
	%reg239 = load long* %h, uint 0		; <long> [#uses=1]
	%reg706 = add long %reg238, 112		; <long> [#uses=1]
	%cast708 = cast long %reg706 to long*		; <long*> [#uses=1]
	store long %reg239, long* %cast708
	%reg240 = load long* %cast550		; <long> [#uses=2]
	%reg242 = shl long %reg240, ubyte 4		; <long> [#uses=1]
	%reg243 = sub long %reg242, %reg240		; <long> [#uses=1]
	%reg244 = shl long %reg243, ubyte 3		; <long> [#uses=1]
	%reg246 = add long %reg117, %reg244		; <long> [#uses=1]
	%cast714 = cast ulong %reg392 to long*		; <long*> [#uses=1]
	store long %reg246, long* %cast714
	%reg717 = add ulong %reg392, 8		; <ulong> [#uses=3]
	%cast719 = cast ulong %reg717 to long*		; <long*> [#uses=1]
	store long %reg198, long* %cast719
	%reg253 = load ulong* %cast722		; <ulong> [#uses=1]
	%reg254 = add ulong %reg253, 15		; <ulong> [#uses=2]
	%reg724 = add ulong %reg392, 32		; <ulong> [#uses=1]
	%cast726 = cast ulong %reg724 to ulong*		; <ulong*> [#uses=1]
	store ulong %reg254, ulong* %cast726
	%reg727 = add ulong %reg392, 40		; <ulong> [#uses=1]
	%cast729 = cast ulong %reg727 to ulong*		; <ulong*> [#uses=1]
	store ulong %reg254, ulong* %cast729
	%cast730 = cast ulong %reg392 to sbyte***		; <sbyte***> [#uses=1]
	%reg256 = load sbyte*** %cast730		; <sbyte**> [#uses=1]
	%reg7311 = getelementptr sbyte** %reg256, uint 9		; <sbyte**> [#uses=1]
	%reg257 = load sbyte** %reg7311		; <sbyte*> [#uses=1]
	%reg734 = add ulong %reg392, 16		; <ulong> [#uses=1]
	%cast736 = cast ulong %reg734 to sbyte**		; <sbyte**> [#uses=1]
	store sbyte* %reg257, sbyte** %cast736
	%cast737 = cast ulong %reg392 to ulong**		; <ulong**> [#uses=1]
	%reg258 = load ulong** %cast737		; <ulong*> [#uses=1]
	%reg7381 = getelementptr ulong* %reg258, uint 9		; <ulong*> [#uses=1]
	store ulong %reg392, ulong* %reg7381
	%cast743 = cast ulong %reg717 to sbyte***		; <sbyte***> [#uses=1]
	%reg259 = load sbyte*** %cast743		; <sbyte**> [#uses=1]
	%reg7441 = getelementptr sbyte** %reg259, uint 10		; <sbyte**> [#uses=1]
	%reg260 = load sbyte** %reg7441		; <sbyte*> [#uses=1]
	%reg747 = add ulong %reg392, 24		; <ulong> [#uses=1]
	%cast749 = cast ulong %reg747 to sbyte**		; <sbyte**> [#uses=1]
	store sbyte* %reg260, sbyte** %cast749
	%cast752 = cast ulong %reg717 to ulong**		; <ulong**> [#uses=1]
	%reg261 = load ulong** %cast752		; <ulong*> [#uses=1]
	%reg7531 = getelementptr ulong* %reg261, uint 10		; <ulong*> [#uses=1]
	store ulong %reg392, ulong* %reg7531
	%reg394 = add ulong %reg392, 64		; <ulong> [#uses=5]
	%reg262 = load long* %cast540		; <long> [#uses=1]
	%reg263 = add long %reg393, %reg262		; <long> [#uses=2]
	%reg265 = shl long %reg263, ubyte 4		; <long> [#uses=1]
	%reg266 = sub long %reg265, %reg263		; <long> [#uses=1]
	%reg267 = shl long %reg266, ubyte 3		; <long> [#uses=1]
	%reg269 = add long %reg117, %reg267		; <long> [#uses=1]
	%cast762 = cast ulong %reg394 to long*		; <long*> [#uses=1]
	store long %reg269, long* %cast762
	%reg270 = load long* %cast550		; <long> [#uses=2]
	%reg272 = shl long %reg270, ubyte 4		; <long> [#uses=1]
	%reg273 = sub long %reg272, %reg270		; <long> [#uses=1]
	%reg274 = shl long %reg273, ubyte 3		; <long> [#uses=1]
	%reg276 = add long %reg117, %reg274		; <long> [#uses=1]
	%reg768 = add ulong %reg392, 72		; <ulong> [#uses=3]
	%cast770 = cast ulong %reg768 to long*		; <long*> [#uses=1]
	store long %reg276, long* %cast770
	%reg771 = add ulong %reg392, 96		; <ulong> [#uses=1]
	%cast773 = cast ulong %reg771 to ulong*		; <ulong*> [#uses=1]
	store ulong 15, ulong* %cast773
	%reg774 = add ulong %reg392, 104		; <ulong> [#uses=1]
	%cast776 = cast ulong %reg774 to ulong*		; <ulong*> [#uses=1]
	store ulong 15, ulong* %cast776
	%cast777 = cast ulong %reg394 to sbyte***		; <sbyte***> [#uses=1]
	%reg279 = load sbyte*** %cast777		; <sbyte**> [#uses=1]
	%reg7781 = getelementptr sbyte** %reg279, uint 9		; <sbyte**> [#uses=1]
	%reg280 = load sbyte** %reg7781		; <sbyte*> [#uses=1]
	%reg781 = add ulong %reg392, 80		; <ulong> [#uses=1]
	%cast783 = cast ulong %reg781 to sbyte**		; <sbyte**> [#uses=1]
	store sbyte* %reg280, sbyte** %cast783
	%cast784 = cast ulong %reg394 to ulong**		; <ulong**> [#uses=1]
	%reg281 = load ulong** %cast784		; <ulong*> [#uses=1]
	%reg7851 = getelementptr ulong* %reg281, uint 9		; <ulong*> [#uses=1]
	store ulong %reg394, ulong* %reg7851
	%cast790 = cast ulong %reg768 to sbyte***		; <sbyte***> [#uses=1]
	%reg282 = load sbyte*** %cast790		; <sbyte**> [#uses=1]
	%reg7911 = getelementptr sbyte** %reg282, uint 10		; <sbyte**> [#uses=1]
	%reg283 = load sbyte** %reg7911		; <sbyte*> [#uses=1]
	%reg794 = add ulong %reg392, 88		; <ulong> [#uses=1]
	%cast796 = cast ulong %reg794 to sbyte**		; <sbyte**> [#uses=1]
	store sbyte* %reg283, sbyte** %cast796
	%cast799 = cast ulong %reg768 to ulong**		; <ulong**> [#uses=1]
	%reg284 = load ulong** %cast799		; <ulong*> [#uses=1]
	%reg8001 = getelementptr ulong* %reg284, uint 10		; <ulong*> [#uses=1]
	store ulong %reg394, ulong* %reg8001
	%reg395 = add ulong %reg392, 128		; <ulong> [#uses=5]
	%cast806 = cast ulong %reg395 to long*		; <long*> [#uses=1]
	store long %reg198, long* %cast806
	%reg291 = load long* %cast540		; <long> [#uses=1]
	%reg292 = add long %reg393, %reg291		; <long> [#uses=2]
	%reg294 = shl long %reg292, ubyte 4		; <long> [#uses=1]
	%reg295 = sub long %reg294, %reg292		; <long> [#uses=1]
	%reg296 = shl long %reg295, ubyte 3		; <long> [#uses=1]
	%reg298 = add long %reg117, %reg296		; <long> [#uses=1]
	%reg812 = add ulong %reg392, 136		; <ulong> [#uses=3]
	%cast814 = cast ulong %reg812 to long*		; <long*> [#uses=1]
	store long %reg298, long* %cast814
	%reg299 = load long* %cast817		; <long> [#uses=2]
	%cond818 = setge long %reg299, 10000000		; <bool> [#uses=1]
	br bool %cond818, label %bb15, label %bb14

bb14:					;[#uses=2]
	br label %bb15

bb15:					;[#uses=5]
	%reg397 = phi long [ 10000000, %bb14 ], [ %reg299, %bb13 ]		; <long> [#uses=2]
	%reg302 = add long %reg397, %reg397		; <long> [#uses=2]
	%reg821 = add ulong %reg392, 160		; <ulong> [#uses=1]
	%cast823 = cast ulong %reg821 to long*		; <long*> [#uses=1]
	store long %reg302, long* %cast823
	%reg824 = add ulong %reg392, 168		; <ulong> [#uses=1]
	%cast826 = cast ulong %reg824 to long*		; <long*> [#uses=1]
	store long %reg302, long* %cast826
	%cast827 = cast ulong %reg395 to sbyte***		; <sbyte***> [#uses=1]
	%reg304 = load sbyte*** %cast827		; <sbyte**> [#uses=1]
	%reg8281 = getelementptr sbyte** %reg304, uint 9		; <sbyte**> [#uses=1]
	%reg305 = load sbyte** %reg8281		; <sbyte*> [#uses=1]
	%reg831 = add ulong %reg392, 144		; <ulong> [#uses=1]
	%cast833 = cast ulong %reg831 to sbyte**		; <sbyte**> [#uses=1]
	store sbyte* %reg305, sbyte** %cast833
	%cast834 = cast ulong %reg395 to ulong**		; <ulong**> [#uses=1]
	%reg306 = load ulong** %cast834		; <ulong*> [#uses=1]
	%reg8351 = getelementptr ulong* %reg306, uint 9		; <ulong*> [#uses=1]
	store ulong %reg395, ulong* %reg8351
	%cast840 = cast ulong %reg812 to sbyte***		; <sbyte***> [#uses=1]
	%reg307 = load sbyte*** %cast840		; <sbyte**> [#uses=1]
	%reg8411 = getelementptr sbyte** %reg307, uint 10		; <sbyte**> [#uses=1]
	%reg308 = load sbyte** %reg8411		; <sbyte*> [#uses=1]
	%reg844 = add ulong %reg392, 152		; <ulong> [#uses=1]
	%cast846 = cast ulong %reg844 to sbyte**		; <sbyte**> [#uses=1]
	store sbyte* %reg308, sbyte** %cast846
	%cast849 = cast ulong %reg812 to ulong**		; <ulong**> [#uses=1]
	%reg309 = load ulong** %cast849		; <ulong*> [#uses=1]
	%reg8501 = getelementptr ulong* %reg309, uint 10		; <ulong*> [#uses=1]
	store ulong %reg395, ulong* %reg8501
	%reg398 = add ulong %reg392, 192		; <ulong> [#uses=1]
	%reg399 = add long %cann-indvar3-casted1, 2		; <long> [#uses=2]
	%reg183 = load long* %cast540		; <long> [#uses=1]
	%cond858 = setle long %reg399, %reg183		; <bool> [#uses=1]
	br bool %cond858, label %bb10, label %bb16

bb16:					;[#uses=2]
	%reg400 = phi ulong [ %reg398, %bb15 ], [ %reg116, %bb9 ]		; <ulong> [#uses=2]
	%reg401 = phi long [ %reg399, %bb15 ], [ 1, %bb9 ]		; <long> [#uses=1]
	%reg310 = load long* %cast540		; <long> [#uses=1]
	%reg311 = add long %reg310, 1		; <long> [#uses=1]
	%cond865 = seteq long %reg401, %reg311		; <bool> [#uses=1]
	br bool %cond865, label %bb18, label %bb17

bb17:					;[#uses=1]
	ret long -1

bb18:					;[#uses=3]
	%reg379 = load long* %cast544		; <long> [#uses=1]
	%cond870 = setle long %reg379, 0		; <bool> [#uses=1]
	%cast1032 = getelementptr [12 x sbyte]* %.LC3, uint 0, uint 0		; <sbyte*> [#uses=1]
	br bool %cond870, label %bb22, label %bb19

bb19:					;[#uses=2]
	%cann-indvar2 = phi uint [ 0, %bb18 ], [ %add1-indvar2, %bb21 ]		; <uint> [#uses=3]
	%add1-indvar2 = add uint %cann-indvar2, 1		; <uint> [#uses=1]
	%cann-indvar2-casted = cast uint %cann-indvar2 to ulong		; <ulong> [#uses=1]
	%reg403-scale = mul ulong %cann-indvar2-casted, 64		; <ulong> [#uses=1]
	%reg403 = add ulong %reg403-scale, %reg400		; <ulong> [#uses=11]
	%cann-indvar2-casted1 = cast uint %cann-indvar2 to long		; <long> [#uses=1]
	%reg874 = call sbyte* %fgets( sbyte* %instring, int 200, %FILE* %reg520 )		; <sbyte*> [#uses=0]
	%reg878 = call int (sbyte*, sbyte*, ...)* %sscanf( sbyte* %instring, sbyte* %cast1032, long* %t, long* %h, long* %c )		; <int> [#uses=1]
	%cond883 = seteq int %reg878, 3		; <bool> [#uses=1]
	br bool %cond883, label %bb21, label %bb20

bb20:					;[#uses=1]
	ret long -1

bb21:					;[#uses=3]
	%reg322 = load long* %t		; <long> [#uses=1]
	%reg323 = load long* %cast540		; <long> [#uses=1]
	%reg324 = add long %reg322, %reg323		; <long> [#uses=2]
	%reg326 = shl long %reg324, ubyte 4		; <long> [#uses=1]
	%reg327 = sub long %reg326, %reg324		; <long> [#uses=1]
	%reg328 = shl long %reg327, ubyte 3		; <long> [#uses=1]
	%reg330 = add long %reg117, %reg328		; <long> [#uses=1]
	%cast892 = cast ulong %reg403 to long*		; <long*> [#uses=1]
	store long %reg330, long* %cast892
	%reg331 = load long* %h		; <long> [#uses=2]
	%reg333 = shl long %reg331, ubyte 4		; <long> [#uses=1]
	%reg334 = sub long %reg333, %reg331		; <long> [#uses=1]
	%reg335 = shl long %reg334, ubyte 3		; <long> [#uses=1]
	%reg337 = add long %reg117, %reg335		; <long> [#uses=1]
	%reg896 = add ulong %reg403, 8		; <ulong> [#uses=3]
	%cast898 = cast ulong %reg896 to long*		; <long*> [#uses=1]
	store long %reg337, long* %cast898
	%reg338 = load long* %c, uint 0		; <long> [#uses=1]
	%reg900 = add ulong %reg403, 40		; <ulong> [#uses=1]
	%cast902 = cast ulong %reg900 to long*		; <long*> [#uses=1]
	store long %reg338, long* %cast902
	%reg339 = load long* %c, uint 0		; <long> [#uses=1]
	%reg904 = add ulong %reg403, 32		; <ulong> [#uses=1]
	%cast906 = cast ulong %reg904 to long*		; <long*> [#uses=1]
	store long %reg339, long* %cast906
	%cast907 = cast ulong %reg403 to sbyte***		; <sbyte***> [#uses=1]
	%reg340 = load sbyte*** %cast907		; <sbyte**> [#uses=1]
	%reg9081 = getelementptr sbyte** %reg340, uint 9		; <sbyte**> [#uses=1]
	%reg341 = load sbyte** %reg9081		; <sbyte*> [#uses=1]
	%reg911 = add ulong %reg403, 16		; <ulong> [#uses=1]
	%cast913 = cast ulong %reg911 to sbyte**		; <sbyte**> [#uses=1]
	store sbyte* %reg341, sbyte** %cast913
	%cast914 = cast ulong %reg403 to ulong**		; <ulong**> [#uses=1]
	%reg342 = load ulong** %cast914		; <ulong*> [#uses=1]
	%reg9151 = getelementptr ulong* %reg342, uint 9		; <ulong*> [#uses=1]
	store ulong %reg403, ulong* %reg9151
	%cast920 = cast ulong %reg896 to sbyte***		; <sbyte***> [#uses=1]
	%reg343 = load sbyte*** %cast920		; <sbyte**> [#uses=1]
	%reg9211 = getelementptr sbyte** %reg343, uint 10		; <sbyte**> [#uses=1]
	%reg344 = load sbyte** %reg9211		; <sbyte*> [#uses=1]
	%reg924 = add ulong %reg403, 24		; <ulong> [#uses=1]
	%cast926 = cast ulong %reg924 to sbyte**		; <sbyte**> [#uses=1]
	store sbyte* %reg344, sbyte** %cast926
	%cast929 = cast ulong %reg896 to ulong**		; <ulong**> [#uses=1]
	%reg345 = load ulong** %cast929		; <ulong*> [#uses=1]
	%reg9301 = getelementptr ulong* %reg345, uint 10		; <ulong*> [#uses=1]
	store ulong %reg403, ulong* %reg9301
	%reg405 = add long %cann-indvar2-casted1, 1		; <long> [#uses=1]
	%reg406 = add ulong %reg403, 64		; <ulong> [#uses=1]
	%reg314 = load long* %cast544		; <long> [#uses=1]
	%cond938 = setlt long %reg405, %reg314		; <bool> [#uses=1]
	br bool %cond938, label %bb19, label %bb22

bb22:					;[#uses=2]
	%reg407 = phi ulong [ %reg406, %bb21 ], [ %reg400, %bb18 ]		; <ulong> [#uses=2]
	%cast943 = cast sbyte* %reg6281 to ulong*		; <ulong*> [#uses=2]
	%reg346 = load ulong* %cast943		; <ulong> [#uses=1]
	%cond944 = seteq ulong %reg346, %reg407		; <bool> [#uses=1]
	br bool %cond944, label %bb26, label %bb23

bb23:					;[#uses=2]
	store ulong %reg407, ulong* %cast943
	%reg408 = load sbyte** %cast588		; <sbyte*> [#uses=2]
	%cast953 = cast sbyte* %reg5551 to ulong*		; <ulong*> [#uses=4]
	store ulong 0, ulong* %cast953
	%reg380 = load sbyte** %cast630		; <sbyte*> [#uses=1]
	%cond957 = setge sbyte* %reg408, %reg380		; <bool> [#uses=1]
	br bool %cond957, label %bb25, label %bb24

bb24:					;[#uses=3]
	%cann-indvar1 = phi uint [ 0, %bb23 ], [ %add1-indvar1, %bb24 ]		; <uint> [#uses=2]
	%add1-indvar1 = add uint %cann-indvar1, 1		; <uint> [#uses=1]
	%cann-indvar1-scale = mul uint %cann-indvar1, 64		; <uint> [#uses=1]
	%reg4091 = getelementptr sbyte* %reg408, uint %cann-indvar1-scale		; <sbyte*> [#uses=1]
	%reg349 = load ulong* %cast953		; <ulong> [#uses=1]
	%reg350 = add ulong %reg349, 1		; <ulong> [#uses=1]
	store ulong %reg350, ulong* %cast953
	%reg4101 = getelementptr sbyte* %reg4091, uint 64		; <sbyte*> [#uses=1]
	%reg348 = load sbyte** %cast630		; <sbyte*> [#uses=1]
	%cond969 = setlt sbyte* %reg4101, %reg348		; <bool> [#uses=1]
	br bool %cond969, label %bb24, label %bb25

bb25:					;[#uses=2]
	%reg351 = load ulong* %cast953		; <ulong> [#uses=1]
	%cast975 = cast sbyte* %reg5421 to ulong*		; <ulong*> [#uses=1]
	store ulong %reg351, ulong* %cast975
	br label %bb26

bb26:					;[#uses=3]
	%reg977 = call int %fclose( %FILE* %reg520 )		; <int> [#uses=0]
	%reg9781 = getelementptr sbyte* %cast518, uint 200		; <sbyte*> [#uses=1]
	store sbyte 0, sbyte* %reg9781
	%reg381 = load long* %cast540		; <long> [#uses=1]
	%cond984 = setlt long %reg381, 1		; <bool> [#uses=1]
	%cast991 = cast sbyte* %reg5861 to long*		; <long*> [#uses=2]
	br bool %cond984, label %bb32, label %bb27

bb27:					;[#uses=3]
	%cann-indvar = phi uint [ 0, %bb26 ], [ %add1-indvar, %bb31 ]		; <uint> [#uses=2]
	%add1-indvar = add uint %cann-indvar, 1		; <uint> [#uses=1]
	%cann-indvar-casted = cast uint %cann-indvar to long		; <long> [#uses=2]
	%reg412 = add long %cann-indvar-casted, 1		; <long> [#uses=2]
	%reg355 = shl long %reg412, ubyte 1		; <long> [#uses=1]
	%reg356 = add long %reg355, %reg412		; <long> [#uses=1]
	%reg357 = shl long %reg356, ubyte 6		; <long> [#uses=2]
	%reg359 = load long* %cast991		; <long> [#uses=1]
	%reg360 = add long %reg359, %reg357		; <long> [#uses=1]
	%reg362 = load ulong* %cast722		; <ulong> [#uses=2]
	%cast997 = cast ulong %reg362 to long		; <long> [#uses=1]
	%cond996 = setge long %cast997, 10000000		; <bool> [#uses=1]
	br bool %cond996, label %bb29, label %bb28

bb28:					;[#uses=2]
	br label %bb29

bb29:					;[#uses=3]
	%reg414 = phi ulong [ 10000000, %bb28 ], [ %reg362, %bb27 ]		; <ulong> [#uses=1]
	%reg364 = mul ulong %reg414, 18446744073709551614		; <ulong> [#uses=1]
	%reg1000 = add long %reg360, -32		; <long> [#uses=1]
	%cast1002 = cast long %reg1000 to ulong*		; <ulong*> [#uses=1]
	store ulong %reg364, ulong* %cast1002
	%reg370 = load long* %cast991		; <long> [#uses=1]
	%reg371 = add long %reg370, %reg357		; <long> [#uses=1]
	%reg373 = load ulong* %cast722		; <ulong> [#uses=2]
	%cast1013 = cast ulong %reg373 to long		; <long> [#uses=1]
	%cond1012 = setge long %cast1013, 10000000		; <bool> [#uses=1]
	br bool %cond1012, label %bb31, label %bb30

bb30:					;[#uses=2]
	br label %bb31

bb31:					;[#uses=3]
	%reg416 = phi ulong [ 10000000, %bb30 ], [ %reg373, %bb29 ]		; <ulong> [#uses=1]
	%reg375 = mul ulong %reg416, 18446744073709551614		; <ulong> [#uses=1]
	%reg1016 = add long %reg371, -24		; <long> [#uses=1]
	%cast1018 = cast long %reg1016 to ulong*		; <ulong*> [#uses=1]
	store ulong %reg375, ulong* %cast1018
	%reg417 = add long %cann-indvar-casted, 2		; <long> [#uses=1]
	%reg353 = load long* %cast540		; <long> [#uses=1]
	%cond1023 = setle long %reg417, %reg353		; <bool> [#uses=1]
	br bool %cond1023, label %bb27, label %bb32

bb32:					;[#uses=2]
	ret long 0

bb33:					;[#uses=0]
	ret long 42
}

declare int %fclose(%FILE*)

declare %FILE* %fopen(sbyte*, sbyte*)

declare int %sscanf(sbyte*, sbyte*, ...)

declare sbyte* %fgets(sbyte*, int, %FILE*)

declare int %puts(sbyte*)

declare sbyte* %calloc(ulong, ulong)

declare long %getfree(%network_t*)
