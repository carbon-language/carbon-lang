; This bytecode test pounds on instruction alignment and showed
; up a bug in llvm-dis performance

; RUN: llvm-as < %s | llvm-dis > %t1
; RUN: llvm-dis < %s.bc-13 > %t2
; RUN: diff %t1 %t2

implementation   ; Functions:

declare int %getchar()

declare int %putchar(int)

ubyte %inputcell() {
entry:
	call int %getchar( )		; <int>:0 [#uses=2]
	seteq int %0, -1		; <bool>:0 [#uses=1]
	br bool %0, label %eof, label %ok

ok:		; preds = %entry
	cast int %0 to ubyte		; <ubyte>:0 [#uses=1]
	ret ubyte %0

eof:		; preds = %entry
	ret ubyte 0
}

void %outputcell(ubyte) {
entry:
	cast ubyte %0 to int		; <int>:0 [#uses=1]
	call int %putchar( int %0 )		; <int>:1 [#uses=0]
	ret void
}

int %main() {
entry:
	%bfarray = malloc [262144 x ubyte]		; <[262144 x ubyte]*> [#uses=3366]
	%bfarray.sub = getelementptr [262144 x ubyte]* %bfarray, int 0, int 0		; <ubyte*> [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, int 3		; <ubyte*>:0 [#uses=2]
	load ubyte* %0		; <ubyte>:0 [#uses=1]
	add ubyte %0, 1		; <ubyte>:1 [#uses=1]
	store ubyte %1, ubyte* %0
	getelementptr [262144 x ubyte]* %bfarray, int 0, int 6		; <ubyte*>:1 [#uses=2]
	load ubyte* %1		; <ubyte>:2 [#uses=2]
	add ubyte %2, 2		; <ubyte>:3 [#uses=1]
	store ubyte %3, ubyte* %1
	seteq ubyte %2, 254		; <bool>:0 [#uses=1]
	br bool %0, label %1, label %0

; <label>:0		; preds = %entry, %3
	phi uint [ 6, %entry ], [ %4, %3 ]		; <uint>:0 [#uses=4]
	add uint %0, 1		; <uint>:1 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1		; <ubyte*>:2 [#uses=2]
	load ubyte* %2		; <ubyte>:4 [#uses=1]
	add ubyte %4, 1		; <ubyte>:5 [#uses=1]
	store ubyte %5, ubyte* %2
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %0		; <ubyte*>:3 [#uses=1]
	load ubyte* %3		; <ubyte>:6 [#uses=1]
	seteq ubyte %6, 0		; <bool>:1 [#uses=1]
	br bool %1, label %3, label %2

; <label>:1		; preds = %entry, %3
	free ubyte* %bfarray.sub
	ret int 0

; <label>:2		; preds = %0, %567
	phi uint [ %0, %0 ], [ %377, %567 ]		; <uint>:2 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %2		; <ubyte*>:4 [#uses=2]
	load ubyte* %4		; <ubyte>:7 [#uses=2]
	add ubyte %7, 255		; <ubyte>:8 [#uses=1]
	store ubyte %8, ubyte* %4
	seteq ubyte %7, 1		; <bool>:2 [#uses=1]
	br bool %2, label %5, label %4

; <label>:3		; preds = %0, %567
	phi uint [ %0, %0 ], [ %377, %567 ]		; <uint>:3 [#uses=1]
	add uint %3, 4294967295		; <uint>:4 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %4		; <ubyte*>:5 [#uses=1]
	load ubyte* %5		; <ubyte>:9 [#uses=1]
	seteq ubyte %9, 0		; <bool>:3 [#uses=1]
	br bool %3, label %1, label %0

; <label>:4		; preds = %2, %11
	phi uint [ %2, %2 ], [ %15, %11 ]		; <uint>:5 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %5		; <ubyte*>:6 [#uses=2]
	load ubyte* %6		; <ubyte>:10 [#uses=2]
	add ubyte %10, 255		; <ubyte>:11 [#uses=1]
	store ubyte %11, ubyte* %6
	seteq ubyte %10, 1		; <bool>:4 [#uses=1]
	br bool %4, label %7, label %6

; <label>:5		; preds = %2, %11
	phi uint [ %2, %2 ], [ %15, %11 ]		; <uint>:6 [#uses=1]
	add uint %6, 1		; <uint>:7 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %7		; <ubyte*>:7 [#uses=1]
	load ubyte* %7		; <ubyte>:12 [#uses=1]
	seteq ubyte %12, 0		; <bool>:5 [#uses=1]
	br bool %5, label %567, label %566

; <label>:6		; preds = %4, %9
	phi uint [ %5, %4 ], [ %11, %9 ]		; <uint>:8 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %8		; <ubyte*>:8 [#uses=1]
	load ubyte* %8		; <ubyte>:13 [#uses=1]
	seteq ubyte %13, 0		; <bool>:6 [#uses=1]
	br bool %6, label %9, label %8

; <label>:7		; preds = %4, %9
	phi uint [ %5, %4 ], [ %11, %9 ]		; <uint>:9 [#uses=1]
	add uint %9, 1		; <uint>:10 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %10		; <ubyte*>:9 [#uses=1]
	load ubyte* %9		; <ubyte>:14 [#uses=1]
	seteq ubyte %14, 0		; <bool>:7 [#uses=1]
	br bool %7, label %11, label %10

; <label>:8		; preds = %6, %8
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %8		; <ubyte*>:10 [#uses=2]
	load ubyte* %10		; <ubyte>:15 [#uses=2]
	add ubyte %15, 255		; <ubyte>:16 [#uses=1]
	store ubyte %16, ubyte* %10
	seteq ubyte %15, 1		; <bool>:8 [#uses=1]
	br bool %8, label %9, label %8

; <label>:9		; preds = %6, %8
	add uint %8, 1		; <uint>:11 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %11		; <ubyte*>:11 [#uses=2]
	load ubyte* %11		; <ubyte>:17 [#uses=2]
	add ubyte %17, 255		; <ubyte>:18 [#uses=1]
	store ubyte %18, ubyte* %11
	seteq ubyte %17, 1		; <bool>:9 [#uses=1]
	br bool %9, label %7, label %6

; <label>:10		; preds = %7, %13
	phi uint [ %10, %7 ], [ %19, %13 ]		; <uint>:12 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %12		; <ubyte*>:12 [#uses=2]
	load ubyte* %12		; <ubyte>:19 [#uses=1]
	add ubyte %19, 255		; <ubyte>:20 [#uses=1]
	store ubyte %20, ubyte* %12
	add uint %12, 4294967292		; <uint>:13 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %13		; <ubyte*>:13 [#uses=1]
	load ubyte* %13		; <ubyte>:21 [#uses=1]
	seteq ubyte %21, 0		; <bool>:10 [#uses=1]
	br bool %10, label %13, label %12

; <label>:11		; preds = %7, %13
	phi uint [ %10, %7 ], [ %19, %13 ]		; <uint>:14 [#uses=1]
	add uint %14, 4294967295		; <uint>:15 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %15		; <ubyte*>:14 [#uses=1]
	load ubyte* %14		; <ubyte>:22 [#uses=1]
	seteq ubyte %22, 0		; <bool>:11 [#uses=1]
	br bool %11, label %5, label %4

; <label>:12		; preds = %10, %15
	phi uint [ %13, %10 ], [ %22, %15 ]		; <uint>:16 [#uses=4]
	add uint %16, 1		; <uint>:17 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %17		; <ubyte*>:15 [#uses=2]
	load ubyte* %15		; <ubyte>:23 [#uses=1]
	add ubyte %23, 1		; <ubyte>:24 [#uses=1]
	store ubyte %24, ubyte* %15
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %16		; <ubyte*>:16 [#uses=1]
	load ubyte* %16		; <ubyte>:25 [#uses=1]
	seteq ubyte %25, 0		; <bool>:12 [#uses=1]
	br bool %12, label %15, label %14

; <label>:13		; preds = %10, %15
	phi uint [ %13, %10 ], [ %22, %15 ]		; <uint>:18 [#uses=1]
	add uint %18, 4294967295		; <uint>:19 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %19		; <ubyte*>:17 [#uses=1]
	load ubyte* %17		; <ubyte>:26 [#uses=1]
	seteq ubyte %26, 0		; <bool>:13 [#uses=1]
	br bool %13, label %11, label %10

; <label>:14		; preds = %12, %557
	phi uint [ %16, %12 ], [ %366, %557 ]		; <uint>:20 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %20		; <ubyte*>:18 [#uses=2]
	load ubyte* %18		; <ubyte>:27 [#uses=2]
	add ubyte %27, 255		; <ubyte>:28 [#uses=1]
	store ubyte %28, ubyte* %18
	seteq ubyte %27, 1		; <bool>:14 [#uses=1]
	br bool %14, label %17, label %16

; <label>:15		; preds = %12, %557
	phi uint [ %16, %12 ], [ %366, %557 ]		; <uint>:21 [#uses=1]
	add uint %21, 4294967295		; <uint>:22 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %22		; <ubyte*>:19 [#uses=1]
	load ubyte* %19		; <ubyte>:29 [#uses=1]
	seteq ubyte %29, 0		; <bool>:15 [#uses=1]
	br bool %15, label %13, label %12

; <label>:16		; preds = %14, %459
	phi uint [ %20, %14 ], [ %293, %459 ]		; <uint>:23 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %23		; <ubyte*>:20 [#uses=2]
	load ubyte* %20		; <ubyte>:30 [#uses=2]
	add ubyte %30, 255		; <ubyte>:31 [#uses=1]
	store ubyte %31, ubyte* %20
	seteq ubyte %30, 1		; <bool>:16 [#uses=1]
	br bool %16, label %19, label %18

; <label>:17		; preds = %14, %459
	phi uint [ %20, %14 ], [ %293, %459 ]		; <uint>:24 [#uses=1]
	add uint %24, 1		; <uint>:25 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %25		; <ubyte*>:21 [#uses=1]
	load ubyte* %21		; <ubyte>:32 [#uses=1]
	seteq ubyte %32, 0		; <bool>:17 [#uses=1]
	br bool %17, label %557, label %556

; <label>:18		; preds = %16, %403
	phi uint [ %23, %16 ], [ %268, %403 ]		; <uint>:26 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %26		; <ubyte*>:22 [#uses=2]
	load ubyte* %22		; <ubyte>:33 [#uses=2]
	add ubyte %33, 255		; <ubyte>:34 [#uses=1]
	store ubyte %34, ubyte* %22
	seteq ubyte %33, 1		; <bool>:18 [#uses=1]
	br bool %18, label %21, label %20

; <label>:19		; preds = %16, %403
	phi uint [ %23, %16 ], [ %268, %403 ]		; <uint>:27 [#uses=1]
	add uint %27, 1		; <uint>:28 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %28		; <ubyte*>:23 [#uses=1]
	load ubyte* %23		; <ubyte>:35 [#uses=1]
	seteq ubyte %35, 0		; <bool>:19 [#uses=1]
	br bool %19, label %459, label %458

; <label>:20		; preds = %18, %361
	phi uint [ %26, %18 ], [ %240, %361 ]		; <uint>:29 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %29		; <ubyte*>:24 [#uses=2]
	load ubyte* %24		; <ubyte>:36 [#uses=2]
	add ubyte %36, 255		; <ubyte>:37 [#uses=1]
	store ubyte %37, ubyte* %24
	seteq ubyte %36, 1		; <bool>:20 [#uses=1]
	br bool %20, label %23, label %22

; <label>:21		; preds = %18, %361
	phi uint [ %26, %18 ], [ %240, %361 ]		; <uint>:30 [#uses=1]
	add uint %30, 1		; <uint>:31 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %31		; <ubyte*>:25 [#uses=1]
	load ubyte* %25		; <ubyte>:38 [#uses=1]
	seteq ubyte %38, 0		; <bool>:21 [#uses=1]
	br bool %21, label %403, label %402

; <label>:22		; preds = %20, %291
	phi uint [ %29, %20 ], [ %201, %291 ]		; <uint>:32 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %32		; <ubyte*>:26 [#uses=2]
	load ubyte* %26		; <ubyte>:39 [#uses=2]
	add ubyte %39, 255		; <ubyte>:40 [#uses=1]
	store ubyte %40, ubyte* %26
	seteq ubyte %39, 1		; <bool>:22 [#uses=1]
	br bool %22, label %25, label %24

; <label>:23		; preds = %20, %291
	phi uint [ %29, %20 ], [ %201, %291 ]		; <uint>:33 [#uses=1]
	add uint %33, 1		; <uint>:34 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %34		; <ubyte*>:27 [#uses=1]
	load ubyte* %27		; <ubyte>:41 [#uses=1]
	seteq ubyte %41, 0		; <bool>:23 [#uses=1]
	br bool %23, label %361, label %360

; <label>:24		; preds = %22, %253
	phi uint [ %32, %22 ], [ %179, %253 ]		; <uint>:35 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %35		; <ubyte*>:28 [#uses=2]
	load ubyte* %28		; <ubyte>:42 [#uses=2]
	add ubyte %42, 255		; <ubyte>:43 [#uses=1]
	store ubyte %43, ubyte* %28
	seteq ubyte %42, 1		; <bool>:24 [#uses=1]
	br bool %24, label %27, label %26

; <label>:25		; preds = %22, %253
	phi uint [ %32, %22 ], [ %179, %253 ]		; <uint>:36 [#uses=1]
	add uint %36, 1		; <uint>:37 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %37		; <ubyte*>:29 [#uses=1]
	load ubyte* %29		; <ubyte>:44 [#uses=1]
	seteq ubyte %44, 0		; <bool>:25 [#uses=1]
	br bool %25, label %291, label %290

; <label>:26		; preds = %24, %197
	phi uint [ %35, %24 ], [ %154, %197 ]		; <uint>:38 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %38		; <ubyte*>:30 [#uses=2]
	load ubyte* %30		; <ubyte>:45 [#uses=2]
	add ubyte %45, 255		; <ubyte>:46 [#uses=1]
	store ubyte %46, ubyte* %30
	seteq ubyte %45, 1		; <bool>:26 [#uses=1]
	br bool %26, label %29, label %28

; <label>:27		; preds = %24, %197
	phi uint [ %35, %24 ], [ %154, %197 ]		; <uint>:39 [#uses=1]
	add uint %39, 1		; <uint>:40 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %40		; <ubyte*>:31 [#uses=1]
	load ubyte* %31		; <ubyte>:47 [#uses=1]
	seteq ubyte %47, 0		; <bool>:27 [#uses=1]
	br bool %27, label %253, label %252

; <label>:28		; preds = %26, %35
	phi uint [ %38, %26 ], [ %51, %35 ]		; <uint>:41 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %41		; <ubyte*>:32 [#uses=2]
	load ubyte* %32		; <ubyte>:48 [#uses=2]
	add ubyte %48, 255		; <ubyte>:49 [#uses=1]
	store ubyte %49, ubyte* %32
	seteq ubyte %48, 1		; <bool>:28 [#uses=1]
	br bool %28, label %31, label %30

; <label>:29		; preds = %26, %35
	phi uint [ %38, %26 ], [ %51, %35 ]		; <uint>:42 [#uses=1]
	add uint %42, 1		; <uint>:43 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %43		; <ubyte*>:33 [#uses=1]
	load ubyte* %33		; <ubyte>:50 [#uses=1]
	seteq ubyte %50, 0		; <bool>:29 [#uses=1]
	br bool %29, label %197, label %196

; <label>:30		; preds = %28, %33
	phi uint [ %41, %28 ], [ %47, %33 ]		; <uint>:44 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %44		; <ubyte*>:34 [#uses=1]
	load ubyte* %34		; <ubyte>:51 [#uses=1]
	seteq ubyte %51, 0		; <bool>:30 [#uses=1]
	br bool %30, label %33, label %32

; <label>:31		; preds = %28, %33
	phi uint [ %41, %28 ], [ %47, %33 ]		; <uint>:45 [#uses=1]
	add uint %45, 1		; <uint>:46 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %46		; <ubyte*>:35 [#uses=1]
	load ubyte* %35		; <ubyte>:52 [#uses=1]
	seteq ubyte %52, 0		; <bool>:31 [#uses=1]
	br bool %31, label %35, label %34

; <label>:32		; preds = %30, %32
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %44		; <ubyte*>:36 [#uses=2]
	load ubyte* %36		; <ubyte>:53 [#uses=2]
	add ubyte %53, 255		; <ubyte>:54 [#uses=1]
	store ubyte %54, ubyte* %36
	seteq ubyte %53, 1		; <bool>:32 [#uses=1]
	br bool %32, label %33, label %32

; <label>:33		; preds = %30, %32
	add uint %44, 1		; <uint>:47 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %47		; <ubyte*>:37 [#uses=2]
	load ubyte* %37		; <ubyte>:55 [#uses=2]
	add ubyte %55, 255		; <ubyte>:56 [#uses=1]
	store ubyte %56, ubyte* %37
	seteq ubyte %55, 1		; <bool>:33 [#uses=1]
	br bool %33, label %31, label %30

; <label>:34		; preds = %31, %195
	phi uint [ %46, %31 ], [ %150, %195 ]		; <uint>:48 [#uses=66]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %48		; <ubyte*>:38 [#uses=2]
	load ubyte* %38		; <ubyte>:57 [#uses=1]
	add ubyte %57, 255		; <ubyte>:58 [#uses=1]
	store ubyte %58, ubyte* %38
	add uint %48, 112		; <uint>:49 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %49		; <ubyte*>:39 [#uses=1]
	load ubyte* %39		; <ubyte>:59 [#uses=1]
	seteq ubyte %59, 0		; <bool>:34 [#uses=1]
	br bool %34, label %37, label %36

; <label>:35		; preds = %31, %195
	phi uint [ %46, %31 ], [ %150, %195 ]		; <uint>:50 [#uses=1]
	add uint %50, 4294967295		; <uint>:51 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %51		; <ubyte*>:40 [#uses=1]
	load ubyte* %40		; <ubyte>:60 [#uses=1]
	seteq ubyte %60, 0		; <bool>:35 [#uses=1]
	br bool %35, label %29, label %28

; <label>:36		; preds = %34, %36
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %49		; <ubyte*>:41 [#uses=2]
	load ubyte* %41		; <ubyte>:61 [#uses=2]
	add ubyte %61, 255		; <ubyte>:62 [#uses=1]
	store ubyte %62, ubyte* %41
	seteq ubyte %61, 1		; <bool>:36 [#uses=1]
	br bool %36, label %37, label %36

; <label>:37		; preds = %34, %36
	add uint %48, 10		; <uint>:52 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %52		; <ubyte*>:42 [#uses=1]
	load ubyte* %42		; <ubyte>:63 [#uses=1]
	seteq ubyte %63, 0		; <bool>:37 [#uses=1]
	br bool %37, label %39, label %38

; <label>:38		; preds = %37, %38
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %52		; <ubyte*>:43 [#uses=2]
	load ubyte* %43		; <ubyte>:64 [#uses=1]
	add ubyte %64, 255		; <ubyte>:65 [#uses=1]
	store ubyte %65, ubyte* %43
	add uint %48, 11		; <uint>:53 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %53		; <ubyte*>:44 [#uses=2]
	load ubyte* %44		; <ubyte>:66 [#uses=1]
	add ubyte %66, 1		; <ubyte>:67 [#uses=1]
	store ubyte %67, ubyte* %44
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %49		; <ubyte*>:45 [#uses=2]
	load ubyte* %45		; <ubyte>:68 [#uses=1]
	add ubyte %68, 1		; <ubyte>:69 [#uses=1]
	store ubyte %69, ubyte* %45
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %52		; <ubyte*>:46 [#uses=1]
	load ubyte* %46		; <ubyte>:70 [#uses=1]
	seteq ubyte %70, 0		; <bool>:38 [#uses=1]
	br bool %38, label %39, label %38

; <label>:39		; preds = %37, %38
	add uint %48, 11		; <uint>:54 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %54		; <ubyte*>:47 [#uses=1]
	load ubyte* %47		; <ubyte>:71 [#uses=1]
	seteq ubyte %71, 0		; <bool>:39 [#uses=1]
	br bool %39, label %41, label %40

; <label>:40		; preds = %39, %40
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %52		; <ubyte*>:48 [#uses=2]
	load ubyte* %48		; <ubyte>:72 [#uses=1]
	add ubyte %72, 1		; <ubyte>:73 [#uses=1]
	store ubyte %73, ubyte* %48
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %54		; <ubyte*>:49 [#uses=2]
	load ubyte* %49		; <ubyte>:74 [#uses=2]
	add ubyte %74, 255		; <ubyte>:75 [#uses=1]
	store ubyte %75, ubyte* %49
	seteq ubyte %74, 1		; <bool>:40 [#uses=1]
	br bool %40, label %41, label %40

; <label>:41		; preds = %39, %40
	add uint %48, 118		; <uint>:55 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %55		; <ubyte*>:50 [#uses=1]
	load ubyte* %50		; <ubyte>:76 [#uses=1]
	seteq ubyte %76, 0		; <bool>:41 [#uses=1]
	br bool %41, label %43, label %42

; <label>:42		; preds = %41, %42
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %55		; <ubyte*>:51 [#uses=2]
	load ubyte* %51		; <ubyte>:77 [#uses=2]
	add ubyte %77, 255		; <ubyte>:78 [#uses=1]
	store ubyte %78, ubyte* %51
	seteq ubyte %77, 1		; <bool>:42 [#uses=1]
	br bool %42, label %43, label %42

; <label>:43		; preds = %41, %42
	add uint %48, 16		; <uint>:56 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %56		; <ubyte*>:52 [#uses=1]
	load ubyte* %52		; <ubyte>:79 [#uses=1]
	seteq ubyte %79, 0		; <bool>:43 [#uses=1]
	br bool %43, label %45, label %44

; <label>:44		; preds = %43, %44
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %56		; <ubyte*>:53 [#uses=2]
	load ubyte* %53		; <ubyte>:80 [#uses=1]
	add ubyte %80, 255		; <ubyte>:81 [#uses=1]
	store ubyte %81, ubyte* %53
	add uint %48, 17		; <uint>:57 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %57		; <ubyte*>:54 [#uses=2]
	load ubyte* %54		; <ubyte>:82 [#uses=1]
	add ubyte %82, 1		; <ubyte>:83 [#uses=1]
	store ubyte %83, ubyte* %54
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %55		; <ubyte*>:55 [#uses=2]
	load ubyte* %55		; <ubyte>:84 [#uses=1]
	add ubyte %84, 1		; <ubyte>:85 [#uses=1]
	store ubyte %85, ubyte* %55
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %56		; <ubyte*>:56 [#uses=1]
	load ubyte* %56		; <ubyte>:86 [#uses=1]
	seteq ubyte %86, 0		; <bool>:44 [#uses=1]
	br bool %44, label %45, label %44

; <label>:45		; preds = %43, %44
	add uint %48, 17		; <uint>:58 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %58		; <ubyte*>:57 [#uses=1]
	load ubyte* %57		; <ubyte>:87 [#uses=1]
	seteq ubyte %87, 0		; <bool>:45 [#uses=1]
	br bool %45, label %47, label %46

; <label>:46		; preds = %45, %46
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %56		; <ubyte*>:58 [#uses=2]
	load ubyte* %58		; <ubyte>:88 [#uses=1]
	add ubyte %88, 1		; <ubyte>:89 [#uses=1]
	store ubyte %89, ubyte* %58
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %58		; <ubyte*>:59 [#uses=2]
	load ubyte* %59		; <ubyte>:90 [#uses=2]
	add ubyte %90, 255		; <ubyte>:91 [#uses=1]
	store ubyte %91, ubyte* %59
	seteq ubyte %90, 1		; <bool>:46 [#uses=1]
	br bool %46, label %47, label %46

; <label>:47		; preds = %45, %46
	add uint %48, 124		; <uint>:59 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %59		; <ubyte*>:60 [#uses=1]
	load ubyte* %60		; <ubyte>:92 [#uses=1]
	seteq ubyte %92, 0		; <bool>:47 [#uses=1]
	br bool %47, label %49, label %48

; <label>:48		; preds = %47, %48
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %59		; <ubyte*>:61 [#uses=2]
	load ubyte* %61		; <ubyte>:93 [#uses=2]
	add ubyte %93, 255		; <ubyte>:94 [#uses=1]
	store ubyte %94, ubyte* %61
	seteq ubyte %93, 1		; <bool>:48 [#uses=1]
	br bool %48, label %49, label %48

; <label>:49		; preds = %47, %48
	add uint %48, 22		; <uint>:60 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %60		; <ubyte*>:62 [#uses=1]
	load ubyte* %62		; <ubyte>:95 [#uses=1]
	seteq ubyte %95, 0		; <bool>:49 [#uses=1]
	br bool %49, label %51, label %50

; <label>:50		; preds = %49, %50
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %60		; <ubyte*>:63 [#uses=2]
	load ubyte* %63		; <ubyte>:96 [#uses=1]
	add ubyte %96, 255		; <ubyte>:97 [#uses=1]
	store ubyte %97, ubyte* %63
	add uint %48, 23		; <uint>:61 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %61		; <ubyte*>:64 [#uses=2]
	load ubyte* %64		; <ubyte>:98 [#uses=1]
	add ubyte %98, 1		; <ubyte>:99 [#uses=1]
	store ubyte %99, ubyte* %64
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %59		; <ubyte*>:65 [#uses=2]
	load ubyte* %65		; <ubyte>:100 [#uses=1]
	add ubyte %100, 1		; <ubyte>:101 [#uses=1]
	store ubyte %101, ubyte* %65
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %60		; <ubyte*>:66 [#uses=1]
	load ubyte* %66		; <ubyte>:102 [#uses=1]
	seteq ubyte %102, 0		; <bool>:50 [#uses=1]
	br bool %50, label %51, label %50

; <label>:51		; preds = %49, %50
	add uint %48, 23		; <uint>:62 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %62		; <ubyte*>:67 [#uses=1]
	load ubyte* %67		; <ubyte>:103 [#uses=1]
	seteq ubyte %103, 0		; <bool>:51 [#uses=1]
	br bool %51, label %53, label %52

; <label>:52		; preds = %51, %52
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %60		; <ubyte*>:68 [#uses=2]
	load ubyte* %68		; <ubyte>:104 [#uses=1]
	add ubyte %104, 1		; <ubyte>:105 [#uses=1]
	store ubyte %105, ubyte* %68
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %62		; <ubyte*>:69 [#uses=2]
	load ubyte* %69		; <ubyte>:106 [#uses=2]
	add ubyte %106, 255		; <ubyte>:107 [#uses=1]
	store ubyte %107, ubyte* %69
	seteq ubyte %106, 1		; <bool>:52 [#uses=1]
	br bool %52, label %53, label %52

; <label>:53		; preds = %51, %52
	add uint %48, 130		; <uint>:63 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %63		; <ubyte*>:70 [#uses=1]
	load ubyte* %70		; <ubyte>:108 [#uses=1]
	seteq ubyte %108, 0		; <bool>:53 [#uses=1]
	br bool %53, label %55, label %54

; <label>:54		; preds = %53, %54
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %63		; <ubyte*>:71 [#uses=2]
	load ubyte* %71		; <ubyte>:109 [#uses=2]
	add ubyte %109, 255		; <ubyte>:110 [#uses=1]
	store ubyte %110, ubyte* %71
	seteq ubyte %109, 1		; <bool>:54 [#uses=1]
	br bool %54, label %55, label %54

; <label>:55		; preds = %53, %54
	add uint %48, 28		; <uint>:64 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %64		; <ubyte*>:72 [#uses=1]
	load ubyte* %72		; <ubyte>:111 [#uses=1]
	seteq ubyte %111, 0		; <bool>:55 [#uses=1]
	br bool %55, label %57, label %56

; <label>:56		; preds = %55, %56
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %64		; <ubyte*>:73 [#uses=2]
	load ubyte* %73		; <ubyte>:112 [#uses=1]
	add ubyte %112, 255		; <ubyte>:113 [#uses=1]
	store ubyte %113, ubyte* %73
	add uint %48, 29		; <uint>:65 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %65		; <ubyte*>:74 [#uses=2]
	load ubyte* %74		; <ubyte>:114 [#uses=1]
	add ubyte %114, 1		; <ubyte>:115 [#uses=1]
	store ubyte %115, ubyte* %74
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %63		; <ubyte*>:75 [#uses=2]
	load ubyte* %75		; <ubyte>:116 [#uses=1]
	add ubyte %116, 1		; <ubyte>:117 [#uses=1]
	store ubyte %117, ubyte* %75
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %64		; <ubyte*>:76 [#uses=1]
	load ubyte* %76		; <ubyte>:118 [#uses=1]
	seteq ubyte %118, 0		; <bool>:56 [#uses=1]
	br bool %56, label %57, label %56

; <label>:57		; preds = %55, %56
	add uint %48, 29		; <uint>:66 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %66		; <ubyte*>:77 [#uses=1]
	load ubyte* %77		; <ubyte>:119 [#uses=1]
	seteq ubyte %119, 0		; <bool>:57 [#uses=1]
	br bool %57, label %59, label %58

; <label>:58		; preds = %57, %58
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %64		; <ubyte*>:78 [#uses=2]
	load ubyte* %78		; <ubyte>:120 [#uses=1]
	add ubyte %120, 1		; <ubyte>:121 [#uses=1]
	store ubyte %121, ubyte* %78
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %66		; <ubyte*>:79 [#uses=2]
	load ubyte* %79		; <ubyte>:122 [#uses=2]
	add ubyte %122, 255		; <ubyte>:123 [#uses=1]
	store ubyte %123, ubyte* %79
	seteq ubyte %122, 1		; <bool>:58 [#uses=1]
	br bool %58, label %59, label %58

; <label>:59		; preds = %57, %58
	add uint %48, 136		; <uint>:67 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %67		; <ubyte*>:80 [#uses=1]
	load ubyte* %80		; <ubyte>:124 [#uses=1]
	seteq ubyte %124, 0		; <bool>:59 [#uses=1]
	br bool %59, label %61, label %60

; <label>:60		; preds = %59, %60
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %67		; <ubyte*>:81 [#uses=2]
	load ubyte* %81		; <ubyte>:125 [#uses=2]
	add ubyte %125, 255		; <ubyte>:126 [#uses=1]
	store ubyte %126, ubyte* %81
	seteq ubyte %125, 1		; <bool>:60 [#uses=1]
	br bool %60, label %61, label %60

; <label>:61		; preds = %59, %60
	add uint %48, 34		; <uint>:68 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %68		; <ubyte*>:82 [#uses=1]
	load ubyte* %82		; <ubyte>:127 [#uses=1]
	seteq ubyte %127, 0		; <bool>:61 [#uses=1]
	br bool %61, label %63, label %62

; <label>:62		; preds = %61, %62
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %68		; <ubyte*>:83 [#uses=2]
	load ubyte* %83		; <ubyte>:128 [#uses=1]
	add ubyte %128, 255		; <ubyte>:129 [#uses=1]
	store ubyte %129, ubyte* %83
	add uint %48, 35		; <uint>:69 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %69		; <ubyte*>:84 [#uses=2]
	load ubyte* %84		; <ubyte>:130 [#uses=1]
	add ubyte %130, 1		; <ubyte>:131 [#uses=1]
	store ubyte %131, ubyte* %84
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %67		; <ubyte*>:85 [#uses=2]
	load ubyte* %85		; <ubyte>:132 [#uses=1]
	add ubyte %132, 1		; <ubyte>:133 [#uses=1]
	store ubyte %133, ubyte* %85
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %68		; <ubyte*>:86 [#uses=1]
	load ubyte* %86		; <ubyte>:134 [#uses=1]
	seteq ubyte %134, 0		; <bool>:62 [#uses=1]
	br bool %62, label %63, label %62

; <label>:63		; preds = %61, %62
	add uint %48, 35		; <uint>:70 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %70		; <ubyte*>:87 [#uses=1]
	load ubyte* %87		; <ubyte>:135 [#uses=1]
	seteq ubyte %135, 0		; <bool>:63 [#uses=1]
	br bool %63, label %65, label %64

; <label>:64		; preds = %63, %64
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %68		; <ubyte*>:88 [#uses=2]
	load ubyte* %88		; <ubyte>:136 [#uses=1]
	add ubyte %136, 1		; <ubyte>:137 [#uses=1]
	store ubyte %137, ubyte* %88
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %70		; <ubyte*>:89 [#uses=2]
	load ubyte* %89		; <ubyte>:138 [#uses=2]
	add ubyte %138, 255		; <ubyte>:139 [#uses=1]
	store ubyte %139, ubyte* %89
	seteq ubyte %138, 1		; <bool>:64 [#uses=1]
	br bool %64, label %65, label %64

; <label>:65		; preds = %63, %64
	add uint %48, 142		; <uint>:71 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %71		; <ubyte*>:90 [#uses=1]
	load ubyte* %90		; <ubyte>:140 [#uses=1]
	seteq ubyte %140, 0		; <bool>:65 [#uses=1]
	br bool %65, label %67, label %66

; <label>:66		; preds = %65, %66
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %71		; <ubyte*>:91 [#uses=2]
	load ubyte* %91		; <ubyte>:141 [#uses=2]
	add ubyte %141, 255		; <ubyte>:142 [#uses=1]
	store ubyte %142, ubyte* %91
	seteq ubyte %141, 1		; <bool>:66 [#uses=1]
	br bool %66, label %67, label %66

; <label>:67		; preds = %65, %66
	add uint %48, 40		; <uint>:72 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %72		; <ubyte*>:92 [#uses=1]
	load ubyte* %92		; <ubyte>:143 [#uses=1]
	seteq ubyte %143, 0		; <bool>:67 [#uses=1]
	br bool %67, label %69, label %68

; <label>:68		; preds = %67, %68
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %72		; <ubyte*>:93 [#uses=2]
	load ubyte* %93		; <ubyte>:144 [#uses=1]
	add ubyte %144, 255		; <ubyte>:145 [#uses=1]
	store ubyte %145, ubyte* %93
	add uint %48, 41		; <uint>:73 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %73		; <ubyte*>:94 [#uses=2]
	load ubyte* %94		; <ubyte>:146 [#uses=1]
	add ubyte %146, 1		; <ubyte>:147 [#uses=1]
	store ubyte %147, ubyte* %94
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %71		; <ubyte*>:95 [#uses=2]
	load ubyte* %95		; <ubyte>:148 [#uses=1]
	add ubyte %148, 1		; <ubyte>:149 [#uses=1]
	store ubyte %149, ubyte* %95
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %72		; <ubyte*>:96 [#uses=1]
	load ubyte* %96		; <ubyte>:150 [#uses=1]
	seteq ubyte %150, 0		; <bool>:68 [#uses=1]
	br bool %68, label %69, label %68

; <label>:69		; preds = %67, %68
	add uint %48, 41		; <uint>:74 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %74		; <ubyte*>:97 [#uses=1]
	load ubyte* %97		; <ubyte>:151 [#uses=1]
	seteq ubyte %151, 0		; <bool>:69 [#uses=1]
	br bool %69, label %71, label %70

; <label>:70		; preds = %69, %70
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %72		; <ubyte*>:98 [#uses=2]
	load ubyte* %98		; <ubyte>:152 [#uses=1]
	add ubyte %152, 1		; <ubyte>:153 [#uses=1]
	store ubyte %153, ubyte* %98
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %74		; <ubyte*>:99 [#uses=2]
	load ubyte* %99		; <ubyte>:154 [#uses=2]
	add ubyte %154, 255		; <ubyte>:155 [#uses=1]
	store ubyte %155, ubyte* %99
	seteq ubyte %154, 1		; <bool>:70 [#uses=1]
	br bool %70, label %71, label %70

; <label>:71		; preds = %69, %70
	add uint %48, 148		; <uint>:75 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %75		; <ubyte*>:100 [#uses=1]
	load ubyte* %100		; <ubyte>:156 [#uses=1]
	seteq ubyte %156, 0		; <bool>:71 [#uses=1]
	br bool %71, label %73, label %72

; <label>:72		; preds = %71, %72
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %75		; <ubyte*>:101 [#uses=2]
	load ubyte* %101		; <ubyte>:157 [#uses=2]
	add ubyte %157, 255		; <ubyte>:158 [#uses=1]
	store ubyte %158, ubyte* %101
	seteq ubyte %157, 1		; <bool>:72 [#uses=1]
	br bool %72, label %73, label %72

; <label>:73		; preds = %71, %72
	add uint %48, 46		; <uint>:76 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %76		; <ubyte*>:102 [#uses=1]
	load ubyte* %102		; <ubyte>:159 [#uses=1]
	seteq ubyte %159, 0		; <bool>:73 [#uses=1]
	br bool %73, label %75, label %74

; <label>:74		; preds = %73, %74
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %76		; <ubyte*>:103 [#uses=2]
	load ubyte* %103		; <ubyte>:160 [#uses=1]
	add ubyte %160, 255		; <ubyte>:161 [#uses=1]
	store ubyte %161, ubyte* %103
	add uint %48, 47		; <uint>:77 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %77		; <ubyte*>:104 [#uses=2]
	load ubyte* %104		; <ubyte>:162 [#uses=1]
	add ubyte %162, 1		; <ubyte>:163 [#uses=1]
	store ubyte %163, ubyte* %104
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %75		; <ubyte*>:105 [#uses=2]
	load ubyte* %105		; <ubyte>:164 [#uses=1]
	add ubyte %164, 1		; <ubyte>:165 [#uses=1]
	store ubyte %165, ubyte* %105
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %76		; <ubyte*>:106 [#uses=1]
	load ubyte* %106		; <ubyte>:166 [#uses=1]
	seteq ubyte %166, 0		; <bool>:74 [#uses=1]
	br bool %74, label %75, label %74

; <label>:75		; preds = %73, %74
	add uint %48, 47		; <uint>:78 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %78		; <ubyte*>:107 [#uses=1]
	load ubyte* %107		; <ubyte>:167 [#uses=1]
	seteq ubyte %167, 0		; <bool>:75 [#uses=1]
	br bool %75, label %77, label %76

; <label>:76		; preds = %75, %76
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %76		; <ubyte*>:108 [#uses=2]
	load ubyte* %108		; <ubyte>:168 [#uses=1]
	add ubyte %168, 1		; <ubyte>:169 [#uses=1]
	store ubyte %169, ubyte* %108
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %78		; <ubyte*>:109 [#uses=2]
	load ubyte* %109		; <ubyte>:170 [#uses=2]
	add ubyte %170, 255		; <ubyte>:171 [#uses=1]
	store ubyte %171, ubyte* %109
	seteq ubyte %170, 1		; <bool>:76 [#uses=1]
	br bool %76, label %77, label %76

; <label>:77		; preds = %75, %76
	add uint %48, 154		; <uint>:79 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %79		; <ubyte*>:110 [#uses=1]
	load ubyte* %110		; <ubyte>:172 [#uses=1]
	seteq ubyte %172, 0		; <bool>:77 [#uses=1]
	br bool %77, label %79, label %78

; <label>:78		; preds = %77, %78
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %79		; <ubyte*>:111 [#uses=2]
	load ubyte* %111		; <ubyte>:173 [#uses=2]
	add ubyte %173, 255		; <ubyte>:174 [#uses=1]
	store ubyte %174, ubyte* %111
	seteq ubyte %173, 1		; <bool>:78 [#uses=1]
	br bool %78, label %79, label %78

; <label>:79		; preds = %77, %78
	add uint %48, 52		; <uint>:80 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %80		; <ubyte*>:112 [#uses=1]
	load ubyte* %112		; <ubyte>:175 [#uses=1]
	seteq ubyte %175, 0		; <bool>:79 [#uses=1]
	br bool %79, label %81, label %80

; <label>:80		; preds = %79, %80
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %80		; <ubyte*>:113 [#uses=2]
	load ubyte* %113		; <ubyte>:176 [#uses=1]
	add ubyte %176, 255		; <ubyte>:177 [#uses=1]
	store ubyte %177, ubyte* %113
	add uint %48, 53		; <uint>:81 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %81		; <ubyte*>:114 [#uses=2]
	load ubyte* %114		; <ubyte>:178 [#uses=1]
	add ubyte %178, 1		; <ubyte>:179 [#uses=1]
	store ubyte %179, ubyte* %114
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %79		; <ubyte*>:115 [#uses=2]
	load ubyte* %115		; <ubyte>:180 [#uses=1]
	add ubyte %180, 1		; <ubyte>:181 [#uses=1]
	store ubyte %181, ubyte* %115
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %80		; <ubyte*>:116 [#uses=1]
	load ubyte* %116		; <ubyte>:182 [#uses=1]
	seteq ubyte %182, 0		; <bool>:80 [#uses=1]
	br bool %80, label %81, label %80

; <label>:81		; preds = %79, %80
	add uint %48, 53		; <uint>:82 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %82		; <ubyte*>:117 [#uses=1]
	load ubyte* %117		; <ubyte>:183 [#uses=1]
	seteq ubyte %183, 0		; <bool>:81 [#uses=1]
	br bool %81, label %83, label %82

; <label>:82		; preds = %81, %82
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %80		; <ubyte*>:118 [#uses=2]
	load ubyte* %118		; <ubyte>:184 [#uses=1]
	add ubyte %184, 1		; <ubyte>:185 [#uses=1]
	store ubyte %185, ubyte* %118
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %82		; <ubyte*>:119 [#uses=2]
	load ubyte* %119		; <ubyte>:186 [#uses=2]
	add ubyte %186, 255		; <ubyte>:187 [#uses=1]
	store ubyte %187, ubyte* %119
	seteq ubyte %186, 1		; <bool>:82 [#uses=1]
	br bool %82, label %83, label %82

; <label>:83		; preds = %81, %82
	add uint %48, 160		; <uint>:83 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %83		; <ubyte*>:120 [#uses=1]
	load ubyte* %120		; <ubyte>:188 [#uses=1]
	seteq ubyte %188, 0		; <bool>:83 [#uses=1]
	br bool %83, label %85, label %84

; <label>:84		; preds = %83, %84
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %83		; <ubyte*>:121 [#uses=2]
	load ubyte* %121		; <ubyte>:189 [#uses=2]
	add ubyte %189, 255		; <ubyte>:190 [#uses=1]
	store ubyte %190, ubyte* %121
	seteq ubyte %189, 1		; <bool>:84 [#uses=1]
	br bool %84, label %85, label %84

; <label>:85		; preds = %83, %84
	add uint %48, 58		; <uint>:84 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %84		; <ubyte*>:122 [#uses=1]
	load ubyte* %122		; <ubyte>:191 [#uses=1]
	seteq ubyte %191, 0		; <bool>:85 [#uses=1]
	br bool %85, label %87, label %86

; <label>:86		; preds = %85, %86
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %84		; <ubyte*>:123 [#uses=2]
	load ubyte* %123		; <ubyte>:192 [#uses=1]
	add ubyte %192, 255		; <ubyte>:193 [#uses=1]
	store ubyte %193, ubyte* %123
	add uint %48, 59		; <uint>:85 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %85		; <ubyte*>:124 [#uses=2]
	load ubyte* %124		; <ubyte>:194 [#uses=1]
	add ubyte %194, 1		; <ubyte>:195 [#uses=1]
	store ubyte %195, ubyte* %124
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %83		; <ubyte*>:125 [#uses=2]
	load ubyte* %125		; <ubyte>:196 [#uses=1]
	add ubyte %196, 1		; <ubyte>:197 [#uses=1]
	store ubyte %197, ubyte* %125
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %84		; <ubyte*>:126 [#uses=1]
	load ubyte* %126		; <ubyte>:198 [#uses=1]
	seteq ubyte %198, 0		; <bool>:86 [#uses=1]
	br bool %86, label %87, label %86

; <label>:87		; preds = %85, %86
	add uint %48, 59		; <uint>:86 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %86		; <ubyte*>:127 [#uses=1]
	load ubyte* %127		; <ubyte>:199 [#uses=1]
	seteq ubyte %199, 0		; <bool>:87 [#uses=1]
	br bool %87, label %89, label %88

; <label>:88		; preds = %87, %88
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %84		; <ubyte*>:128 [#uses=2]
	load ubyte* %128		; <ubyte>:200 [#uses=1]
	add ubyte %200, 1		; <ubyte>:201 [#uses=1]
	store ubyte %201, ubyte* %128
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %86		; <ubyte*>:129 [#uses=2]
	load ubyte* %129		; <ubyte>:202 [#uses=2]
	add ubyte %202, 255		; <ubyte>:203 [#uses=1]
	store ubyte %203, ubyte* %129
	seteq ubyte %202, 1		; <bool>:88 [#uses=1]
	br bool %88, label %89, label %88

; <label>:89		; preds = %87, %88
	add uint %48, 166		; <uint>:87 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %87		; <ubyte*>:130 [#uses=1]
	load ubyte* %130		; <ubyte>:204 [#uses=1]
	seteq ubyte %204, 0		; <bool>:89 [#uses=1]
	br bool %89, label %91, label %90

; <label>:90		; preds = %89, %90
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %87		; <ubyte*>:131 [#uses=2]
	load ubyte* %131		; <ubyte>:205 [#uses=2]
	add ubyte %205, 255		; <ubyte>:206 [#uses=1]
	store ubyte %206, ubyte* %131
	seteq ubyte %205, 1		; <bool>:90 [#uses=1]
	br bool %90, label %91, label %90

; <label>:91		; preds = %89, %90
	add uint %48, 64		; <uint>:88 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %88		; <ubyte*>:132 [#uses=1]
	load ubyte* %132		; <ubyte>:207 [#uses=1]
	seteq ubyte %207, 0		; <bool>:91 [#uses=1]
	br bool %91, label %93, label %92

; <label>:92		; preds = %91, %92
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %88		; <ubyte*>:133 [#uses=2]
	load ubyte* %133		; <ubyte>:208 [#uses=1]
	add ubyte %208, 255		; <ubyte>:209 [#uses=1]
	store ubyte %209, ubyte* %133
	add uint %48, 65		; <uint>:89 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %89		; <ubyte*>:134 [#uses=2]
	load ubyte* %134		; <ubyte>:210 [#uses=1]
	add ubyte %210, 1		; <ubyte>:211 [#uses=1]
	store ubyte %211, ubyte* %134
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %87		; <ubyte*>:135 [#uses=2]
	load ubyte* %135		; <ubyte>:212 [#uses=1]
	add ubyte %212, 1		; <ubyte>:213 [#uses=1]
	store ubyte %213, ubyte* %135
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %88		; <ubyte*>:136 [#uses=1]
	load ubyte* %136		; <ubyte>:214 [#uses=1]
	seteq ubyte %214, 0		; <bool>:92 [#uses=1]
	br bool %92, label %93, label %92

; <label>:93		; preds = %91, %92
	add uint %48, 65		; <uint>:90 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %90		; <ubyte*>:137 [#uses=1]
	load ubyte* %137		; <ubyte>:215 [#uses=1]
	seteq ubyte %215, 0		; <bool>:93 [#uses=1]
	br bool %93, label %95, label %94

; <label>:94		; preds = %93, %94
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %88		; <ubyte*>:138 [#uses=2]
	load ubyte* %138		; <ubyte>:216 [#uses=1]
	add ubyte %216, 1		; <ubyte>:217 [#uses=1]
	store ubyte %217, ubyte* %138
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %90		; <ubyte*>:139 [#uses=2]
	load ubyte* %139		; <ubyte>:218 [#uses=2]
	add ubyte %218, 255		; <ubyte>:219 [#uses=1]
	store ubyte %219, ubyte* %139
	seteq ubyte %218, 1		; <bool>:94 [#uses=1]
	br bool %94, label %95, label %94

; <label>:95		; preds = %93, %94
	add uint %48, 172		; <uint>:91 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %91		; <ubyte*>:140 [#uses=1]
	load ubyte* %140		; <ubyte>:220 [#uses=1]
	seteq ubyte %220, 0		; <bool>:95 [#uses=1]
	br bool %95, label %97, label %96

; <label>:96		; preds = %95, %96
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %91		; <ubyte*>:141 [#uses=2]
	load ubyte* %141		; <ubyte>:221 [#uses=2]
	add ubyte %221, 255		; <ubyte>:222 [#uses=1]
	store ubyte %222, ubyte* %141
	seteq ubyte %221, 1		; <bool>:96 [#uses=1]
	br bool %96, label %97, label %96

; <label>:97		; preds = %95, %96
	add uint %48, 70		; <uint>:92 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %92		; <ubyte*>:142 [#uses=1]
	load ubyte* %142		; <ubyte>:223 [#uses=1]
	seteq ubyte %223, 0		; <bool>:97 [#uses=1]
	br bool %97, label %99, label %98

; <label>:98		; preds = %97, %98
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %92		; <ubyte*>:143 [#uses=2]
	load ubyte* %143		; <ubyte>:224 [#uses=1]
	add ubyte %224, 255		; <ubyte>:225 [#uses=1]
	store ubyte %225, ubyte* %143
	add uint %48, 71		; <uint>:93 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %93		; <ubyte*>:144 [#uses=2]
	load ubyte* %144		; <ubyte>:226 [#uses=1]
	add ubyte %226, 1		; <ubyte>:227 [#uses=1]
	store ubyte %227, ubyte* %144
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %91		; <ubyte*>:145 [#uses=2]
	load ubyte* %145		; <ubyte>:228 [#uses=1]
	add ubyte %228, 1		; <ubyte>:229 [#uses=1]
	store ubyte %229, ubyte* %145
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %92		; <ubyte*>:146 [#uses=1]
	load ubyte* %146		; <ubyte>:230 [#uses=1]
	seteq ubyte %230, 0		; <bool>:98 [#uses=1]
	br bool %98, label %99, label %98

; <label>:99		; preds = %97, %98
	add uint %48, 71		; <uint>:94 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %94		; <ubyte*>:147 [#uses=1]
	load ubyte* %147		; <ubyte>:231 [#uses=1]
	seteq ubyte %231, 0		; <bool>:99 [#uses=1]
	br bool %99, label %101, label %100

; <label>:100		; preds = %99, %100
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %92		; <ubyte*>:148 [#uses=2]
	load ubyte* %148		; <ubyte>:232 [#uses=1]
	add ubyte %232, 1		; <ubyte>:233 [#uses=1]
	store ubyte %233, ubyte* %148
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %94		; <ubyte*>:149 [#uses=2]
	load ubyte* %149		; <ubyte>:234 [#uses=2]
	add ubyte %234, 255		; <ubyte>:235 [#uses=1]
	store ubyte %235, ubyte* %149
	seteq ubyte %234, 1		; <bool>:100 [#uses=1]
	br bool %100, label %101, label %100

; <label>:101		; preds = %99, %100
	add uint %48, 178		; <uint>:95 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %95		; <ubyte*>:150 [#uses=1]
	load ubyte* %150		; <ubyte>:236 [#uses=1]
	seteq ubyte %236, 0		; <bool>:101 [#uses=1]
	br bool %101, label %103, label %102

; <label>:102		; preds = %101, %102
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %95		; <ubyte*>:151 [#uses=2]
	load ubyte* %151		; <ubyte>:237 [#uses=2]
	add ubyte %237, 255		; <ubyte>:238 [#uses=1]
	store ubyte %238, ubyte* %151
	seteq ubyte %237, 1		; <bool>:102 [#uses=1]
	br bool %102, label %103, label %102

; <label>:103		; preds = %101, %102
	add uint %48, 76		; <uint>:96 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %96		; <ubyte*>:152 [#uses=1]
	load ubyte* %152		; <ubyte>:239 [#uses=1]
	seteq ubyte %239, 0		; <bool>:103 [#uses=1]
	br bool %103, label %105, label %104

; <label>:104		; preds = %103, %104
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %96		; <ubyte*>:153 [#uses=2]
	load ubyte* %153		; <ubyte>:240 [#uses=1]
	add ubyte %240, 255		; <ubyte>:241 [#uses=1]
	store ubyte %241, ubyte* %153
	add uint %48, 77		; <uint>:97 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %97		; <ubyte*>:154 [#uses=2]
	load ubyte* %154		; <ubyte>:242 [#uses=1]
	add ubyte %242, 1		; <ubyte>:243 [#uses=1]
	store ubyte %243, ubyte* %154
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %95		; <ubyte*>:155 [#uses=2]
	load ubyte* %155		; <ubyte>:244 [#uses=1]
	add ubyte %244, 1		; <ubyte>:245 [#uses=1]
	store ubyte %245, ubyte* %155
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %96		; <ubyte*>:156 [#uses=1]
	load ubyte* %156		; <ubyte>:246 [#uses=1]
	seteq ubyte %246, 0		; <bool>:104 [#uses=1]
	br bool %104, label %105, label %104

; <label>:105		; preds = %103, %104
	add uint %48, 77		; <uint>:98 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %98		; <ubyte*>:157 [#uses=1]
	load ubyte* %157		; <ubyte>:247 [#uses=1]
	seteq ubyte %247, 0		; <bool>:105 [#uses=1]
	br bool %105, label %107, label %106

; <label>:106		; preds = %105, %106
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %96		; <ubyte*>:158 [#uses=2]
	load ubyte* %158		; <ubyte>:248 [#uses=1]
	add ubyte %248, 1		; <ubyte>:249 [#uses=1]
	store ubyte %249, ubyte* %158
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %98		; <ubyte*>:159 [#uses=2]
	load ubyte* %159		; <ubyte>:250 [#uses=2]
	add ubyte %250, 255		; <ubyte>:251 [#uses=1]
	store ubyte %251, ubyte* %159
	seteq ubyte %250, 1		; <bool>:106 [#uses=1]
	br bool %106, label %107, label %106

; <label>:107		; preds = %105, %106
	add uint %48, 184		; <uint>:99 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %99		; <ubyte*>:160 [#uses=1]
	load ubyte* %160		; <ubyte>:252 [#uses=1]
	seteq ubyte %252, 0		; <bool>:107 [#uses=1]
	br bool %107, label %109, label %108

; <label>:108		; preds = %107, %108
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %99		; <ubyte*>:161 [#uses=2]
	load ubyte* %161		; <ubyte>:253 [#uses=2]
	add ubyte %253, 255		; <ubyte>:254 [#uses=1]
	store ubyte %254, ubyte* %161
	seteq ubyte %253, 1		; <bool>:108 [#uses=1]
	br bool %108, label %109, label %108

; <label>:109		; preds = %107, %108
	add uint %48, 82		; <uint>:100 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %100		; <ubyte*>:162 [#uses=1]
	load ubyte* %162		; <ubyte>:255 [#uses=1]
	seteq ubyte %255, 0		; <bool>:109 [#uses=1]
	br bool %109, label %111, label %110

; <label>:110		; preds = %109, %110
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %100		; <ubyte*>:163 [#uses=2]
	load ubyte* %163		; <ubyte>:256 [#uses=1]
	add ubyte %256, 255		; <ubyte>:257 [#uses=1]
	store ubyte %257, ubyte* %163
	add uint %48, 83		; <uint>:101 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %101		; <ubyte*>:164 [#uses=2]
	load ubyte* %164		; <ubyte>:258 [#uses=1]
	add ubyte %258, 1		; <ubyte>:259 [#uses=1]
	store ubyte %259, ubyte* %164
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %99		; <ubyte*>:165 [#uses=2]
	load ubyte* %165		; <ubyte>:260 [#uses=1]
	add ubyte %260, 1		; <ubyte>:261 [#uses=1]
	store ubyte %261, ubyte* %165
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %100		; <ubyte*>:166 [#uses=1]
	load ubyte* %166		; <ubyte>:262 [#uses=1]
	seteq ubyte %262, 0		; <bool>:110 [#uses=1]
	br bool %110, label %111, label %110

; <label>:111		; preds = %109, %110
	add uint %48, 83		; <uint>:102 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %102		; <ubyte*>:167 [#uses=1]
	load ubyte* %167		; <ubyte>:263 [#uses=1]
	seteq ubyte %263, 0		; <bool>:111 [#uses=1]
	br bool %111, label %113, label %112

; <label>:112		; preds = %111, %112
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %100		; <ubyte*>:168 [#uses=2]
	load ubyte* %168		; <ubyte>:264 [#uses=1]
	add ubyte %264, 1		; <ubyte>:265 [#uses=1]
	store ubyte %265, ubyte* %168
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %102		; <ubyte*>:169 [#uses=2]
	load ubyte* %169		; <ubyte>:266 [#uses=2]
	add ubyte %266, 255		; <ubyte>:267 [#uses=1]
	store ubyte %267, ubyte* %169
	seteq ubyte %266, 1		; <bool>:112 [#uses=1]
	br bool %112, label %113, label %112

; <label>:113		; preds = %111, %112
	add uint %48, 190		; <uint>:103 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %103		; <ubyte*>:170 [#uses=1]
	load ubyte* %170		; <ubyte>:268 [#uses=1]
	seteq ubyte %268, 0		; <bool>:113 [#uses=1]
	br bool %113, label %115, label %114

; <label>:114		; preds = %113, %114
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %103		; <ubyte*>:171 [#uses=2]
	load ubyte* %171		; <ubyte>:269 [#uses=2]
	add ubyte %269, 255		; <ubyte>:270 [#uses=1]
	store ubyte %270, ubyte* %171
	seteq ubyte %269, 1		; <bool>:114 [#uses=1]
	br bool %114, label %115, label %114

; <label>:115		; preds = %113, %114
	add uint %48, 88		; <uint>:104 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %104		; <ubyte*>:172 [#uses=1]
	load ubyte* %172		; <ubyte>:271 [#uses=1]
	seteq ubyte %271, 0		; <bool>:115 [#uses=1]
	br bool %115, label %117, label %116

; <label>:116		; preds = %115, %116
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %104		; <ubyte*>:173 [#uses=2]
	load ubyte* %173		; <ubyte>:272 [#uses=1]
	add ubyte %272, 255		; <ubyte>:273 [#uses=1]
	store ubyte %273, ubyte* %173
	add uint %48, 89		; <uint>:105 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %105		; <ubyte*>:174 [#uses=2]
	load ubyte* %174		; <ubyte>:274 [#uses=1]
	add ubyte %274, 1		; <ubyte>:275 [#uses=1]
	store ubyte %275, ubyte* %174
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %103		; <ubyte*>:175 [#uses=2]
	load ubyte* %175		; <ubyte>:276 [#uses=1]
	add ubyte %276, 1		; <ubyte>:277 [#uses=1]
	store ubyte %277, ubyte* %175
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %104		; <ubyte*>:176 [#uses=1]
	load ubyte* %176		; <ubyte>:278 [#uses=1]
	seteq ubyte %278, 0		; <bool>:116 [#uses=1]
	br bool %116, label %117, label %116

; <label>:117		; preds = %115, %116
	add uint %48, 89		; <uint>:106 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %106		; <ubyte*>:177 [#uses=1]
	load ubyte* %177		; <ubyte>:279 [#uses=1]
	seteq ubyte %279, 0		; <bool>:117 [#uses=1]
	br bool %117, label %119, label %118

; <label>:118		; preds = %117, %118
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %104		; <ubyte*>:178 [#uses=2]
	load ubyte* %178		; <ubyte>:280 [#uses=1]
	add ubyte %280, 1		; <ubyte>:281 [#uses=1]
	store ubyte %281, ubyte* %178
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %106		; <ubyte*>:179 [#uses=2]
	load ubyte* %179		; <ubyte>:282 [#uses=2]
	add ubyte %282, 255		; <ubyte>:283 [#uses=1]
	store ubyte %283, ubyte* %179
	seteq ubyte %282, 1		; <bool>:118 [#uses=1]
	br bool %118, label %119, label %118

; <label>:119		; preds = %117, %118
	add uint %48, 196		; <uint>:107 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %107		; <ubyte*>:180 [#uses=1]
	load ubyte* %180		; <ubyte>:284 [#uses=1]
	seteq ubyte %284, 0		; <bool>:119 [#uses=1]
	br bool %119, label %121, label %120

; <label>:120		; preds = %119, %120
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %107		; <ubyte*>:181 [#uses=2]
	load ubyte* %181		; <ubyte>:285 [#uses=2]
	add ubyte %285, 255		; <ubyte>:286 [#uses=1]
	store ubyte %286, ubyte* %181
	seteq ubyte %285, 1		; <bool>:120 [#uses=1]
	br bool %120, label %121, label %120

; <label>:121		; preds = %119, %120
	add uint %48, 94		; <uint>:108 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %108		; <ubyte*>:182 [#uses=1]
	load ubyte* %182		; <ubyte>:287 [#uses=1]
	seteq ubyte %287, 0		; <bool>:121 [#uses=1]
	br bool %121, label %123, label %122

; <label>:122		; preds = %121, %122
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %108		; <ubyte*>:183 [#uses=2]
	load ubyte* %183		; <ubyte>:288 [#uses=1]
	add ubyte %288, 255		; <ubyte>:289 [#uses=1]
	store ubyte %289, ubyte* %183
	add uint %48, 95		; <uint>:109 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %109		; <ubyte*>:184 [#uses=2]
	load ubyte* %184		; <ubyte>:290 [#uses=1]
	add ubyte %290, 1		; <ubyte>:291 [#uses=1]
	store ubyte %291, ubyte* %184
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %107		; <ubyte*>:185 [#uses=2]
	load ubyte* %185		; <ubyte>:292 [#uses=1]
	add ubyte %292, 1		; <ubyte>:293 [#uses=1]
	store ubyte %293, ubyte* %185
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %108		; <ubyte*>:186 [#uses=1]
	load ubyte* %186		; <ubyte>:294 [#uses=1]
	seteq ubyte %294, 0		; <bool>:122 [#uses=1]
	br bool %122, label %123, label %122

; <label>:123		; preds = %121, %122
	add uint %48, 95		; <uint>:110 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %110		; <ubyte*>:187 [#uses=1]
	load ubyte* %187		; <ubyte>:295 [#uses=1]
	seteq ubyte %295, 0		; <bool>:123 [#uses=1]
	br bool %123, label %125, label %124

; <label>:124		; preds = %123, %124
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %108		; <ubyte*>:188 [#uses=2]
	load ubyte* %188		; <ubyte>:296 [#uses=1]
	add ubyte %296, 1		; <ubyte>:297 [#uses=1]
	store ubyte %297, ubyte* %188
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %110		; <ubyte*>:189 [#uses=2]
	load ubyte* %189		; <ubyte>:298 [#uses=2]
	add ubyte %298, 255		; <ubyte>:299 [#uses=1]
	store ubyte %299, ubyte* %189
	seteq ubyte %298, 1		; <bool>:124 [#uses=1]
	br bool %124, label %125, label %124

; <label>:125		; preds = %123, %124
	add uint %48, 200		; <uint>:111 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %111		; <ubyte*>:190 [#uses=1]
	load ubyte* %190		; <ubyte>:300 [#uses=1]
	seteq ubyte %300, 0		; <bool>:125 [#uses=1]
	br bool %125, label %127, label %126

; <label>:126		; preds = %125, %126
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %111		; <ubyte*>:191 [#uses=2]
	load ubyte* %191		; <ubyte>:301 [#uses=2]
	add ubyte %301, 255		; <ubyte>:302 [#uses=1]
	store ubyte %302, ubyte* %191
	seteq ubyte %301, 1		; <bool>:126 [#uses=1]
	br bool %126, label %127, label %126

; <label>:127		; preds = %125, %126
	add uint %48, 100		; <uint>:112 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %112		; <ubyte*>:192 [#uses=1]
	load ubyte* %192		; <ubyte>:303 [#uses=1]
	seteq ubyte %303, 0		; <bool>:127 [#uses=1]
	br bool %127, label %129, label %128

; <label>:128		; preds = %127, %128
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %112		; <ubyte*>:193 [#uses=2]
	load ubyte* %193		; <ubyte>:304 [#uses=1]
	add ubyte %304, 255		; <ubyte>:305 [#uses=1]
	store ubyte %305, ubyte* %193
	add uint %48, 101		; <uint>:113 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %113		; <ubyte*>:194 [#uses=2]
	load ubyte* %194		; <ubyte>:306 [#uses=1]
	add ubyte %306, 1		; <ubyte>:307 [#uses=1]
	store ubyte %307, ubyte* %194
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %111		; <ubyte*>:195 [#uses=2]
	load ubyte* %195		; <ubyte>:308 [#uses=1]
	add ubyte %308, 1		; <ubyte>:309 [#uses=1]
	store ubyte %309, ubyte* %195
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %112		; <ubyte*>:196 [#uses=1]
	load ubyte* %196		; <ubyte>:310 [#uses=1]
	seteq ubyte %310, 0		; <bool>:128 [#uses=1]
	br bool %128, label %129, label %128

; <label>:129		; preds = %127, %128
	add uint %48, 101		; <uint>:114 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %114		; <ubyte*>:197 [#uses=1]
	load ubyte* %197		; <ubyte>:311 [#uses=1]
	seteq ubyte %311, 0		; <bool>:129 [#uses=1]
	br bool %129, label %131, label %130

; <label>:130		; preds = %129, %130
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %112		; <ubyte*>:198 [#uses=2]
	load ubyte* %198		; <ubyte>:312 [#uses=1]
	add ubyte %312, 1		; <ubyte>:313 [#uses=1]
	store ubyte %313, ubyte* %198
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %114		; <ubyte*>:199 [#uses=2]
	load ubyte* %199		; <ubyte>:314 [#uses=2]
	add ubyte %314, 255		; <ubyte>:315 [#uses=1]
	store ubyte %315, ubyte* %199
	seteq ubyte %314, 1		; <bool>:130 [#uses=1]
	br bool %130, label %131, label %130

; <label>:131		; preds = %129, %130
	add uint %48, 114		; <uint>:115 [#uses=6]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %115		; <ubyte*>:200 [#uses=1]
	load ubyte* %200		; <ubyte>:316 [#uses=1]
	seteq ubyte %316, 0		; <bool>:131 [#uses=1]
	br bool %131, label %133, label %132

; <label>:132		; preds = %131, %132
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %115		; <ubyte*>:201 [#uses=2]
	load ubyte* %201		; <ubyte>:317 [#uses=2]
	add ubyte %317, 255		; <ubyte>:318 [#uses=1]
	store ubyte %318, ubyte* %201
	seteq ubyte %317, 1		; <bool>:132 [#uses=1]
	br bool %132, label %133, label %132

; <label>:133		; preds = %131, %132
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %111		; <ubyte*>:202 [#uses=1]
	load ubyte* %202		; <ubyte>:319 [#uses=1]
	seteq ubyte %319, 0		; <bool>:133 [#uses=1]
	br bool %133, label %135, label %134

; <label>:134		; preds = %133, %134
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %115		; <ubyte*>:203 [#uses=2]
	load ubyte* %203		; <ubyte>:320 [#uses=1]
	add ubyte %320, 1		; <ubyte>:321 [#uses=1]
	store ubyte %321, ubyte* %203
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %111		; <ubyte*>:204 [#uses=2]
	load ubyte* %204		; <ubyte>:322 [#uses=2]
	add ubyte %322, 255		; <ubyte>:323 [#uses=1]
	store ubyte %323, ubyte* %204
	seteq ubyte %322, 1		; <bool>:134 [#uses=1]
	br bool %134, label %135, label %134

; <label>:135		; preds = %133, %134
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %115		; <ubyte*>:205 [#uses=1]
	load ubyte* %205		; <ubyte>:324 [#uses=1]
	seteq ubyte %324, 0		; <bool>:135 [#uses=1]
	br bool %135, label %137, label %136

; <label>:136		; preds = %135, %139
	phi uint [ %115, %135 ], [ %120, %139 ]		; <uint>:116 [#uses=6]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %116		; <ubyte*>:206 [#uses=1]
	load ubyte* %206		; <ubyte>:325 [#uses=1]
	seteq ubyte %325, 0		; <bool>:136 [#uses=1]
	br bool %136, label %139, label %138

; <label>:137		; preds = %135, %139
	phi uint [ %115, %135 ], [ %120, %139 ]		; <uint>:117 [#uses=7]
	add uint %117, 4294967292		; <uint>:118 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %118		; <ubyte*>:207 [#uses=1]
	load ubyte* %207		; <ubyte>:326 [#uses=1]
	seteq ubyte %326, 0		; <bool>:137 [#uses=1]
	br bool %137, label %141, label %140

; <label>:138		; preds = %136, %138
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %116		; <ubyte*>:208 [#uses=2]
	load ubyte* %208		; <ubyte>:327 [#uses=1]
	add ubyte %327, 255		; <ubyte>:328 [#uses=1]
	store ubyte %328, ubyte* %208
	add uint %116, 6		; <uint>:119 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %119		; <ubyte*>:209 [#uses=2]
	load ubyte* %209		; <ubyte>:329 [#uses=1]
	add ubyte %329, 1		; <ubyte>:330 [#uses=1]
	store ubyte %330, ubyte* %209
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %116		; <ubyte*>:210 [#uses=1]
	load ubyte* %210		; <ubyte>:331 [#uses=1]
	seteq ubyte %331, 0		; <bool>:138 [#uses=1]
	br bool %138, label %139, label %138

; <label>:139		; preds = %136, %138
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %116		; <ubyte*>:211 [#uses=2]
	load ubyte* %211		; <ubyte>:332 [#uses=1]
	add ubyte %332, 1		; <ubyte>:333 [#uses=1]
	store ubyte %333, ubyte* %211
	add uint %116, 6		; <uint>:120 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %120		; <ubyte*>:212 [#uses=2]
	load ubyte* %212		; <ubyte>:334 [#uses=2]
	add ubyte %334, 255		; <ubyte>:335 [#uses=1]
	store ubyte %335, ubyte* %212
	seteq ubyte %334, 1		; <bool>:139 [#uses=1]
	br bool %139, label %137, label %136

; <label>:140		; preds = %137, %140
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %118		; <ubyte*>:213 [#uses=2]
	load ubyte* %213		; <ubyte>:336 [#uses=2]
	add ubyte %336, 255		; <ubyte>:337 [#uses=1]
	store ubyte %337, ubyte* %213
	seteq ubyte %336, 1		; <bool>:140 [#uses=1]
	br bool %140, label %141, label %140

; <label>:141		; preds = %137, %140
	add uint %117, 4294967294		; <uint>:121 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %121		; <ubyte*>:214 [#uses=1]
	load ubyte* %214		; <ubyte>:338 [#uses=1]
	seteq ubyte %338, 0		; <bool>:141 [#uses=1]
	br bool %141, label %143, label %142

; <label>:142		; preds = %141, %142
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %118		; <ubyte*>:215 [#uses=2]
	load ubyte* %215		; <ubyte>:339 [#uses=1]
	add ubyte %339, 1		; <ubyte>:340 [#uses=1]
	store ubyte %340, ubyte* %215
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %121		; <ubyte*>:216 [#uses=2]
	load ubyte* %216		; <ubyte>:341 [#uses=1]
	add ubyte %341, 255		; <ubyte>:342 [#uses=1]
	store ubyte %342, ubyte* %216
	add uint %117, 4294967295		; <uint>:122 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %122		; <ubyte*>:217 [#uses=2]
	load ubyte* %217		; <ubyte>:343 [#uses=1]
	add ubyte %343, 1		; <ubyte>:344 [#uses=1]
	store ubyte %344, ubyte* %217
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %121		; <ubyte*>:218 [#uses=1]
	load ubyte* %218		; <ubyte>:345 [#uses=1]
	seteq ubyte %345, 0		; <bool>:142 [#uses=1]
	br bool %142, label %143, label %142

; <label>:143		; preds = %141, %142
	add uint %117, 4294967295		; <uint>:123 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %123		; <ubyte*>:219 [#uses=1]
	load ubyte* %219		; <ubyte>:346 [#uses=1]
	seteq ubyte %346, 0		; <bool>:143 [#uses=1]
	br bool %143, label %145, label %144

; <label>:144		; preds = %143, %144
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %121		; <ubyte*>:220 [#uses=2]
	load ubyte* %220		; <ubyte>:347 [#uses=1]
	add ubyte %347, 1		; <ubyte>:348 [#uses=1]
	store ubyte %348, ubyte* %220
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %123		; <ubyte*>:221 [#uses=2]
	load ubyte* %221		; <ubyte>:349 [#uses=2]
	add ubyte %349, 255		; <ubyte>:350 [#uses=1]
	store ubyte %350, ubyte* %221
	seteq ubyte %349, 1		; <bool>:144 [#uses=1]
	br bool %144, label %145, label %144

; <label>:145		; preds = %143, %144
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %117		; <ubyte*>:222 [#uses=2]
	load ubyte* %222		; <ubyte>:351 [#uses=2]
	add ubyte %351, 1		; <ubyte>:352 [#uses=1]
	store ubyte %352, ubyte* %222
	seteq ubyte %351, 255		; <bool>:145 [#uses=1]
	br bool %145, label %147, label %146

; <label>:146		; preds = %145, %151
	phi uint [ %117, %145 ], [ %129, %151 ]		; <uint>:124 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %124		; <ubyte*>:223 [#uses=2]
	load ubyte* %223		; <ubyte>:353 [#uses=1]
	add ubyte %353, 255		; <ubyte>:354 [#uses=1]
	store ubyte %354, ubyte* %223
	add uint %124, 4294967286		; <uint>:125 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %125		; <ubyte*>:224 [#uses=1]
	load ubyte* %224		; <ubyte>:355 [#uses=1]
	seteq ubyte %355, 0		; <bool>:146 [#uses=1]
	br bool %146, label %149, label %148

; <label>:147		; preds = %145, %151
	phi uint [ %117, %145 ], [ %129, %151 ]		; <uint>:126 [#uses=22]
	add uint %126, 4		; <uint>:127 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %127		; <ubyte*>:225 [#uses=1]
	load ubyte* %225		; <ubyte>:356 [#uses=1]
	seteq ubyte %356, 0		; <bool>:147 [#uses=1]
	br bool %147, label %153, label %152

; <label>:148		; preds = %146, %148
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %125		; <ubyte*>:226 [#uses=2]
	load ubyte* %226		; <ubyte>:357 [#uses=2]
	add ubyte %357, 255		; <ubyte>:358 [#uses=1]
	store ubyte %358, ubyte* %226
	seteq ubyte %357, 1		; <bool>:148 [#uses=1]
	br bool %148, label %149, label %148

; <label>:149		; preds = %146, %148
	add uint %124, 4294967292		; <uint>:128 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %128		; <ubyte*>:227 [#uses=1]
	load ubyte* %227		; <ubyte>:359 [#uses=1]
	seteq ubyte %359, 0		; <bool>:149 [#uses=1]
	br bool %149, label %151, label %150

; <label>:150		; preds = %149, %150
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %125		; <ubyte*>:228 [#uses=2]
	load ubyte* %228		; <ubyte>:360 [#uses=1]
	add ubyte %360, 1		; <ubyte>:361 [#uses=1]
	store ubyte %361, ubyte* %228
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %128		; <ubyte*>:229 [#uses=2]
	load ubyte* %229		; <ubyte>:362 [#uses=2]
	add ubyte %362, 255		; <ubyte>:363 [#uses=1]
	store ubyte %363, ubyte* %229
	seteq ubyte %362, 1		; <bool>:150 [#uses=1]
	br bool %150, label %151, label %150

; <label>:151		; preds = %149, %150
	add uint %124, 4294967290		; <uint>:129 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %129		; <ubyte*>:230 [#uses=1]
	load ubyte* %230		; <ubyte>:364 [#uses=1]
	seteq ubyte %364, 0		; <bool>:151 [#uses=1]
	br bool %151, label %147, label %146

; <label>:152		; preds = %147, %152
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %127		; <ubyte*>:231 [#uses=2]
	load ubyte* %231		; <ubyte>:365 [#uses=2]
	add ubyte %365, 255		; <ubyte>:366 [#uses=1]
	store ubyte %366, ubyte* %231
	seteq ubyte %365, 1		; <bool>:152 [#uses=1]
	br bool %152, label %153, label %152

; <label>:153		; preds = %147, %152
	add uint %126, 10		; <uint>:130 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %130		; <ubyte*>:232 [#uses=1]
	load ubyte* %232		; <ubyte>:367 [#uses=1]
	seteq ubyte %367, 0		; <bool>:153 [#uses=1]
	br bool %153, label %155, label %154

; <label>:154		; preds = %153, %154
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %130		; <ubyte*>:233 [#uses=2]
	load ubyte* %233		; <ubyte>:368 [#uses=2]
	add ubyte %368, 255		; <ubyte>:369 [#uses=1]
	store ubyte %369, ubyte* %233
	seteq ubyte %368, 1		; <bool>:154 [#uses=1]
	br bool %154, label %155, label %154

; <label>:155		; preds = %153, %154
	add uint %126, 16		; <uint>:131 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %131		; <ubyte*>:234 [#uses=1]
	load ubyte* %234		; <ubyte>:370 [#uses=1]
	seteq ubyte %370, 0		; <bool>:155 [#uses=1]
	br bool %155, label %157, label %156

; <label>:156		; preds = %155, %156
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %131		; <ubyte*>:235 [#uses=2]
	load ubyte* %235		; <ubyte>:371 [#uses=2]
	add ubyte %371, 255		; <ubyte>:372 [#uses=1]
	store ubyte %372, ubyte* %235
	seteq ubyte %371, 1		; <bool>:156 [#uses=1]
	br bool %156, label %157, label %156

; <label>:157		; preds = %155, %156
	add uint %126, 22		; <uint>:132 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %132		; <ubyte*>:236 [#uses=1]
	load ubyte* %236		; <ubyte>:373 [#uses=1]
	seteq ubyte %373, 0		; <bool>:157 [#uses=1]
	br bool %157, label %159, label %158

; <label>:158		; preds = %157, %158
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %132		; <ubyte*>:237 [#uses=2]
	load ubyte* %237		; <ubyte>:374 [#uses=2]
	add ubyte %374, 255		; <ubyte>:375 [#uses=1]
	store ubyte %375, ubyte* %237
	seteq ubyte %374, 1		; <bool>:158 [#uses=1]
	br bool %158, label %159, label %158

; <label>:159		; preds = %157, %158
	add uint %126, 28		; <uint>:133 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %133		; <ubyte*>:238 [#uses=1]
	load ubyte* %238		; <ubyte>:376 [#uses=1]
	seteq ubyte %376, 0		; <bool>:159 [#uses=1]
	br bool %159, label %161, label %160

; <label>:160		; preds = %159, %160
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %133		; <ubyte*>:239 [#uses=2]
	load ubyte* %239		; <ubyte>:377 [#uses=2]
	add ubyte %377, 255		; <ubyte>:378 [#uses=1]
	store ubyte %378, ubyte* %239
	seteq ubyte %377, 1		; <bool>:160 [#uses=1]
	br bool %160, label %161, label %160

; <label>:161		; preds = %159, %160
	add uint %126, 34		; <uint>:134 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %134		; <ubyte*>:240 [#uses=1]
	load ubyte* %240		; <ubyte>:379 [#uses=1]
	seteq ubyte %379, 0		; <bool>:161 [#uses=1]
	br bool %161, label %163, label %162

; <label>:162		; preds = %161, %162
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %134		; <ubyte*>:241 [#uses=2]
	load ubyte* %241		; <ubyte>:380 [#uses=2]
	add ubyte %380, 255		; <ubyte>:381 [#uses=1]
	store ubyte %381, ubyte* %241
	seteq ubyte %380, 1		; <bool>:162 [#uses=1]
	br bool %162, label %163, label %162

; <label>:163		; preds = %161, %162
	add uint %126, 40		; <uint>:135 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %135		; <ubyte*>:242 [#uses=1]
	load ubyte* %242		; <ubyte>:382 [#uses=1]
	seteq ubyte %382, 0		; <bool>:163 [#uses=1]
	br bool %163, label %165, label %164

; <label>:164		; preds = %163, %164
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %135		; <ubyte*>:243 [#uses=2]
	load ubyte* %243		; <ubyte>:383 [#uses=2]
	add ubyte %383, 255		; <ubyte>:384 [#uses=1]
	store ubyte %384, ubyte* %243
	seteq ubyte %383, 1		; <bool>:164 [#uses=1]
	br bool %164, label %165, label %164

; <label>:165		; preds = %163, %164
	add uint %126, 46		; <uint>:136 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %136		; <ubyte*>:244 [#uses=1]
	load ubyte* %244		; <ubyte>:385 [#uses=1]
	seteq ubyte %385, 0		; <bool>:165 [#uses=1]
	br bool %165, label %167, label %166

; <label>:166		; preds = %165, %166
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %136		; <ubyte*>:245 [#uses=2]
	load ubyte* %245		; <ubyte>:386 [#uses=2]
	add ubyte %386, 255		; <ubyte>:387 [#uses=1]
	store ubyte %387, ubyte* %245
	seteq ubyte %386, 1		; <bool>:166 [#uses=1]
	br bool %166, label %167, label %166

; <label>:167		; preds = %165, %166
	add uint %126, 52		; <uint>:137 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %137		; <ubyte*>:246 [#uses=1]
	load ubyte* %246		; <ubyte>:388 [#uses=1]
	seteq ubyte %388, 0		; <bool>:167 [#uses=1]
	br bool %167, label %169, label %168

; <label>:168		; preds = %167, %168
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %137		; <ubyte*>:247 [#uses=2]
	load ubyte* %247		; <ubyte>:389 [#uses=2]
	add ubyte %389, 255		; <ubyte>:390 [#uses=1]
	store ubyte %390, ubyte* %247
	seteq ubyte %389, 1		; <bool>:168 [#uses=1]
	br bool %168, label %169, label %168

; <label>:169		; preds = %167, %168
	add uint %126, 58		; <uint>:138 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %138		; <ubyte*>:248 [#uses=1]
	load ubyte* %248		; <ubyte>:391 [#uses=1]
	seteq ubyte %391, 0		; <bool>:169 [#uses=1]
	br bool %169, label %171, label %170

; <label>:170		; preds = %169, %170
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %138		; <ubyte*>:249 [#uses=2]
	load ubyte* %249		; <ubyte>:392 [#uses=2]
	add ubyte %392, 255		; <ubyte>:393 [#uses=1]
	store ubyte %393, ubyte* %249
	seteq ubyte %392, 1		; <bool>:170 [#uses=1]
	br bool %170, label %171, label %170

; <label>:171		; preds = %169, %170
	add uint %126, 64		; <uint>:139 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %139		; <ubyte*>:250 [#uses=1]
	load ubyte* %250		; <ubyte>:394 [#uses=1]
	seteq ubyte %394, 0		; <bool>:171 [#uses=1]
	br bool %171, label %173, label %172

; <label>:172		; preds = %171, %172
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %139		; <ubyte*>:251 [#uses=2]
	load ubyte* %251		; <ubyte>:395 [#uses=2]
	add ubyte %395, 255		; <ubyte>:396 [#uses=1]
	store ubyte %396, ubyte* %251
	seteq ubyte %395, 1		; <bool>:172 [#uses=1]
	br bool %172, label %173, label %172

; <label>:173		; preds = %171, %172
	add uint %126, 70		; <uint>:140 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %140		; <ubyte*>:252 [#uses=1]
	load ubyte* %252		; <ubyte>:397 [#uses=1]
	seteq ubyte %397, 0		; <bool>:173 [#uses=1]
	br bool %173, label %175, label %174

; <label>:174		; preds = %173, %174
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %140		; <ubyte*>:253 [#uses=2]
	load ubyte* %253		; <ubyte>:398 [#uses=2]
	add ubyte %398, 255		; <ubyte>:399 [#uses=1]
	store ubyte %399, ubyte* %253
	seteq ubyte %398, 1		; <bool>:174 [#uses=1]
	br bool %174, label %175, label %174

; <label>:175		; preds = %173, %174
	add uint %126, 76		; <uint>:141 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %141		; <ubyte*>:254 [#uses=1]
	load ubyte* %254		; <ubyte>:400 [#uses=1]
	seteq ubyte %400, 0		; <bool>:175 [#uses=1]
	br bool %175, label %177, label %176

; <label>:176		; preds = %175, %176
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %141		; <ubyte*>:255 [#uses=2]
	load ubyte* %255		; <ubyte>:401 [#uses=2]
	add ubyte %401, 255		; <ubyte>:402 [#uses=1]
	store ubyte %402, ubyte* %255
	seteq ubyte %401, 1		; <bool>:176 [#uses=1]
	br bool %176, label %177, label %176

; <label>:177		; preds = %175, %176
	add uint %126, 82		; <uint>:142 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %142		; <ubyte*>:256 [#uses=1]
	load ubyte* %256		; <ubyte>:403 [#uses=1]
	seteq ubyte %403, 0		; <bool>:177 [#uses=1]
	br bool %177, label %179, label %178

; <label>:178		; preds = %177, %178
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %142		; <ubyte*>:257 [#uses=2]
	load ubyte* %257		; <ubyte>:404 [#uses=2]
	add ubyte %404, 255		; <ubyte>:405 [#uses=1]
	store ubyte %405, ubyte* %257
	seteq ubyte %404, 1		; <bool>:178 [#uses=1]
	br bool %178, label %179, label %178

; <label>:179		; preds = %177, %178
	add uint %126, 88		; <uint>:143 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %143		; <ubyte*>:258 [#uses=1]
	load ubyte* %258		; <ubyte>:406 [#uses=1]
	seteq ubyte %406, 0		; <bool>:179 [#uses=1]
	br bool %179, label %181, label %180

; <label>:180		; preds = %179, %180
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %143		; <ubyte*>:259 [#uses=2]
	load ubyte* %259		; <ubyte>:407 [#uses=2]
	add ubyte %407, 255		; <ubyte>:408 [#uses=1]
	store ubyte %408, ubyte* %259
	seteq ubyte %407, 1		; <bool>:180 [#uses=1]
	br bool %180, label %181, label %180

; <label>:181		; preds = %179, %180
	add uint %126, 4294967292		; <uint>:144 [#uses=8]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %144		; <ubyte*>:260 [#uses=2]
	load ubyte* %260		; <ubyte>:409 [#uses=1]
	call void %outputcell( ubyte %409 )
	load ubyte* %260		; <ubyte>:410 [#uses=1]
	seteq ubyte %410, 0		; <bool>:181 [#uses=1]
	br bool %181, label %183, label %182

; <label>:182		; preds = %181, %182
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %144		; <ubyte*>:261 [#uses=2]
	load ubyte* %261		; <ubyte>:411 [#uses=2]
	add ubyte %411, 255		; <ubyte>:412 [#uses=1]
	store ubyte %412, ubyte* %261
	seteq ubyte %411, 1		; <bool>:182 [#uses=1]
	br bool %182, label %183, label %182

; <label>:183		; preds = %181, %182
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %144		; <ubyte*>:262 [#uses=1]
	load ubyte* %262		; <ubyte>:413 [#uses=1]
	seteq ubyte %413, 0		; <bool>:183 [#uses=1]
	br bool %183, label %185, label %184

; <label>:184		; preds = %183, %184
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %144		; <ubyte*>:263 [#uses=2]
	load ubyte* %263		; <ubyte>:414 [#uses=2]
	add ubyte %414, 255		; <ubyte>:415 [#uses=1]
	store ubyte %415, ubyte* %263
	seteq ubyte %414, 1		; <bool>:184 [#uses=1]
	br bool %184, label %185, label %184

; <label>:185		; preds = %183, %184
	add uint %126, 4294967288		; <uint>:145 [#uses=7]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %145		; <ubyte*>:264 [#uses=1]
	load ubyte* %264		; <ubyte>:416 [#uses=1]
	seteq ubyte %416, 0		; <bool>:185 [#uses=1]
	br bool %185, label %187, label %186

; <label>:186		; preds = %185, %186
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %145		; <ubyte*>:265 [#uses=2]
	load ubyte* %265		; <ubyte>:417 [#uses=1]
	add ubyte %417, 255		; <ubyte>:418 [#uses=1]
	store ubyte %418, ubyte* %265
	add uint %126, 4294967289		; <uint>:146 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %146		; <ubyte*>:266 [#uses=2]
	load ubyte* %266		; <ubyte>:419 [#uses=1]
	add ubyte %419, 1		; <ubyte>:420 [#uses=1]
	store ubyte %420, ubyte* %266
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %144		; <ubyte*>:267 [#uses=2]
	load ubyte* %267		; <ubyte>:421 [#uses=1]
	add ubyte %421, 1		; <ubyte>:422 [#uses=1]
	store ubyte %422, ubyte* %267
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %145		; <ubyte*>:268 [#uses=1]
	load ubyte* %268		; <ubyte>:423 [#uses=1]
	seteq ubyte %423, 0		; <bool>:186 [#uses=1]
	br bool %186, label %187, label %186

; <label>:187		; preds = %185, %186
	add uint %126, 4294967289		; <uint>:147 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %147		; <ubyte*>:269 [#uses=1]
	load ubyte* %269		; <ubyte>:424 [#uses=1]
	seteq ubyte %424, 0		; <bool>:187 [#uses=1]
	br bool %187, label %189, label %188

; <label>:188		; preds = %187, %188
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %145		; <ubyte*>:270 [#uses=2]
	load ubyte* %270		; <ubyte>:425 [#uses=1]
	add ubyte %425, 1		; <ubyte>:426 [#uses=1]
	store ubyte %426, ubyte* %270
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %147		; <ubyte*>:271 [#uses=2]
	load ubyte* %271		; <ubyte>:427 [#uses=2]
	add ubyte %427, 255		; <ubyte>:428 [#uses=1]
	store ubyte %428, ubyte* %271
	seteq ubyte %427, 1		; <bool>:188 [#uses=1]
	br bool %188, label %189, label %188

; <label>:189		; preds = %187, %188
	add uint %126, 4294967294		; <uint>:148 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %148		; <ubyte*>:272 [#uses=2]
	load ubyte* %272		; <ubyte>:429 [#uses=2]
	add ubyte %429, 1		; <ubyte>:430 [#uses=1]
	store ubyte %430, ubyte* %272
	seteq ubyte %429, 255		; <bool>:189 [#uses=1]
	br bool %189, label %191, label %190

; <label>:190		; preds = %189, %190
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %144		; <ubyte*>:273 [#uses=2]
	load ubyte* %273		; <ubyte>:431 [#uses=1]
	add ubyte %431, 1		; <ubyte>:432 [#uses=1]
	store ubyte %432, ubyte* %273
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %148		; <ubyte*>:274 [#uses=2]
	load ubyte* %274		; <ubyte>:433 [#uses=2]
	add ubyte %433, 255		; <ubyte>:434 [#uses=1]
	store ubyte %434, ubyte* %274
	seteq ubyte %433, 1		; <bool>:190 [#uses=1]
	br bool %190, label %191, label %190

; <label>:191		; preds = %189, %190
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %145		; <ubyte*>:275 [#uses=1]
	load ubyte* %275		; <ubyte>:435 [#uses=1]
	seteq ubyte %435, 0		; <bool>:191 [#uses=1]
	br bool %191, label %193, label %192

; <label>:192		; preds = %191, %192
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %145		; <ubyte*>:276 [#uses=2]
	load ubyte* %276		; <ubyte>:436 [#uses=2]
	add ubyte %436, 255		; <ubyte>:437 [#uses=1]
	store ubyte %437, ubyte* %276
	seteq ubyte %436, 1		; <bool>:192 [#uses=1]
	br bool %192, label %193, label %192

; <label>:193		; preds = %191, %192
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %144		; <ubyte*>:277 [#uses=1]
	load ubyte* %277		; <ubyte>:438 [#uses=1]
	seteq ubyte %438, 0		; <bool>:193 [#uses=1]
	br bool %193, label %195, label %194

; <label>:194		; preds = %193, %194
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %145		; <ubyte*>:278 [#uses=2]
	load ubyte* %278		; <ubyte>:439 [#uses=1]
	add ubyte %439, 1		; <ubyte>:440 [#uses=1]
	store ubyte %440, ubyte* %278
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %144		; <ubyte*>:279 [#uses=2]
	load ubyte* %279		; <ubyte>:441 [#uses=2]
	add ubyte %441, 255		; <ubyte>:442 [#uses=1]
	store ubyte %442, ubyte* %279
	seteq ubyte %441, 1		; <bool>:194 [#uses=1]
	br bool %194, label %195, label %194

; <label>:195		; preds = %193, %194
	add uint %126, 4294967187		; <uint>:149 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %149		; <ubyte*>:280 [#uses=2]
	load ubyte* %280		; <ubyte>:443 [#uses=1]
	add ubyte %443, 7		; <ubyte>:444 [#uses=1]
	store ubyte %444, ubyte* %280
	add uint %126, 4294967189		; <uint>:150 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %150		; <ubyte*>:281 [#uses=1]
	load ubyte* %281		; <ubyte>:445 [#uses=1]
	seteq ubyte %445, 0		; <bool>:195 [#uses=1]
	br bool %195, label %35, label %34

; <label>:196		; preds = %29, %249
	phi uint [ %43, %29 ], [ %175, %249 ]		; <uint>:151 [#uses=23]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %151		; <ubyte*>:282 [#uses=2]
	load ubyte* %282		; <ubyte>:446 [#uses=1]
	add ubyte %446, 255		; <ubyte>:447 [#uses=1]
	store ubyte %447, ubyte* %282
	add uint %151, 104		; <uint>:152 [#uses=17]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %152		; <ubyte*>:283 [#uses=1]
	load ubyte* %283		; <ubyte>:448 [#uses=1]
	seteq ubyte %448, 0		; <bool>:196 [#uses=1]
	br bool %196, label %199, label %198

; <label>:197		; preds = %29, %249
	phi uint [ %43, %29 ], [ %175, %249 ]		; <uint>:153 [#uses=1]
	add uint %153, 4294967295		; <uint>:154 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %154		; <ubyte*>:284 [#uses=1]
	load ubyte* %284		; <ubyte>:449 [#uses=1]
	seteq ubyte %449, 0		; <bool>:197 [#uses=1]
	br bool %197, label %27, label %26

; <label>:198		; preds = %196, %198
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %152		; <ubyte*>:285 [#uses=2]
	load ubyte* %285		; <ubyte>:450 [#uses=2]
	add ubyte %450, 255		; <ubyte>:451 [#uses=1]
	store ubyte %451, ubyte* %285
	seteq ubyte %450, 1		; <bool>:198 [#uses=1]
	br bool %198, label %199, label %198

; <label>:199		; preds = %196, %198
	add uint %151, 100		; <uint>:155 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %155		; <ubyte*>:286 [#uses=1]
	load ubyte* %286		; <ubyte>:452 [#uses=1]
	seteq ubyte %452, 0		; <bool>:199 [#uses=1]
	br bool %199, label %201, label %200

; <label>:200		; preds = %199, %200
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %155		; <ubyte*>:287 [#uses=2]
	load ubyte* %287		; <ubyte>:453 [#uses=1]
	add ubyte %453, 255		; <ubyte>:454 [#uses=1]
	store ubyte %454, ubyte* %287
	add uint %151, 101		; <uint>:156 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %156		; <ubyte*>:288 [#uses=2]
	load ubyte* %288		; <ubyte>:455 [#uses=1]
	add ubyte %455, 1		; <ubyte>:456 [#uses=1]
	store ubyte %456, ubyte* %288
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %152		; <ubyte*>:289 [#uses=2]
	load ubyte* %289		; <ubyte>:457 [#uses=1]
	add ubyte %457, 1		; <ubyte>:458 [#uses=1]
	store ubyte %458, ubyte* %289
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %155		; <ubyte*>:290 [#uses=1]
	load ubyte* %290		; <ubyte>:459 [#uses=1]
	seteq ubyte %459, 0		; <bool>:200 [#uses=1]
	br bool %200, label %201, label %200

; <label>:201		; preds = %199, %200
	add uint %151, 101		; <uint>:157 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %157		; <ubyte*>:291 [#uses=1]
	load ubyte* %291		; <ubyte>:460 [#uses=1]
	seteq ubyte %460, 0		; <bool>:201 [#uses=1]
	br bool %201, label %203, label %202

; <label>:202		; preds = %201, %202
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %155		; <ubyte*>:292 [#uses=2]
	load ubyte* %292		; <ubyte>:461 [#uses=1]
	add ubyte %461, 1		; <ubyte>:462 [#uses=1]
	store ubyte %462, ubyte* %292
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %157		; <ubyte*>:293 [#uses=2]
	load ubyte* %293		; <ubyte>:463 [#uses=2]
	add ubyte %463, 255		; <ubyte>:464 [#uses=1]
	store ubyte %464, ubyte* %293
	seteq ubyte %463, 1		; <bool>:202 [#uses=1]
	br bool %202, label %203, label %202

; <label>:203		; preds = %201, %202
	add uint %151, 106		; <uint>:158 [#uses=12]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %158		; <ubyte*>:294 [#uses=1]
	load ubyte* %294		; <ubyte>:465 [#uses=1]
	seteq ubyte %465, 0		; <bool>:203 [#uses=1]
	br bool %203, label %205, label %204

; <label>:204		; preds = %203, %204
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %158		; <ubyte*>:295 [#uses=2]
	load ubyte* %295		; <ubyte>:466 [#uses=2]
	add ubyte %466, 255		; <ubyte>:467 [#uses=1]
	store ubyte %467, ubyte* %295
	seteq ubyte %466, 1		; <bool>:204 [#uses=1]
	br bool %204, label %205, label %204

; <label>:205		; preds = %203, %204
	add uint %151, 98		; <uint>:159 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %159		; <ubyte*>:296 [#uses=1]
	load ubyte* %296		; <ubyte>:468 [#uses=1]
	seteq ubyte %468, 0		; <bool>:205 [#uses=1]
	br bool %205, label %207, label %206

; <label>:206		; preds = %205, %206
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %159		; <ubyte*>:297 [#uses=2]
	load ubyte* %297		; <ubyte>:469 [#uses=1]
	add ubyte %469, 255		; <ubyte>:470 [#uses=1]
	store ubyte %470, ubyte* %297
	add uint %151, 99		; <uint>:160 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %160		; <ubyte*>:298 [#uses=2]
	load ubyte* %298		; <ubyte>:471 [#uses=1]
	add ubyte %471, 1		; <ubyte>:472 [#uses=1]
	store ubyte %472, ubyte* %298
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %158		; <ubyte*>:299 [#uses=2]
	load ubyte* %299		; <ubyte>:473 [#uses=1]
	add ubyte %473, 1		; <ubyte>:474 [#uses=1]
	store ubyte %474, ubyte* %299
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %159		; <ubyte*>:300 [#uses=1]
	load ubyte* %300		; <ubyte>:475 [#uses=1]
	seteq ubyte %475, 0		; <bool>:206 [#uses=1]
	br bool %206, label %207, label %206

; <label>:207		; preds = %205, %206
	add uint %151, 99		; <uint>:161 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %161		; <ubyte*>:301 [#uses=1]
	load ubyte* %301		; <ubyte>:476 [#uses=1]
	seteq ubyte %476, 0		; <bool>:207 [#uses=1]
	br bool %207, label %209, label %208

; <label>:208		; preds = %207, %208
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %159		; <ubyte*>:302 [#uses=2]
	load ubyte* %302		; <ubyte>:477 [#uses=1]
	add ubyte %477, 1		; <ubyte>:478 [#uses=1]
	store ubyte %478, ubyte* %302
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %161		; <ubyte*>:303 [#uses=2]
	load ubyte* %303		; <ubyte>:479 [#uses=2]
	add ubyte %479, 255		; <ubyte>:480 [#uses=1]
	store ubyte %480, ubyte* %303
	seteq ubyte %479, 1		; <bool>:208 [#uses=1]
	br bool %208, label %209, label %208

; <label>:209		; preds = %207, %208
	add uint %151, 108		; <uint>:162 [#uses=9]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %162		; <ubyte*>:304 [#uses=2]
	load ubyte* %304		; <ubyte>:481 [#uses=2]
	add ubyte %481, 1		; <ubyte>:482 [#uses=1]
	store ubyte %482, ubyte* %304
	seteq ubyte %481, 255		; <bool>:209 [#uses=1]
	br bool %209, label %211, label %210

; <label>:210		; preds = %209, %233
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %162		; <ubyte*>:305 [#uses=2]
	load ubyte* %305		; <ubyte>:483 [#uses=1]
	add ubyte %483, 1		; <ubyte>:484 [#uses=1]
	store ubyte %484, ubyte* %305
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %152		; <ubyte*>:306 [#uses=1]
	load ubyte* %306		; <ubyte>:485 [#uses=1]
	seteq ubyte %485, 0		; <bool>:210 [#uses=1]
	br bool %210, label %213, label %212

; <label>:211		; preds = %209, %233
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %152		; <ubyte*>:307 [#uses=1]
	load ubyte* %307		; <ubyte>:486 [#uses=1]
	seteq ubyte %486, 0		; <bool>:211 [#uses=1]
	br bool %211, label %235, label %234

; <label>:212		; preds = %210, %212
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %152		; <ubyte*>:308 [#uses=2]
	load ubyte* %308		; <ubyte>:487 [#uses=1]
	add ubyte %487, 255		; <ubyte>:488 [#uses=1]
	store ubyte %488, ubyte* %308
	add uint %151, 105		; <uint>:163 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %163		; <ubyte*>:309 [#uses=2]
	load ubyte* %309		; <ubyte>:489 [#uses=1]
	add ubyte %489, 1		; <ubyte>:490 [#uses=1]
	store ubyte %490, ubyte* %309
	add uint %151, 109		; <uint>:164 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %164		; <ubyte*>:310 [#uses=2]
	load ubyte* %310		; <ubyte>:491 [#uses=1]
	add ubyte %491, 1		; <ubyte>:492 [#uses=1]
	store ubyte %492, ubyte* %310
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %152		; <ubyte*>:311 [#uses=1]
	load ubyte* %311		; <ubyte>:493 [#uses=1]
	seteq ubyte %493, 0		; <bool>:212 [#uses=1]
	br bool %212, label %213, label %212

; <label>:213		; preds = %210, %212
	add uint %151, 105		; <uint>:165 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %165		; <ubyte*>:312 [#uses=1]
	load ubyte* %312		; <ubyte>:494 [#uses=1]
	seteq ubyte %494, 0		; <bool>:213 [#uses=1]
	br bool %213, label %215, label %214

; <label>:214		; preds = %213, %214
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %152		; <ubyte*>:313 [#uses=2]
	load ubyte* %313		; <ubyte>:495 [#uses=1]
	add ubyte %495, 1		; <ubyte>:496 [#uses=1]
	store ubyte %496, ubyte* %313
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %165		; <ubyte*>:314 [#uses=2]
	load ubyte* %314		; <ubyte>:497 [#uses=2]
	add ubyte %497, 255		; <ubyte>:498 [#uses=1]
	store ubyte %498, ubyte* %314
	seteq ubyte %497, 1		; <bool>:214 [#uses=1]
	br bool %214, label %215, label %214

; <label>:215		; preds = %213, %214
	add uint %151, 109		; <uint>:166 [#uses=13]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %166		; <ubyte*>:315 [#uses=1]
	load ubyte* %315		; <ubyte>:499 [#uses=1]
	seteq ubyte %499, 0		; <bool>:215 [#uses=1]
	br bool %215, label %217, label %216

; <label>:216		; preds = %215, %219
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %166		; <ubyte*>:316 [#uses=1]
	load ubyte* %316		; <ubyte>:500 [#uses=1]
	seteq ubyte %500, 0		; <bool>:216 [#uses=1]
	br bool %216, label %219, label %218

; <label>:217		; preds = %215, %219
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %158		; <ubyte*>:317 [#uses=1]
	load ubyte* %317		; <ubyte>:501 [#uses=1]
	seteq ubyte %501, 0		; <bool>:217 [#uses=1]
	br bool %217, label %221, label %220

; <label>:218		; preds = %216, %218
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %166		; <ubyte*>:318 [#uses=2]
	load ubyte* %318		; <ubyte>:502 [#uses=2]
	add ubyte %502, 255		; <ubyte>:503 [#uses=1]
	store ubyte %503, ubyte* %318
	seteq ubyte %502, 1		; <bool>:218 [#uses=1]
	br bool %218, label %219, label %218

; <label>:219		; preds = %216, %218
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %162		; <ubyte*>:319 [#uses=2]
	load ubyte* %319		; <ubyte>:504 [#uses=1]
	add ubyte %504, 255		; <ubyte>:505 [#uses=1]
	store ubyte %505, ubyte* %319
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %166		; <ubyte*>:320 [#uses=1]
	load ubyte* %320		; <ubyte>:506 [#uses=1]
	seteq ubyte %506, 0		; <bool>:219 [#uses=1]
	br bool %219, label %217, label %216

; <label>:220		; preds = %217, %220
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %158		; <ubyte*>:321 [#uses=2]
	load ubyte* %321		; <ubyte>:507 [#uses=1]
	add ubyte %507, 255		; <ubyte>:508 [#uses=1]
	store ubyte %508, ubyte* %321
	add uint %151, 107		; <uint>:167 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %167		; <ubyte*>:322 [#uses=2]
	load ubyte* %322		; <ubyte>:509 [#uses=1]
	add ubyte %509, 1		; <ubyte>:510 [#uses=1]
	store ubyte %510, ubyte* %322
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %166		; <ubyte*>:323 [#uses=2]
	load ubyte* %323		; <ubyte>:511 [#uses=1]
	add ubyte %511, 1		; <ubyte>:512 [#uses=1]
	store ubyte %512, ubyte* %323
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %158		; <ubyte*>:324 [#uses=1]
	load ubyte* %324		; <ubyte>:513 [#uses=1]
	seteq ubyte %513, 0		; <bool>:220 [#uses=1]
	br bool %220, label %221, label %220

; <label>:221		; preds = %217, %220
	add uint %151, 107		; <uint>:168 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %168		; <ubyte*>:325 [#uses=1]
	load ubyte* %325		; <ubyte>:514 [#uses=1]
	seteq ubyte %514, 0		; <bool>:221 [#uses=1]
	br bool %221, label %223, label %222

; <label>:222		; preds = %221, %222
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %158		; <ubyte*>:326 [#uses=2]
	load ubyte* %326		; <ubyte>:515 [#uses=1]
	add ubyte %515, 1		; <ubyte>:516 [#uses=1]
	store ubyte %516, ubyte* %326
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %168		; <ubyte*>:327 [#uses=2]
	load ubyte* %327		; <ubyte>:517 [#uses=2]
	add ubyte %517, 255		; <ubyte>:518 [#uses=1]
	store ubyte %518, ubyte* %327
	seteq ubyte %517, 1		; <bool>:222 [#uses=1]
	br bool %222, label %223, label %222

; <label>:223		; preds = %221, %222
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %166		; <ubyte*>:328 [#uses=1]
	load ubyte* %328		; <ubyte>:519 [#uses=1]
	seteq ubyte %519, 0		; <bool>:223 [#uses=1]
	br bool %223, label %225, label %224

; <label>:224		; preds = %223, %227
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %166		; <ubyte*>:329 [#uses=1]
	load ubyte* %329		; <ubyte>:520 [#uses=1]
	seteq ubyte %520, 0		; <bool>:224 [#uses=1]
	br bool %224, label %227, label %226

; <label>:225		; preds = %223, %227
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %166		; <ubyte*>:330 [#uses=2]
	load ubyte* %330		; <ubyte>:521 [#uses=1]
	add ubyte %521, 1		; <ubyte>:522 [#uses=1]
	store ubyte %522, ubyte* %330
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %162		; <ubyte*>:331 [#uses=1]
	load ubyte* %331		; <ubyte>:523 [#uses=1]
	seteq ubyte %523, 0		; <bool>:225 [#uses=1]
	br bool %225, label %229, label %228

; <label>:226		; preds = %224, %226
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %166		; <ubyte*>:332 [#uses=2]
	load ubyte* %332		; <ubyte>:524 [#uses=2]
	add ubyte %524, 255		; <ubyte>:525 [#uses=1]
	store ubyte %525, ubyte* %332
	seteq ubyte %524, 1		; <bool>:226 [#uses=1]
	br bool %226, label %227, label %226

; <label>:227		; preds = %224, %226
	add uint %151, 108		; <uint>:169 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %169		; <ubyte*>:333 [#uses=2]
	load ubyte* %333		; <ubyte>:526 [#uses=1]
	add ubyte %526, 255		; <ubyte>:527 [#uses=1]
	store ubyte %527, ubyte* %333
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %166		; <ubyte*>:334 [#uses=1]
	load ubyte* %334		; <ubyte>:528 [#uses=1]
	seteq ubyte %528, 0		; <bool>:227 [#uses=1]
	br bool %227, label %225, label %224

; <label>:228		; preds = %225, %231
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %162		; <ubyte*>:335 [#uses=1]
	load ubyte* %335		; <ubyte>:529 [#uses=1]
	seteq ubyte %529, 0		; <bool>:228 [#uses=1]
	br bool %228, label %231, label %230

; <label>:229		; preds = %225, %231
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %166		; <ubyte*>:336 [#uses=1]
	load ubyte* %336		; <ubyte>:530 [#uses=1]
	seteq ubyte %530, 0		; <bool>:229 [#uses=1]
	br bool %229, label %233, label %232

; <label>:230		; preds = %228, %230
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %162		; <ubyte*>:337 [#uses=2]
	load ubyte* %337		; <ubyte>:531 [#uses=2]
	add ubyte %531, 255		; <ubyte>:532 [#uses=1]
	store ubyte %532, ubyte* %337
	seteq ubyte %531, 1		; <bool>:230 [#uses=1]
	br bool %230, label %231, label %230

; <label>:231		; preds = %228, %230
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %166		; <ubyte*>:338 [#uses=2]
	load ubyte* %338		; <ubyte>:533 [#uses=1]
	add ubyte %533, 255		; <ubyte>:534 [#uses=1]
	store ubyte %534, ubyte* %338
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %162		; <ubyte*>:339 [#uses=1]
	load ubyte* %339		; <ubyte>:535 [#uses=1]
	seteq ubyte %535, 0		; <bool>:231 [#uses=1]
	br bool %231, label %229, label %228

; <label>:232		; preds = %229, %232
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %152		; <ubyte*>:340 [#uses=2]
	load ubyte* %340		; <ubyte>:536 [#uses=1]
	add ubyte %536, 255		; <ubyte>:537 [#uses=1]
	store ubyte %537, ubyte* %340
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %158		; <ubyte*>:341 [#uses=2]
	load ubyte* %341		; <ubyte>:538 [#uses=1]
	add ubyte %538, 255		; <ubyte>:539 [#uses=1]
	store ubyte %539, ubyte* %341
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %162		; <ubyte*>:342 [#uses=2]
	load ubyte* %342		; <ubyte>:540 [#uses=1]
	add ubyte %540, 1		; <ubyte>:541 [#uses=1]
	store ubyte %541, ubyte* %342
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %166		; <ubyte*>:343 [#uses=2]
	load ubyte* %343		; <ubyte>:542 [#uses=2]
	add ubyte %542, 255		; <ubyte>:543 [#uses=1]
	store ubyte %543, ubyte* %343
	seteq ubyte %542, 1		; <bool>:232 [#uses=1]
	br bool %232, label %233, label %232

; <label>:233		; preds = %229, %232
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %162		; <ubyte*>:344 [#uses=1]
	load ubyte* %344		; <ubyte>:544 [#uses=1]
	seteq ubyte %544, 0		; <bool>:233 [#uses=1]
	br bool %233, label %211, label %210

; <label>:234		; preds = %211, %237
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %152		; <ubyte*>:345 [#uses=1]
	load ubyte* %345		; <ubyte>:545 [#uses=1]
	seteq ubyte %545, 0		; <bool>:234 [#uses=1]
	br bool %234, label %237, label %236

; <label>:235		; preds = %211, %237
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %158		; <ubyte*>:346 [#uses=1]
	load ubyte* %346		; <ubyte>:546 [#uses=1]
	seteq ubyte %546, 0		; <bool>:235 [#uses=1]
	br bool %235, label %239, label %238

; <label>:236		; preds = %234, %236
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %152		; <ubyte*>:347 [#uses=2]
	load ubyte* %347		; <ubyte>:547 [#uses=2]
	add ubyte %547, 255		; <ubyte>:548 [#uses=1]
	store ubyte %548, ubyte* %347
	seteq ubyte %547, 1		; <bool>:236 [#uses=1]
	br bool %236, label %237, label %236

; <label>:237		; preds = %234, %236
	add uint %151, 105		; <uint>:170 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %170		; <ubyte*>:348 [#uses=2]
	load ubyte* %348		; <ubyte>:549 [#uses=1]
	add ubyte %549, 1		; <ubyte>:550 [#uses=1]
	store ubyte %550, ubyte* %348
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %152		; <ubyte*>:349 [#uses=1]
	load ubyte* %349		; <ubyte>:551 [#uses=1]
	seteq ubyte %551, 0		; <bool>:237 [#uses=1]
	br bool %237, label %235, label %234

; <label>:238		; preds = %235, %241
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %158		; <ubyte*>:350 [#uses=1]
	load ubyte* %350		; <ubyte>:552 [#uses=1]
	seteq ubyte %552, 0		; <bool>:238 [#uses=1]
	br bool %238, label %241, label %240

; <label>:239		; preds = %235, %241
	add uint %151, 105		; <uint>:171 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %171		; <ubyte*>:351 [#uses=1]
	load ubyte* %351		; <ubyte>:553 [#uses=1]
	seteq ubyte %553, 0		; <bool>:239 [#uses=1]
	br bool %239, label %243, label %242

; <label>:240		; preds = %238, %240
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %158		; <ubyte*>:352 [#uses=2]
	load ubyte* %352		; <ubyte>:554 [#uses=2]
	add ubyte %554, 255		; <ubyte>:555 [#uses=1]
	store ubyte %555, ubyte* %352
	seteq ubyte %554, 1		; <bool>:240 [#uses=1]
	br bool %240, label %241, label %240

; <label>:241		; preds = %238, %240
	add uint %151, 107		; <uint>:172 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %172		; <ubyte*>:353 [#uses=2]
	load ubyte* %353		; <ubyte>:556 [#uses=1]
	add ubyte %556, 1		; <ubyte>:557 [#uses=1]
	store ubyte %557, ubyte* %353
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %158		; <ubyte*>:354 [#uses=1]
	load ubyte* %354		; <ubyte>:558 [#uses=1]
	seteq ubyte %558, 0		; <bool>:241 [#uses=1]
	br bool %241, label %239, label %238

; <label>:242		; preds = %239, %242
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %171		; <ubyte*>:355 [#uses=2]
	load ubyte* %355		; <ubyte>:559 [#uses=2]
	add ubyte %559, 255		; <ubyte>:560 [#uses=1]
	store ubyte %560, ubyte* %355
	seteq ubyte %559, 1		; <bool>:242 [#uses=1]
	br bool %242, label %243, label %242

; <label>:243		; preds = %239, %242
	add uint %151, 107		; <uint>:173 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %173		; <ubyte*>:356 [#uses=1]
	load ubyte* %356		; <ubyte>:561 [#uses=1]
	seteq ubyte %561, 0		; <bool>:243 [#uses=1]
	br bool %243, label %245, label %244

; <label>:244		; preds = %243, %247
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %173		; <ubyte*>:357 [#uses=1]
	load ubyte* %357		; <ubyte>:562 [#uses=1]
	seteq ubyte %562, 0		; <bool>:244 [#uses=1]
	br bool %244, label %247, label %246

; <label>:245		; preds = %243, %247
	add uint %151, 4294967295		; <uint>:174 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %174		; <ubyte*>:358 [#uses=2]
	load ubyte* %358		; <ubyte>:563 [#uses=1]
	add ubyte %563, 6		; <ubyte>:564 [#uses=1]
	store ubyte %564, ubyte* %358
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %152		; <ubyte*>:359 [#uses=1]
	load ubyte* %359		; <ubyte>:565 [#uses=1]
	seteq ubyte %565, 0		; <bool>:245 [#uses=1]
	br bool %245, label %249, label %248

; <label>:246		; preds = %244, %246
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %173		; <ubyte*>:360 [#uses=2]
	load ubyte* %360		; <ubyte>:566 [#uses=2]
	add ubyte %566, 255		; <ubyte>:567 [#uses=1]
	store ubyte %567, ubyte* %360
	seteq ubyte %566, 1		; <bool>:246 [#uses=1]
	br bool %246, label %247, label %246

; <label>:247		; preds = %244, %246
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %152		; <ubyte*>:361 [#uses=2]
	load ubyte* %361		; <ubyte>:568 [#uses=1]
	add ubyte %568, 1		; <ubyte>:569 [#uses=1]
	store ubyte %569, ubyte* %361
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %173		; <ubyte*>:362 [#uses=1]
	load ubyte* %362		; <ubyte>:570 [#uses=1]
	seteq ubyte %570, 0		; <bool>:247 [#uses=1]
	br bool %247, label %245, label %244

; <label>:248		; preds = %245, %251
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %152		; <ubyte*>:363 [#uses=1]
	load ubyte* %363		; <ubyte>:571 [#uses=1]
	seteq ubyte %571, 0		; <bool>:248 [#uses=1]
	br bool %248, label %251, label %250

; <label>:249		; preds = %245, %251
	add uint %151, 1		; <uint>:175 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %175		; <ubyte*>:364 [#uses=1]
	load ubyte* %364		; <ubyte>:572 [#uses=1]
	seteq ubyte %572, 0		; <bool>:249 [#uses=1]
	br bool %249, label %197, label %196

; <label>:250		; preds = %248, %250
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %152		; <ubyte*>:365 [#uses=2]
	load ubyte* %365		; <ubyte>:573 [#uses=2]
	add ubyte %573, 255		; <ubyte>:574 [#uses=1]
	store ubyte %574, ubyte* %365
	seteq ubyte %573, 1		; <bool>:250 [#uses=1]
	br bool %250, label %251, label %250

; <label>:251		; preds = %248, %250
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %174		; <ubyte*>:366 [#uses=2]
	load ubyte* %366		; <ubyte>:575 [#uses=1]
	add ubyte %575, 2		; <ubyte>:576 [#uses=1]
	store ubyte %576, ubyte* %366
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %152		; <ubyte*>:367 [#uses=1]
	load ubyte* %367		; <ubyte>:577 [#uses=1]
	seteq ubyte %577, 0		; <bool>:251 [#uses=1]
	br bool %251, label %249, label %248

; <label>:252		; preds = %27, %289
	phi uint [ %40, %27 ], [ %197, %289 ]		; <uint>:176 [#uses=20]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %176		; <ubyte*>:368 [#uses=2]
	load ubyte* %368		; <ubyte>:578 [#uses=1]
	add ubyte %578, 255		; <ubyte>:579 [#uses=1]
	store ubyte %579, ubyte* %368
	add uint %176, 10		; <uint>:177 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %177		; <ubyte*>:369 [#uses=1]
	load ubyte* %369		; <ubyte>:580 [#uses=1]
	seteq ubyte %580, 0		; <bool>:252 [#uses=1]
	br bool %252, label %255, label %254

; <label>:253		; preds = %27, %289
	phi uint [ %40, %27 ], [ %197, %289 ]		; <uint>:178 [#uses=1]
	add uint %178, 4294967295		; <uint>:179 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %179		; <ubyte*>:370 [#uses=1]
	load ubyte* %370		; <ubyte>:581 [#uses=1]
	seteq ubyte %581, 0		; <bool>:253 [#uses=1]
	br bool %253, label %25, label %24

; <label>:254		; preds = %252, %254
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %177		; <ubyte*>:371 [#uses=2]
	load ubyte* %371		; <ubyte>:582 [#uses=2]
	add ubyte %582, 255		; <ubyte>:583 [#uses=1]
	store ubyte %583, ubyte* %371
	seteq ubyte %582, 1		; <bool>:254 [#uses=1]
	br bool %254, label %255, label %254

; <label>:255		; preds = %252, %254
	add uint %176, 16		; <uint>:180 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %180		; <ubyte*>:372 [#uses=1]
	load ubyte* %372		; <ubyte>:584 [#uses=1]
	seteq ubyte %584, 0		; <bool>:255 [#uses=1]
	br bool %255, label %257, label %256

; <label>:256		; preds = %255, %256
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %180		; <ubyte*>:373 [#uses=2]
	load ubyte* %373		; <ubyte>:585 [#uses=2]
	add ubyte %585, 255		; <ubyte>:586 [#uses=1]
	store ubyte %586, ubyte* %373
	seteq ubyte %585, 1		; <bool>:256 [#uses=1]
	br bool %256, label %257, label %256

; <label>:257		; preds = %255, %256
	add uint %176, 22		; <uint>:181 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %181		; <ubyte*>:374 [#uses=1]
	load ubyte* %374		; <ubyte>:587 [#uses=1]
	seteq ubyte %587, 0		; <bool>:257 [#uses=1]
	br bool %257, label %259, label %258

; <label>:258		; preds = %257, %258
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %181		; <ubyte*>:375 [#uses=2]
	load ubyte* %375		; <ubyte>:588 [#uses=2]
	add ubyte %588, 255		; <ubyte>:589 [#uses=1]
	store ubyte %589, ubyte* %375
	seteq ubyte %588, 1		; <bool>:258 [#uses=1]
	br bool %258, label %259, label %258

; <label>:259		; preds = %257, %258
	add uint %176, 28		; <uint>:182 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %182		; <ubyte*>:376 [#uses=1]
	load ubyte* %376		; <ubyte>:590 [#uses=1]
	seteq ubyte %590, 0		; <bool>:259 [#uses=1]
	br bool %259, label %261, label %260

; <label>:260		; preds = %259, %260
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %182		; <ubyte*>:377 [#uses=2]
	load ubyte* %377		; <ubyte>:591 [#uses=2]
	add ubyte %591, 255		; <ubyte>:592 [#uses=1]
	store ubyte %592, ubyte* %377
	seteq ubyte %591, 1		; <bool>:260 [#uses=1]
	br bool %260, label %261, label %260

; <label>:261		; preds = %259, %260
	add uint %176, 34		; <uint>:183 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %183		; <ubyte*>:378 [#uses=1]
	load ubyte* %378		; <ubyte>:593 [#uses=1]
	seteq ubyte %593, 0		; <bool>:261 [#uses=1]
	br bool %261, label %263, label %262

; <label>:262		; preds = %261, %262
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %183		; <ubyte*>:379 [#uses=2]
	load ubyte* %379		; <ubyte>:594 [#uses=2]
	add ubyte %594, 255		; <ubyte>:595 [#uses=1]
	store ubyte %595, ubyte* %379
	seteq ubyte %594, 1		; <bool>:262 [#uses=1]
	br bool %262, label %263, label %262

; <label>:263		; preds = %261, %262
	add uint %176, 40		; <uint>:184 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %184		; <ubyte*>:380 [#uses=1]
	load ubyte* %380		; <ubyte>:596 [#uses=1]
	seteq ubyte %596, 0		; <bool>:263 [#uses=1]
	br bool %263, label %265, label %264

; <label>:264		; preds = %263, %264
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %184		; <ubyte*>:381 [#uses=2]
	load ubyte* %381		; <ubyte>:597 [#uses=2]
	add ubyte %597, 255		; <ubyte>:598 [#uses=1]
	store ubyte %598, ubyte* %381
	seteq ubyte %597, 1		; <bool>:264 [#uses=1]
	br bool %264, label %265, label %264

; <label>:265		; preds = %263, %264
	add uint %176, 46		; <uint>:185 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %185		; <ubyte*>:382 [#uses=1]
	load ubyte* %382		; <ubyte>:599 [#uses=1]
	seteq ubyte %599, 0		; <bool>:265 [#uses=1]
	br bool %265, label %267, label %266

; <label>:266		; preds = %265, %266
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %185		; <ubyte*>:383 [#uses=2]
	load ubyte* %383		; <ubyte>:600 [#uses=2]
	add ubyte %600, 255		; <ubyte>:601 [#uses=1]
	store ubyte %601, ubyte* %383
	seteq ubyte %600, 1		; <bool>:266 [#uses=1]
	br bool %266, label %267, label %266

; <label>:267		; preds = %265, %266
	add uint %176, 52		; <uint>:186 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %186		; <ubyte*>:384 [#uses=1]
	load ubyte* %384		; <ubyte>:602 [#uses=1]
	seteq ubyte %602, 0		; <bool>:267 [#uses=1]
	br bool %267, label %269, label %268

; <label>:268		; preds = %267, %268
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %186		; <ubyte*>:385 [#uses=2]
	load ubyte* %385		; <ubyte>:603 [#uses=2]
	add ubyte %603, 255		; <ubyte>:604 [#uses=1]
	store ubyte %604, ubyte* %385
	seteq ubyte %603, 1		; <bool>:268 [#uses=1]
	br bool %268, label %269, label %268

; <label>:269		; preds = %267, %268
	add uint %176, 58		; <uint>:187 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %187		; <ubyte*>:386 [#uses=1]
	load ubyte* %386		; <ubyte>:605 [#uses=1]
	seteq ubyte %605, 0		; <bool>:269 [#uses=1]
	br bool %269, label %271, label %270

; <label>:270		; preds = %269, %270
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %187		; <ubyte*>:387 [#uses=2]
	load ubyte* %387		; <ubyte>:606 [#uses=2]
	add ubyte %606, 255		; <ubyte>:607 [#uses=1]
	store ubyte %607, ubyte* %387
	seteq ubyte %606, 1		; <bool>:270 [#uses=1]
	br bool %270, label %271, label %270

; <label>:271		; preds = %269, %270
	add uint %176, 64		; <uint>:188 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %188		; <ubyte*>:388 [#uses=1]
	load ubyte* %388		; <ubyte>:608 [#uses=1]
	seteq ubyte %608, 0		; <bool>:271 [#uses=1]
	br bool %271, label %273, label %272

; <label>:272		; preds = %271, %272
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %188		; <ubyte*>:389 [#uses=2]
	load ubyte* %389		; <ubyte>:609 [#uses=2]
	add ubyte %609, 255		; <ubyte>:610 [#uses=1]
	store ubyte %610, ubyte* %389
	seteq ubyte %609, 1		; <bool>:272 [#uses=1]
	br bool %272, label %273, label %272

; <label>:273		; preds = %271, %272
	add uint %176, 70		; <uint>:189 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %189		; <ubyte*>:390 [#uses=1]
	load ubyte* %390		; <ubyte>:611 [#uses=1]
	seteq ubyte %611, 0		; <bool>:273 [#uses=1]
	br bool %273, label %275, label %274

; <label>:274		; preds = %273, %274
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %189		; <ubyte*>:391 [#uses=2]
	load ubyte* %391		; <ubyte>:612 [#uses=2]
	add ubyte %612, 255		; <ubyte>:613 [#uses=1]
	store ubyte %613, ubyte* %391
	seteq ubyte %612, 1		; <bool>:274 [#uses=1]
	br bool %274, label %275, label %274

; <label>:275		; preds = %273, %274
	add uint %176, 76		; <uint>:190 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %190		; <ubyte*>:392 [#uses=1]
	load ubyte* %392		; <ubyte>:614 [#uses=1]
	seteq ubyte %614, 0		; <bool>:275 [#uses=1]
	br bool %275, label %277, label %276

; <label>:276		; preds = %275, %276
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %190		; <ubyte*>:393 [#uses=2]
	load ubyte* %393		; <ubyte>:615 [#uses=2]
	add ubyte %615, 255		; <ubyte>:616 [#uses=1]
	store ubyte %616, ubyte* %393
	seteq ubyte %615, 1		; <bool>:276 [#uses=1]
	br bool %276, label %277, label %276

; <label>:277		; preds = %275, %276
	add uint %176, 82		; <uint>:191 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %191		; <ubyte*>:394 [#uses=1]
	load ubyte* %394		; <ubyte>:617 [#uses=1]
	seteq ubyte %617, 0		; <bool>:277 [#uses=1]
	br bool %277, label %279, label %278

; <label>:278		; preds = %277, %278
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %191		; <ubyte*>:395 [#uses=2]
	load ubyte* %395		; <ubyte>:618 [#uses=2]
	add ubyte %618, 255		; <ubyte>:619 [#uses=1]
	store ubyte %619, ubyte* %395
	seteq ubyte %618, 1		; <bool>:278 [#uses=1]
	br bool %278, label %279, label %278

; <label>:279		; preds = %277, %278
	add uint %176, 88		; <uint>:192 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %192		; <ubyte*>:396 [#uses=1]
	load ubyte* %396		; <ubyte>:620 [#uses=1]
	seteq ubyte %620, 0		; <bool>:279 [#uses=1]
	br bool %279, label %281, label %280

; <label>:280		; preds = %279, %280
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %192		; <ubyte*>:397 [#uses=2]
	load ubyte* %397		; <ubyte>:621 [#uses=2]
	add ubyte %621, 255		; <ubyte>:622 [#uses=1]
	store ubyte %622, ubyte* %397
	seteq ubyte %621, 1		; <bool>:280 [#uses=1]
	br bool %280, label %281, label %280

; <label>:281		; preds = %279, %280
	add uint %176, 94		; <uint>:193 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %193		; <ubyte*>:398 [#uses=1]
	load ubyte* %398		; <ubyte>:623 [#uses=1]
	seteq ubyte %623, 0		; <bool>:281 [#uses=1]
	br bool %281, label %283, label %282

; <label>:282		; preds = %281, %282
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %193		; <ubyte*>:399 [#uses=2]
	load ubyte* %399		; <ubyte>:624 [#uses=2]
	add ubyte %624, 255		; <ubyte>:625 [#uses=1]
	store ubyte %625, ubyte* %399
	seteq ubyte %624, 1		; <bool>:282 [#uses=1]
	br bool %282, label %283, label %282

; <label>:283		; preds = %281, %282
	add uint %176, 98		; <uint>:194 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %194		; <ubyte*>:400 [#uses=1]
	load ubyte* %400		; <ubyte>:626 [#uses=1]
	seteq ubyte %626, 0		; <bool>:283 [#uses=1]
	br bool %283, label %285, label %284

; <label>:284		; preds = %283, %284
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %194		; <ubyte*>:401 [#uses=2]
	load ubyte* %401		; <ubyte>:627 [#uses=2]
	add ubyte %627, 255		; <ubyte>:628 [#uses=1]
	store ubyte %628, ubyte* %401
	seteq ubyte %627, 1		; <bool>:284 [#uses=1]
	br bool %284, label %285, label %284

; <label>:285		; preds = %283, %284
	add uint %176, 100		; <uint>:195 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %195		; <ubyte*>:402 [#uses=1]
	load ubyte* %402		; <ubyte>:629 [#uses=1]
	seteq ubyte %629, 0		; <bool>:285 [#uses=1]
	br bool %285, label %287, label %286

; <label>:286		; preds = %285, %286
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %195		; <ubyte*>:403 [#uses=2]
	load ubyte* %403		; <ubyte>:630 [#uses=2]
	add ubyte %630, 255		; <ubyte>:631 [#uses=1]
	store ubyte %631, ubyte* %403
	seteq ubyte %630, 1		; <bool>:286 [#uses=1]
	br bool %286, label %287, label %286

; <label>:287		; preds = %285, %286
	add uint %176, 102		; <uint>:196 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %196		; <ubyte*>:404 [#uses=1]
	load ubyte* %404		; <ubyte>:632 [#uses=1]
	seteq ubyte %632, 0		; <bool>:287 [#uses=1]
	br bool %287, label %289, label %288

; <label>:288		; preds = %287, %288
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %196		; <ubyte*>:405 [#uses=2]
	load ubyte* %405		; <ubyte>:633 [#uses=2]
	add ubyte %633, 255		; <ubyte>:634 [#uses=1]
	store ubyte %634, ubyte* %405
	seteq ubyte %633, 1		; <bool>:288 [#uses=1]
	br bool %288, label %289, label %288

; <label>:289		; preds = %287, %288
	add uint %176, 1		; <uint>:197 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %197		; <ubyte*>:406 [#uses=1]
	load ubyte* %406		; <ubyte>:635 [#uses=1]
	seteq ubyte %635, 0		; <bool>:289 [#uses=1]
	br bool %289, label %253, label %252

; <label>:290		; preds = %25, %359
	phi uint [ %37, %25 ], [ %235, %359 ]		; <uint>:198 [#uses=36]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %198		; <ubyte*>:407 [#uses=2]
	load ubyte* %407		; <ubyte>:636 [#uses=1]
	add ubyte %636, 255		; <ubyte>:637 [#uses=1]
	store ubyte %637, ubyte* %407
	add uint %198, 4294967101		; <uint>:199 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %199		; <ubyte*>:408 [#uses=1]
	load ubyte* %408		; <ubyte>:638 [#uses=1]
	seteq ubyte %638, 0		; <bool>:290 [#uses=1]
	br bool %290, label %293, label %292

; <label>:291		; preds = %25, %359
	phi uint [ %37, %25 ], [ %235, %359 ]		; <uint>:200 [#uses=1]
	add uint %200, 4294967295		; <uint>:201 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %201		; <ubyte*>:409 [#uses=1]
	load ubyte* %409		; <ubyte>:639 [#uses=1]
	seteq ubyte %639, 0		; <bool>:291 [#uses=1]
	br bool %291, label %23, label %22

; <label>:292		; preds = %290, %292
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %199		; <ubyte*>:410 [#uses=2]
	load ubyte* %410		; <ubyte>:640 [#uses=2]
	add ubyte %640, 255		; <ubyte>:641 [#uses=1]
	store ubyte %641, ubyte* %410
	seteq ubyte %640, 1		; <bool>:292 [#uses=1]
	br bool %292, label %293, label %292

; <label>:293		; preds = %290, %292
	add uint %198, 4294967207		; <uint>:202 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %202		; <ubyte*>:411 [#uses=1]
	load ubyte* %411		; <ubyte>:642 [#uses=1]
	seteq ubyte %642, 0		; <bool>:293 [#uses=1]
	br bool %293, label %295, label %294

; <label>:294		; preds = %293, %294
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %199		; <ubyte*>:412 [#uses=2]
	load ubyte* %412		; <ubyte>:643 [#uses=1]
	add ubyte %643, 1		; <ubyte>:644 [#uses=1]
	store ubyte %644, ubyte* %412
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %202		; <ubyte*>:413 [#uses=2]
	load ubyte* %413		; <ubyte>:645 [#uses=2]
	add ubyte %645, 255		; <ubyte>:646 [#uses=1]
	store ubyte %646, ubyte* %413
	seteq ubyte %645, 1		; <bool>:294 [#uses=1]
	br bool %294, label %295, label %294

; <label>:295		; preds = %293, %294
	add uint %198, 4294967107		; <uint>:203 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %203		; <ubyte*>:414 [#uses=1]
	load ubyte* %414		; <ubyte>:647 [#uses=1]
	seteq ubyte %647, 0		; <bool>:295 [#uses=1]
	br bool %295, label %297, label %296

; <label>:296		; preds = %295, %296
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %203		; <ubyte*>:415 [#uses=2]
	load ubyte* %415		; <ubyte>:648 [#uses=2]
	add ubyte %648, 255		; <ubyte>:649 [#uses=1]
	store ubyte %649, ubyte* %415
	seteq ubyte %648, 1		; <bool>:296 [#uses=1]
	br bool %296, label %297, label %296

; <label>:297		; preds = %295, %296
	add uint %198, 4294967213		; <uint>:204 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %204		; <ubyte*>:416 [#uses=1]
	load ubyte* %416		; <ubyte>:650 [#uses=1]
	seteq ubyte %650, 0		; <bool>:297 [#uses=1]
	br bool %297, label %299, label %298

; <label>:298		; preds = %297, %298
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %203		; <ubyte*>:417 [#uses=2]
	load ubyte* %417		; <ubyte>:651 [#uses=1]
	add ubyte %651, 1		; <ubyte>:652 [#uses=1]
	store ubyte %652, ubyte* %417
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %204		; <ubyte*>:418 [#uses=2]
	load ubyte* %418		; <ubyte>:653 [#uses=2]
	add ubyte %653, 255		; <ubyte>:654 [#uses=1]
	store ubyte %654, ubyte* %418
	seteq ubyte %653, 1		; <bool>:298 [#uses=1]
	br bool %298, label %299, label %298

; <label>:299		; preds = %297, %298
	add uint %198, 4294967113		; <uint>:205 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %205		; <ubyte*>:419 [#uses=1]
	load ubyte* %419		; <ubyte>:655 [#uses=1]
	seteq ubyte %655, 0		; <bool>:299 [#uses=1]
	br bool %299, label %301, label %300

; <label>:300		; preds = %299, %300
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %205		; <ubyte*>:420 [#uses=2]
	load ubyte* %420		; <ubyte>:656 [#uses=2]
	add ubyte %656, 255		; <ubyte>:657 [#uses=1]
	store ubyte %657, ubyte* %420
	seteq ubyte %656, 1		; <bool>:300 [#uses=1]
	br bool %300, label %301, label %300

; <label>:301		; preds = %299, %300
	add uint %198, 4294967219		; <uint>:206 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %206		; <ubyte*>:421 [#uses=1]
	load ubyte* %421		; <ubyte>:658 [#uses=1]
	seteq ubyte %658, 0		; <bool>:301 [#uses=1]
	br bool %301, label %303, label %302

; <label>:302		; preds = %301, %302
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %205		; <ubyte*>:422 [#uses=2]
	load ubyte* %422		; <ubyte>:659 [#uses=1]
	add ubyte %659, 1		; <ubyte>:660 [#uses=1]
	store ubyte %660, ubyte* %422
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %206		; <ubyte*>:423 [#uses=2]
	load ubyte* %423		; <ubyte>:661 [#uses=2]
	add ubyte %661, 255		; <ubyte>:662 [#uses=1]
	store ubyte %662, ubyte* %423
	seteq ubyte %661, 1		; <bool>:302 [#uses=1]
	br bool %302, label %303, label %302

; <label>:303		; preds = %301, %302
	add uint %198, 4294967119		; <uint>:207 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %207		; <ubyte*>:424 [#uses=1]
	load ubyte* %424		; <ubyte>:663 [#uses=1]
	seteq ubyte %663, 0		; <bool>:303 [#uses=1]
	br bool %303, label %305, label %304

; <label>:304		; preds = %303, %304
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %207		; <ubyte*>:425 [#uses=2]
	load ubyte* %425		; <ubyte>:664 [#uses=2]
	add ubyte %664, 255		; <ubyte>:665 [#uses=1]
	store ubyte %665, ubyte* %425
	seteq ubyte %664, 1		; <bool>:304 [#uses=1]
	br bool %304, label %305, label %304

; <label>:305		; preds = %303, %304
	add uint %198, 4294967225		; <uint>:208 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %208		; <ubyte*>:426 [#uses=1]
	load ubyte* %426		; <ubyte>:666 [#uses=1]
	seteq ubyte %666, 0		; <bool>:305 [#uses=1]
	br bool %305, label %307, label %306

; <label>:306		; preds = %305, %306
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %207		; <ubyte*>:427 [#uses=2]
	load ubyte* %427		; <ubyte>:667 [#uses=1]
	add ubyte %667, 1		; <ubyte>:668 [#uses=1]
	store ubyte %668, ubyte* %427
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %208		; <ubyte*>:428 [#uses=2]
	load ubyte* %428		; <ubyte>:669 [#uses=2]
	add ubyte %669, 255		; <ubyte>:670 [#uses=1]
	store ubyte %670, ubyte* %428
	seteq ubyte %669, 1		; <bool>:306 [#uses=1]
	br bool %306, label %307, label %306

; <label>:307		; preds = %305, %306
	add uint %198, 4294967125		; <uint>:209 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %209		; <ubyte*>:429 [#uses=1]
	load ubyte* %429		; <ubyte>:671 [#uses=1]
	seteq ubyte %671, 0		; <bool>:307 [#uses=1]
	br bool %307, label %309, label %308

; <label>:308		; preds = %307, %308
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %209		; <ubyte*>:430 [#uses=2]
	load ubyte* %430		; <ubyte>:672 [#uses=2]
	add ubyte %672, 255		; <ubyte>:673 [#uses=1]
	store ubyte %673, ubyte* %430
	seteq ubyte %672, 1		; <bool>:308 [#uses=1]
	br bool %308, label %309, label %308

; <label>:309		; preds = %307, %308
	add uint %198, 4294967231		; <uint>:210 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %210		; <ubyte*>:431 [#uses=1]
	load ubyte* %431		; <ubyte>:674 [#uses=1]
	seteq ubyte %674, 0		; <bool>:309 [#uses=1]
	br bool %309, label %311, label %310

; <label>:310		; preds = %309, %310
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %209		; <ubyte*>:432 [#uses=2]
	load ubyte* %432		; <ubyte>:675 [#uses=1]
	add ubyte %675, 1		; <ubyte>:676 [#uses=1]
	store ubyte %676, ubyte* %432
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %210		; <ubyte*>:433 [#uses=2]
	load ubyte* %433		; <ubyte>:677 [#uses=2]
	add ubyte %677, 255		; <ubyte>:678 [#uses=1]
	store ubyte %678, ubyte* %433
	seteq ubyte %677, 1		; <bool>:310 [#uses=1]
	br bool %310, label %311, label %310

; <label>:311		; preds = %309, %310
	add uint %198, 4294967131		; <uint>:211 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %211		; <ubyte*>:434 [#uses=1]
	load ubyte* %434		; <ubyte>:679 [#uses=1]
	seteq ubyte %679, 0		; <bool>:311 [#uses=1]
	br bool %311, label %313, label %312

; <label>:312		; preds = %311, %312
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %211		; <ubyte*>:435 [#uses=2]
	load ubyte* %435		; <ubyte>:680 [#uses=2]
	add ubyte %680, 255		; <ubyte>:681 [#uses=1]
	store ubyte %681, ubyte* %435
	seteq ubyte %680, 1		; <bool>:312 [#uses=1]
	br bool %312, label %313, label %312

; <label>:313		; preds = %311, %312
	add uint %198, 4294967237		; <uint>:212 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %212		; <ubyte*>:436 [#uses=1]
	load ubyte* %436		; <ubyte>:682 [#uses=1]
	seteq ubyte %682, 0		; <bool>:313 [#uses=1]
	br bool %313, label %315, label %314

; <label>:314		; preds = %313, %314
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %211		; <ubyte*>:437 [#uses=2]
	load ubyte* %437		; <ubyte>:683 [#uses=1]
	add ubyte %683, 1		; <ubyte>:684 [#uses=1]
	store ubyte %684, ubyte* %437
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %212		; <ubyte*>:438 [#uses=2]
	load ubyte* %438		; <ubyte>:685 [#uses=2]
	add ubyte %685, 255		; <ubyte>:686 [#uses=1]
	store ubyte %686, ubyte* %438
	seteq ubyte %685, 1		; <bool>:314 [#uses=1]
	br bool %314, label %315, label %314

; <label>:315		; preds = %313, %314
	add uint %198, 4294967137		; <uint>:213 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %213		; <ubyte*>:439 [#uses=1]
	load ubyte* %439		; <ubyte>:687 [#uses=1]
	seteq ubyte %687, 0		; <bool>:315 [#uses=1]
	br bool %315, label %317, label %316

; <label>:316		; preds = %315, %316
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %213		; <ubyte*>:440 [#uses=2]
	load ubyte* %440		; <ubyte>:688 [#uses=2]
	add ubyte %688, 255		; <ubyte>:689 [#uses=1]
	store ubyte %689, ubyte* %440
	seteq ubyte %688, 1		; <bool>:316 [#uses=1]
	br bool %316, label %317, label %316

; <label>:317		; preds = %315, %316
	add uint %198, 4294967243		; <uint>:214 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %214		; <ubyte*>:441 [#uses=1]
	load ubyte* %441		; <ubyte>:690 [#uses=1]
	seteq ubyte %690, 0		; <bool>:317 [#uses=1]
	br bool %317, label %319, label %318

; <label>:318		; preds = %317, %318
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %213		; <ubyte*>:442 [#uses=2]
	load ubyte* %442		; <ubyte>:691 [#uses=1]
	add ubyte %691, 1		; <ubyte>:692 [#uses=1]
	store ubyte %692, ubyte* %442
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %214		; <ubyte*>:443 [#uses=2]
	load ubyte* %443		; <ubyte>:693 [#uses=2]
	add ubyte %693, 255		; <ubyte>:694 [#uses=1]
	store ubyte %694, ubyte* %443
	seteq ubyte %693, 1		; <bool>:318 [#uses=1]
	br bool %318, label %319, label %318

; <label>:319		; preds = %317, %318
	add uint %198, 4294967143		; <uint>:215 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %215		; <ubyte*>:444 [#uses=1]
	load ubyte* %444		; <ubyte>:695 [#uses=1]
	seteq ubyte %695, 0		; <bool>:319 [#uses=1]
	br bool %319, label %321, label %320

; <label>:320		; preds = %319, %320
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %215		; <ubyte*>:445 [#uses=2]
	load ubyte* %445		; <ubyte>:696 [#uses=2]
	add ubyte %696, 255		; <ubyte>:697 [#uses=1]
	store ubyte %697, ubyte* %445
	seteq ubyte %696, 1		; <bool>:320 [#uses=1]
	br bool %320, label %321, label %320

; <label>:321		; preds = %319, %320
	add uint %198, 4294967249		; <uint>:216 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %216		; <ubyte*>:446 [#uses=1]
	load ubyte* %446		; <ubyte>:698 [#uses=1]
	seteq ubyte %698, 0		; <bool>:321 [#uses=1]
	br bool %321, label %323, label %322

; <label>:322		; preds = %321, %322
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %215		; <ubyte*>:447 [#uses=2]
	load ubyte* %447		; <ubyte>:699 [#uses=1]
	add ubyte %699, 1		; <ubyte>:700 [#uses=1]
	store ubyte %700, ubyte* %447
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %216		; <ubyte*>:448 [#uses=2]
	load ubyte* %448		; <ubyte>:701 [#uses=2]
	add ubyte %701, 255		; <ubyte>:702 [#uses=1]
	store ubyte %702, ubyte* %448
	seteq ubyte %701, 1		; <bool>:322 [#uses=1]
	br bool %322, label %323, label %322

; <label>:323		; preds = %321, %322
	add uint %198, 4294967149		; <uint>:217 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %217		; <ubyte*>:449 [#uses=1]
	load ubyte* %449		; <ubyte>:703 [#uses=1]
	seteq ubyte %703, 0		; <bool>:323 [#uses=1]
	br bool %323, label %325, label %324

; <label>:324		; preds = %323, %324
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %217		; <ubyte*>:450 [#uses=2]
	load ubyte* %450		; <ubyte>:704 [#uses=2]
	add ubyte %704, 255		; <ubyte>:705 [#uses=1]
	store ubyte %705, ubyte* %450
	seteq ubyte %704, 1		; <bool>:324 [#uses=1]
	br bool %324, label %325, label %324

; <label>:325		; preds = %323, %324
	add uint %198, 4294967255		; <uint>:218 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %218		; <ubyte*>:451 [#uses=1]
	load ubyte* %451		; <ubyte>:706 [#uses=1]
	seteq ubyte %706, 0		; <bool>:325 [#uses=1]
	br bool %325, label %327, label %326

; <label>:326		; preds = %325, %326
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %217		; <ubyte*>:452 [#uses=2]
	load ubyte* %452		; <ubyte>:707 [#uses=1]
	add ubyte %707, 1		; <ubyte>:708 [#uses=1]
	store ubyte %708, ubyte* %452
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %218		; <ubyte*>:453 [#uses=2]
	load ubyte* %453		; <ubyte>:709 [#uses=2]
	add ubyte %709, 255		; <ubyte>:710 [#uses=1]
	store ubyte %710, ubyte* %453
	seteq ubyte %709, 1		; <bool>:326 [#uses=1]
	br bool %326, label %327, label %326

; <label>:327		; preds = %325, %326
	add uint %198, 4294967155		; <uint>:219 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %219		; <ubyte*>:454 [#uses=1]
	load ubyte* %454		; <ubyte>:711 [#uses=1]
	seteq ubyte %711, 0		; <bool>:327 [#uses=1]
	br bool %327, label %329, label %328

; <label>:328		; preds = %327, %328
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %219		; <ubyte*>:455 [#uses=2]
	load ubyte* %455		; <ubyte>:712 [#uses=2]
	add ubyte %712, 255		; <ubyte>:713 [#uses=1]
	store ubyte %713, ubyte* %455
	seteq ubyte %712, 1		; <bool>:328 [#uses=1]
	br bool %328, label %329, label %328

; <label>:329		; preds = %327, %328
	add uint %198, 4294967261		; <uint>:220 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %220		; <ubyte*>:456 [#uses=1]
	load ubyte* %456		; <ubyte>:714 [#uses=1]
	seteq ubyte %714, 0		; <bool>:329 [#uses=1]
	br bool %329, label %331, label %330

; <label>:330		; preds = %329, %330
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %219		; <ubyte*>:457 [#uses=2]
	load ubyte* %457		; <ubyte>:715 [#uses=1]
	add ubyte %715, 1		; <ubyte>:716 [#uses=1]
	store ubyte %716, ubyte* %457
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %220		; <ubyte*>:458 [#uses=2]
	load ubyte* %458		; <ubyte>:717 [#uses=2]
	add ubyte %717, 255		; <ubyte>:718 [#uses=1]
	store ubyte %718, ubyte* %458
	seteq ubyte %717, 1		; <bool>:330 [#uses=1]
	br bool %330, label %331, label %330

; <label>:331		; preds = %329, %330
	add uint %198, 4294967161		; <uint>:221 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %221		; <ubyte*>:459 [#uses=1]
	load ubyte* %459		; <ubyte>:719 [#uses=1]
	seteq ubyte %719, 0		; <bool>:331 [#uses=1]
	br bool %331, label %333, label %332

; <label>:332		; preds = %331, %332
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %221		; <ubyte*>:460 [#uses=2]
	load ubyte* %460		; <ubyte>:720 [#uses=2]
	add ubyte %720, 255		; <ubyte>:721 [#uses=1]
	store ubyte %721, ubyte* %460
	seteq ubyte %720, 1		; <bool>:332 [#uses=1]
	br bool %332, label %333, label %332

; <label>:333		; preds = %331, %332
	add uint %198, 4294967267		; <uint>:222 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %222		; <ubyte*>:461 [#uses=1]
	load ubyte* %461		; <ubyte>:722 [#uses=1]
	seteq ubyte %722, 0		; <bool>:333 [#uses=1]
	br bool %333, label %335, label %334

; <label>:334		; preds = %333, %334
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %221		; <ubyte*>:462 [#uses=2]
	load ubyte* %462		; <ubyte>:723 [#uses=1]
	add ubyte %723, 1		; <ubyte>:724 [#uses=1]
	store ubyte %724, ubyte* %462
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %222		; <ubyte*>:463 [#uses=2]
	load ubyte* %463		; <ubyte>:725 [#uses=2]
	add ubyte %725, 255		; <ubyte>:726 [#uses=1]
	store ubyte %726, ubyte* %463
	seteq ubyte %725, 1		; <bool>:334 [#uses=1]
	br bool %334, label %335, label %334

; <label>:335		; preds = %333, %334
	add uint %198, 4294967167		; <uint>:223 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %223		; <ubyte*>:464 [#uses=1]
	load ubyte* %464		; <ubyte>:727 [#uses=1]
	seteq ubyte %727, 0		; <bool>:335 [#uses=1]
	br bool %335, label %337, label %336

; <label>:336		; preds = %335, %336
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %223		; <ubyte*>:465 [#uses=2]
	load ubyte* %465		; <ubyte>:728 [#uses=2]
	add ubyte %728, 255		; <ubyte>:729 [#uses=1]
	store ubyte %729, ubyte* %465
	seteq ubyte %728, 1		; <bool>:336 [#uses=1]
	br bool %336, label %337, label %336

; <label>:337		; preds = %335, %336
	add uint %198, 4294967273		; <uint>:224 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %224		; <ubyte*>:466 [#uses=1]
	load ubyte* %466		; <ubyte>:730 [#uses=1]
	seteq ubyte %730, 0		; <bool>:337 [#uses=1]
	br bool %337, label %339, label %338

; <label>:338		; preds = %337, %338
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %223		; <ubyte*>:467 [#uses=2]
	load ubyte* %467		; <ubyte>:731 [#uses=1]
	add ubyte %731, 1		; <ubyte>:732 [#uses=1]
	store ubyte %732, ubyte* %467
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %224		; <ubyte*>:468 [#uses=2]
	load ubyte* %468		; <ubyte>:733 [#uses=2]
	add ubyte %733, 255		; <ubyte>:734 [#uses=1]
	store ubyte %734, ubyte* %468
	seteq ubyte %733, 1		; <bool>:338 [#uses=1]
	br bool %338, label %339, label %338

; <label>:339		; preds = %337, %338
	add uint %198, 4294967173		; <uint>:225 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %225		; <ubyte*>:469 [#uses=1]
	load ubyte* %469		; <ubyte>:735 [#uses=1]
	seteq ubyte %735, 0		; <bool>:339 [#uses=1]
	br bool %339, label %341, label %340

; <label>:340		; preds = %339, %340
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %225		; <ubyte*>:470 [#uses=2]
	load ubyte* %470		; <ubyte>:736 [#uses=2]
	add ubyte %736, 255		; <ubyte>:737 [#uses=1]
	store ubyte %737, ubyte* %470
	seteq ubyte %736, 1		; <bool>:340 [#uses=1]
	br bool %340, label %341, label %340

; <label>:341		; preds = %339, %340
	add uint %198, 4294967279		; <uint>:226 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %226		; <ubyte*>:471 [#uses=1]
	load ubyte* %471		; <ubyte>:738 [#uses=1]
	seteq ubyte %738, 0		; <bool>:341 [#uses=1]
	br bool %341, label %343, label %342

; <label>:342		; preds = %341, %342
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %225		; <ubyte*>:472 [#uses=2]
	load ubyte* %472		; <ubyte>:739 [#uses=1]
	add ubyte %739, 1		; <ubyte>:740 [#uses=1]
	store ubyte %740, ubyte* %472
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %226		; <ubyte*>:473 [#uses=2]
	load ubyte* %473		; <ubyte>:741 [#uses=2]
	add ubyte %741, 255		; <ubyte>:742 [#uses=1]
	store ubyte %742, ubyte* %473
	seteq ubyte %741, 1		; <bool>:342 [#uses=1]
	br bool %342, label %343, label %342

; <label>:343		; preds = %341, %342
	add uint %198, 4294967179		; <uint>:227 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %227		; <ubyte*>:474 [#uses=1]
	load ubyte* %474		; <ubyte>:743 [#uses=1]
	seteq ubyte %743, 0		; <bool>:343 [#uses=1]
	br bool %343, label %345, label %344

; <label>:344		; preds = %343, %344
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %227		; <ubyte*>:475 [#uses=2]
	load ubyte* %475		; <ubyte>:744 [#uses=2]
	add ubyte %744, 255		; <ubyte>:745 [#uses=1]
	store ubyte %745, ubyte* %475
	seteq ubyte %744, 1		; <bool>:344 [#uses=1]
	br bool %344, label %345, label %344

; <label>:345		; preds = %343, %344
	add uint %198, 4294967285		; <uint>:228 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %228		; <ubyte*>:476 [#uses=1]
	load ubyte* %476		; <ubyte>:746 [#uses=1]
	seteq ubyte %746, 0		; <bool>:345 [#uses=1]
	br bool %345, label %347, label %346

; <label>:346		; preds = %345, %346
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %227		; <ubyte*>:477 [#uses=2]
	load ubyte* %477		; <ubyte>:747 [#uses=1]
	add ubyte %747, 1		; <ubyte>:748 [#uses=1]
	store ubyte %748, ubyte* %477
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %228		; <ubyte*>:478 [#uses=2]
	load ubyte* %478		; <ubyte>:749 [#uses=2]
	add ubyte %749, 255		; <ubyte>:750 [#uses=1]
	store ubyte %750, ubyte* %478
	seteq ubyte %749, 1		; <bool>:346 [#uses=1]
	br bool %346, label %347, label %346

; <label>:347		; preds = %345, %346
	add uint %198, 4294967185		; <uint>:229 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %229		; <ubyte*>:479 [#uses=1]
	load ubyte* %479		; <ubyte>:751 [#uses=1]
	seteq ubyte %751, 0		; <bool>:347 [#uses=1]
	br bool %347, label %349, label %348

; <label>:348		; preds = %347, %348
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %229		; <ubyte*>:480 [#uses=2]
	load ubyte* %480		; <ubyte>:752 [#uses=2]
	add ubyte %752, 255		; <ubyte>:753 [#uses=1]
	store ubyte %753, ubyte* %480
	seteq ubyte %752, 1		; <bool>:348 [#uses=1]
	br bool %348, label %349, label %348

; <label>:349		; preds = %347, %348
	add uint %198, 4294967291		; <uint>:230 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %230		; <ubyte*>:481 [#uses=1]
	load ubyte* %481		; <ubyte>:754 [#uses=1]
	seteq ubyte %754, 0		; <bool>:349 [#uses=1]
	br bool %349, label %351, label %350

; <label>:350		; preds = %349, %350
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %229		; <ubyte*>:482 [#uses=2]
	load ubyte* %482		; <ubyte>:755 [#uses=1]
	add ubyte %755, 1		; <ubyte>:756 [#uses=1]
	store ubyte %756, ubyte* %482
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %230		; <ubyte*>:483 [#uses=2]
	load ubyte* %483		; <ubyte>:757 [#uses=2]
	add ubyte %757, 255		; <ubyte>:758 [#uses=1]
	store ubyte %758, ubyte* %483
	seteq ubyte %757, 1		; <bool>:350 [#uses=1]
	br bool %350, label %351, label %350

; <label>:351		; preds = %349, %350
	add uint %198, 4294967197		; <uint>:231 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %231		; <ubyte*>:484 [#uses=1]
	load ubyte* %484		; <ubyte>:759 [#uses=1]
	seteq ubyte %759, 0		; <bool>:351 [#uses=1]
	br bool %351, label %353, label %352

; <label>:352		; preds = %351, %352
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %231		; <ubyte*>:485 [#uses=2]
	load ubyte* %485		; <ubyte>:760 [#uses=2]
	add ubyte %760, 255		; <ubyte>:761 [#uses=1]
	store ubyte %761, ubyte* %485
	seteq ubyte %760, 1		; <bool>:352 [#uses=1]
	br bool %352, label %353, label %352

; <label>:353		; preds = %351, %352
	add uint %198, 4294967195		; <uint>:232 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %232		; <ubyte*>:486 [#uses=1]
	load ubyte* %486		; <ubyte>:762 [#uses=1]
	seteq ubyte %762, 0		; <bool>:353 [#uses=1]
	br bool %353, label %355, label %354

; <label>:354		; preds = %353, %354
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %232		; <ubyte*>:487 [#uses=2]
	load ubyte* %487		; <ubyte>:763 [#uses=2]
	add ubyte %763, 255		; <ubyte>:764 [#uses=1]
	store ubyte %764, ubyte* %487
	seteq ubyte %763, 1		; <bool>:354 [#uses=1]
	br bool %354, label %355, label %354

; <label>:355		; preds = %353, %354
	add uint %198, 4294967191		; <uint>:233 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %233		; <ubyte*>:488 [#uses=1]
	load ubyte* %488		; <ubyte>:765 [#uses=1]
	seteq ubyte %765, 0		; <bool>:355 [#uses=1]
	br bool %355, label %357, label %356

; <label>:356		; preds = %355, %356
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %233		; <ubyte*>:489 [#uses=2]
	load ubyte* %489		; <ubyte>:766 [#uses=2]
	add ubyte %766, 255		; <ubyte>:767 [#uses=1]
	store ubyte %767, ubyte* %489
	seteq ubyte %766, 1		; <bool>:356 [#uses=1]
	br bool %356, label %357, label %356

; <label>:357		; preds = %355, %356
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %232		; <ubyte*>:490 [#uses=1]
	load ubyte* %490		; <ubyte>:768 [#uses=1]
	seteq ubyte %768, 0		; <bool>:357 [#uses=1]
	br bool %357, label %359, label %358

; <label>:358		; preds = %357, %358
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %233		; <ubyte*>:491 [#uses=2]
	load ubyte* %491		; <ubyte>:769 [#uses=1]
	add ubyte %769, 1		; <ubyte>:770 [#uses=1]
	store ubyte %770, ubyte* %491
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %232		; <ubyte*>:492 [#uses=2]
	load ubyte* %492		; <ubyte>:771 [#uses=2]
	add ubyte %771, 255		; <ubyte>:772 [#uses=1]
	store ubyte %772, ubyte* %492
	seteq ubyte %771, 1		; <bool>:358 [#uses=1]
	br bool %358, label %359, label %358

; <label>:359		; preds = %357, %358
	add uint %198, 4294967090		; <uint>:234 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %234		; <ubyte*>:493 [#uses=2]
	load ubyte* %493		; <ubyte>:773 [#uses=1]
	add ubyte %773, 7		; <ubyte>:774 [#uses=1]
	store ubyte %774, ubyte* %493
	add uint %198, 4294967092		; <uint>:235 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %235		; <ubyte*>:494 [#uses=1]
	load ubyte* %494		; <ubyte>:775 [#uses=1]
	seteq ubyte %775, 0		; <bool>:359 [#uses=1]
	br bool %359, label %291, label %290

; <label>:360		; preds = %23, %401
	phi uint [ %34, %23 ], [ %264, %401 ]		; <uint>:236 [#uses=9]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %236		; <ubyte*>:495 [#uses=2]
	load ubyte* %495		; <ubyte>:776 [#uses=1]
	add ubyte %776, 255		; <ubyte>:777 [#uses=1]
	store ubyte %777, ubyte* %495
	add uint %236, 104		; <uint>:237 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %237		; <ubyte*>:496 [#uses=1]
	call ubyte %inputcell( )		; <ubyte>:778 [#uses=1]
	store ubyte %778, ubyte* %496
	add uint %236, 106		; <uint>:238 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %238		; <ubyte*>:497 [#uses=1]
	load ubyte* %497		; <ubyte>:779 [#uses=1]
	seteq ubyte %779, 0		; <bool>:360 [#uses=1]
	br bool %360, label %363, label %362

; <label>:361		; preds = %23, %401
	phi uint [ %34, %23 ], [ %264, %401 ]		; <uint>:239 [#uses=1]
	add uint %239, 4294967295		; <uint>:240 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %240		; <ubyte*>:498 [#uses=1]
	load ubyte* %498		; <ubyte>:780 [#uses=1]
	seteq ubyte %780, 0		; <bool>:361 [#uses=1]
	br bool %361, label %21, label %20

; <label>:362		; preds = %360, %362
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %238		; <ubyte*>:499 [#uses=2]
	load ubyte* %499		; <ubyte>:781 [#uses=2]
	add ubyte %781, 255		; <ubyte>:782 [#uses=1]
	store ubyte %782, ubyte* %499
	seteq ubyte %781, 1		; <bool>:362 [#uses=1]
	br bool %362, label %363, label %362

; <label>:363		; preds = %360, %362
	add uint %236, 100		; <uint>:241 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %241		; <ubyte*>:500 [#uses=1]
	load ubyte* %500		; <ubyte>:783 [#uses=1]
	seteq ubyte %783, 0		; <bool>:363 [#uses=1]
	br bool %363, label %365, label %364

; <label>:364		; preds = %363, %364
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %241		; <ubyte*>:501 [#uses=2]
	load ubyte* %501		; <ubyte>:784 [#uses=1]
	add ubyte %784, 255		; <ubyte>:785 [#uses=1]
	store ubyte %785, ubyte* %501
	add uint %236, 101		; <uint>:242 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %242		; <ubyte*>:502 [#uses=2]
	load ubyte* %502		; <ubyte>:786 [#uses=1]
	add ubyte %786, 1		; <ubyte>:787 [#uses=1]
	store ubyte %787, ubyte* %502
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %238		; <ubyte*>:503 [#uses=2]
	load ubyte* %503		; <ubyte>:788 [#uses=1]
	add ubyte %788, 1		; <ubyte>:789 [#uses=1]
	store ubyte %789, ubyte* %503
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %241		; <ubyte*>:504 [#uses=1]
	load ubyte* %504		; <ubyte>:790 [#uses=1]
	seteq ubyte %790, 0		; <bool>:364 [#uses=1]
	br bool %364, label %365, label %364

; <label>:365		; preds = %363, %364
	add uint %236, 101		; <uint>:243 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %243		; <ubyte*>:505 [#uses=1]
	load ubyte* %505		; <ubyte>:791 [#uses=1]
	seteq ubyte %791, 0		; <bool>:365 [#uses=1]
	br bool %365, label %367, label %366

; <label>:366		; preds = %365, %366
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %241		; <ubyte*>:506 [#uses=2]
	load ubyte* %506		; <ubyte>:792 [#uses=1]
	add ubyte %792, 1		; <ubyte>:793 [#uses=1]
	store ubyte %793, ubyte* %506
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %243		; <ubyte*>:507 [#uses=2]
	load ubyte* %507		; <ubyte>:794 [#uses=2]
	add ubyte %794, 255		; <ubyte>:795 [#uses=1]
	store ubyte %795, ubyte* %507
	seteq ubyte %794, 1		; <bool>:366 [#uses=1]
	br bool %366, label %367, label %366

; <label>:367		; preds = %365, %366
	add uint %236, 12		; <uint>:244 [#uses=6]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %244		; <ubyte*>:508 [#uses=1]
	load ubyte* %508		; <ubyte>:796 [#uses=1]
	seteq ubyte %796, 0		; <bool>:367 [#uses=1]
	br bool %367, label %369, label %368

; <label>:368		; preds = %367, %368
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %244		; <ubyte*>:509 [#uses=2]
	load ubyte* %509		; <ubyte>:797 [#uses=2]
	add ubyte %797, 255		; <ubyte>:798 [#uses=1]
	store ubyte %798, ubyte* %509
	seteq ubyte %797, 1		; <bool>:368 [#uses=1]
	br bool %368, label %369, label %368

; <label>:369		; preds = %367, %368
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %238		; <ubyte*>:510 [#uses=1]
	load ubyte* %510		; <ubyte>:799 [#uses=1]
	seteq ubyte %799, 0		; <bool>:369 [#uses=1]
	br bool %369, label %371, label %370

; <label>:370		; preds = %369, %370
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %244		; <ubyte*>:511 [#uses=2]
	load ubyte* %511		; <ubyte>:800 [#uses=1]
	add ubyte %800, 1		; <ubyte>:801 [#uses=1]
	store ubyte %801, ubyte* %511
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %238		; <ubyte*>:512 [#uses=2]
	load ubyte* %512		; <ubyte>:802 [#uses=2]
	add ubyte %802, 255		; <ubyte>:803 [#uses=1]
	store ubyte %803, ubyte* %512
	seteq ubyte %802, 1		; <bool>:370 [#uses=1]
	br bool %370, label %371, label %370

; <label>:371		; preds = %369, %370
	add uint %236, 2		; <uint>:245 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %245		; <ubyte*>:513 [#uses=1]
	load ubyte* %513		; <ubyte>:804 [#uses=1]
	seteq ubyte %804, 0		; <bool>:371 [#uses=1]
	br bool %371, label %373, label %372

; <label>:372		; preds = %371, %372
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %245		; <ubyte*>:514 [#uses=2]
	load ubyte* %514		; <ubyte>:805 [#uses=2]
	add ubyte %805, 255		; <ubyte>:806 [#uses=1]
	store ubyte %806, ubyte* %514
	seteq ubyte %805, 1		; <bool>:372 [#uses=1]
	br bool %372, label %373, label %372

; <label>:373		; preds = %371, %372
	add uint %236, 104		; <uint>:246 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %246		; <ubyte*>:515 [#uses=1]
	load ubyte* %515		; <ubyte>:807 [#uses=1]
	seteq ubyte %807, 0		; <bool>:373 [#uses=1]
	br bool %373, label %375, label %374

; <label>:374		; preds = %373, %374
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %245		; <ubyte*>:516 [#uses=2]
	load ubyte* %516		; <ubyte>:808 [#uses=1]
	add ubyte %808, 1		; <ubyte>:809 [#uses=1]
	store ubyte %809, ubyte* %516
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %246		; <ubyte*>:517 [#uses=2]
	load ubyte* %517		; <ubyte>:810 [#uses=2]
	add ubyte %810, 255		; <ubyte>:811 [#uses=1]
	store ubyte %811, ubyte* %517
	seteq ubyte %810, 1		; <bool>:374 [#uses=1]
	br bool %374, label %375, label %374

; <label>:375		; preds = %373, %374
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %244		; <ubyte*>:518 [#uses=1]
	load ubyte* %518		; <ubyte>:812 [#uses=1]
	seteq ubyte %812, 0		; <bool>:375 [#uses=1]
	br bool %375, label %377, label %376

; <label>:376		; preds = %375, %383
	phi uint [ %244, %375 ], [ %253, %383 ]		; <uint>:247 [#uses=8]
	add uint %247, 4294967292		; <uint>:248 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %248		; <ubyte*>:519 [#uses=1]
	load ubyte* %519		; <ubyte>:813 [#uses=1]
	seteq ubyte %813, 0		; <bool>:376 [#uses=1]
	br bool %376, label %379, label %378

; <label>:377		; preds = %375, %383
	phi uint [ %244, %375 ], [ %253, %383 ]		; <uint>:249 [#uses=5]
	add uint %249, 4294967294		; <uint>:250 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %250		; <ubyte*>:520 [#uses=1]
	load ubyte* %520		; <ubyte>:814 [#uses=1]
	seteq ubyte %814, 0		; <bool>:377 [#uses=1]
	br bool %377, label %385, label %384

; <label>:378		; preds = %376, %378
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %248		; <ubyte*>:521 [#uses=2]
	load ubyte* %521		; <ubyte>:815 [#uses=2]
	add ubyte %815, 255		; <ubyte>:816 [#uses=1]
	store ubyte %816, ubyte* %521
	seteq ubyte %815, 1		; <bool>:378 [#uses=1]
	br bool %378, label %379, label %378

; <label>:379		; preds = %376, %378
	add uint %247, 4294967286		; <uint>:251 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %251		; <ubyte*>:522 [#uses=1]
	load ubyte* %522		; <ubyte>:817 [#uses=1]
	seteq ubyte %817, 0		; <bool>:379 [#uses=1]
	br bool %379, label %381, label %380

; <label>:380		; preds = %379, %380
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %251		; <ubyte*>:523 [#uses=2]
	load ubyte* %523		; <ubyte>:818 [#uses=1]
	add ubyte %818, 255		; <ubyte>:819 [#uses=1]
	store ubyte %819, ubyte* %523
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %248		; <ubyte*>:524 [#uses=2]
	load ubyte* %524		; <ubyte>:820 [#uses=1]
	add ubyte %820, 1		; <ubyte>:821 [#uses=1]
	store ubyte %821, ubyte* %524
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %251		; <ubyte*>:525 [#uses=1]
	load ubyte* %525		; <ubyte>:822 [#uses=1]
	seteq ubyte %822, 0		; <bool>:380 [#uses=1]
	br bool %380, label %381, label %380

; <label>:381		; preds = %379, %380
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %247		; <ubyte*>:526 [#uses=1]
	load ubyte* %526		; <ubyte>:823 [#uses=1]
	seteq ubyte %823, 0		; <bool>:381 [#uses=1]
	br bool %381, label %383, label %382

; <label>:382		; preds = %381, %382
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %247		; <ubyte*>:527 [#uses=2]
	load ubyte* %527		; <ubyte>:824 [#uses=1]
	add ubyte %824, 255		; <ubyte>:825 [#uses=1]
	store ubyte %825, ubyte* %527
	add uint %247, 6		; <uint>:252 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %252		; <ubyte*>:528 [#uses=2]
	load ubyte* %528		; <ubyte>:826 [#uses=1]
	add ubyte %826, 1		; <ubyte>:827 [#uses=1]
	store ubyte %827, ubyte* %528
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %247		; <ubyte*>:529 [#uses=1]
	load ubyte* %529		; <ubyte>:828 [#uses=1]
	seteq ubyte %828, 0		; <bool>:382 [#uses=1]
	br bool %382, label %383, label %382

; <label>:383		; preds = %381, %382
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %247		; <ubyte*>:530 [#uses=2]
	load ubyte* %530		; <ubyte>:829 [#uses=1]
	add ubyte %829, 1		; <ubyte>:830 [#uses=1]
	store ubyte %830, ubyte* %530
	add uint %247, 6		; <uint>:253 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %253		; <ubyte*>:531 [#uses=2]
	load ubyte* %531		; <ubyte>:831 [#uses=2]
	add ubyte %831, 255		; <ubyte>:832 [#uses=1]
	store ubyte %832, ubyte* %531
	seteq ubyte %831, 1		; <bool>:383 [#uses=1]
	br bool %383, label %377, label %376

; <label>:384		; preds = %377, %384
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %250		; <ubyte*>:532 [#uses=2]
	load ubyte* %532		; <ubyte>:833 [#uses=2]
	add ubyte %833, 255		; <ubyte>:834 [#uses=1]
	store ubyte %834, ubyte* %532
	seteq ubyte %833, 1		; <bool>:384 [#uses=1]
	br bool %384, label %385, label %384

; <label>:385		; preds = %377, %384
	add uint %249, 4294967286		; <uint>:254 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %254		; <ubyte*>:533 [#uses=1]
	load ubyte* %533		; <ubyte>:835 [#uses=1]
	seteq ubyte %835, 0		; <bool>:385 [#uses=1]
	br bool %385, label %387, label %386

; <label>:386		; preds = %385, %386
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %254		; <ubyte*>:534 [#uses=2]
	load ubyte* %534		; <ubyte>:836 [#uses=1]
	add ubyte %836, 255		; <ubyte>:837 [#uses=1]
	store ubyte %837, ubyte* %534
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %250		; <ubyte*>:535 [#uses=2]
	load ubyte* %535		; <ubyte>:838 [#uses=1]
	add ubyte %838, 1		; <ubyte>:839 [#uses=1]
	store ubyte %839, ubyte* %535
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %254		; <ubyte*>:536 [#uses=1]
	load ubyte* %536		; <ubyte>:840 [#uses=1]
	seteq ubyte %840, 0		; <bool>:386 [#uses=1]
	br bool %386, label %387, label %386

; <label>:387		; preds = %385, %386
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %249		; <ubyte*>:537 [#uses=2]
	load ubyte* %537		; <ubyte>:841 [#uses=2]
	add ubyte %841, 1		; <ubyte>:842 [#uses=1]
	store ubyte %842, ubyte* %537
	seteq ubyte %841, 255		; <bool>:387 [#uses=1]
	br bool %387, label %389, label %388

; <label>:388		; preds = %387, %388
	phi uint [ %249, %387 ], [ %256, %388 ]		; <uint>:255 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %255		; <ubyte*>:538 [#uses=2]
	load ubyte* %538		; <ubyte>:843 [#uses=1]
	add ubyte %843, 255		; <ubyte>:844 [#uses=1]
	store ubyte %844, ubyte* %538
	add uint %255, 4294967290		; <uint>:256 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %256		; <ubyte*>:539 [#uses=1]
	load ubyte* %539		; <ubyte>:845 [#uses=1]
	seteq ubyte %845, 0		; <bool>:388 [#uses=1]
	br bool %388, label %389, label %388

; <label>:389		; preds = %387, %388
	phi uint [ %249, %387 ], [ %256, %388 ]		; <uint>:257 [#uses=7]
	add uint %257, 98		; <uint>:258 [#uses=6]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %258		; <ubyte*>:540 [#uses=1]
	load ubyte* %540		; <ubyte>:846 [#uses=1]
	seteq ubyte %846, 0		; <bool>:389 [#uses=1]
	br bool %389, label %391, label %390

; <label>:390		; preds = %389, %390
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %258		; <ubyte*>:541 [#uses=2]
	load ubyte* %541		; <ubyte>:847 [#uses=2]
	add ubyte %847, 255		; <ubyte>:848 [#uses=1]
	store ubyte %848, ubyte* %541
	seteq ubyte %847, 1		; <bool>:390 [#uses=1]
	br bool %390, label %391, label %390

; <label>:391		; preds = %389, %390
	add uint %257, 94		; <uint>:259 [#uses=7]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %259		; <ubyte*>:542 [#uses=1]
	load ubyte* %542		; <ubyte>:849 [#uses=1]
	seteq ubyte %849, 0		; <bool>:391 [#uses=1]
	br bool %391, label %393, label %392

; <label>:392		; preds = %391, %392
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %259		; <ubyte*>:543 [#uses=2]
	load ubyte* %543		; <ubyte>:850 [#uses=1]
	add ubyte %850, 255		; <ubyte>:851 [#uses=1]
	store ubyte %851, ubyte* %543
	add uint %257, 95		; <uint>:260 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %260		; <ubyte*>:544 [#uses=2]
	load ubyte* %544		; <ubyte>:852 [#uses=1]
	add ubyte %852, 1		; <ubyte>:853 [#uses=1]
	store ubyte %853, ubyte* %544
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %258		; <ubyte*>:545 [#uses=2]
	load ubyte* %545		; <ubyte>:854 [#uses=1]
	add ubyte %854, 1		; <ubyte>:855 [#uses=1]
	store ubyte %855, ubyte* %545
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %259		; <ubyte*>:546 [#uses=1]
	load ubyte* %546		; <ubyte>:856 [#uses=1]
	seteq ubyte %856, 0		; <bool>:392 [#uses=1]
	br bool %392, label %393, label %392

; <label>:393		; preds = %391, %392
	add uint %257, 95		; <uint>:261 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %261		; <ubyte*>:547 [#uses=1]
	load ubyte* %547		; <ubyte>:857 [#uses=1]
	seteq ubyte %857, 0		; <bool>:393 [#uses=1]
	br bool %393, label %395, label %394

; <label>:394		; preds = %393, %394
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %259		; <ubyte*>:548 [#uses=2]
	load ubyte* %548		; <ubyte>:858 [#uses=1]
	add ubyte %858, 1		; <ubyte>:859 [#uses=1]
	store ubyte %859, ubyte* %548
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %261		; <ubyte*>:549 [#uses=2]
	load ubyte* %549		; <ubyte>:860 [#uses=2]
	add ubyte %860, 255		; <ubyte>:861 [#uses=1]
	store ubyte %861, ubyte* %549
	seteq ubyte %860, 1		; <bool>:394 [#uses=1]
	br bool %394, label %395, label %394

; <label>:395		; preds = %393, %394
	add uint %257, 100		; <uint>:262 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %262		; <ubyte*>:550 [#uses=2]
	load ubyte* %550		; <ubyte>:862 [#uses=2]
	add ubyte %862, 1		; <ubyte>:863 [#uses=1]
	store ubyte %863, ubyte* %550
	seteq ubyte %862, 255		; <bool>:395 [#uses=1]
	br bool %395, label %397, label %396

; <label>:396		; preds = %395, %396
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %258		; <ubyte*>:551 [#uses=2]
	load ubyte* %551		; <ubyte>:864 [#uses=1]
	add ubyte %864, 1		; <ubyte>:865 [#uses=1]
	store ubyte %865, ubyte* %551
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %262		; <ubyte*>:552 [#uses=2]
	load ubyte* %552		; <ubyte>:866 [#uses=2]
	add ubyte %866, 255		; <ubyte>:867 [#uses=1]
	store ubyte %867, ubyte* %552
	seteq ubyte %866, 1		; <bool>:396 [#uses=1]
	br bool %396, label %397, label %396

; <label>:397		; preds = %395, %396
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %259		; <ubyte*>:553 [#uses=1]
	load ubyte* %553		; <ubyte>:868 [#uses=1]
	seteq ubyte %868, 0		; <bool>:397 [#uses=1]
	br bool %397, label %399, label %398

; <label>:398		; preds = %397, %398
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %259		; <ubyte*>:554 [#uses=2]
	load ubyte* %554		; <ubyte>:869 [#uses=2]
	add ubyte %869, 255		; <ubyte>:870 [#uses=1]
	store ubyte %870, ubyte* %554
	seteq ubyte %869, 1		; <bool>:398 [#uses=1]
	br bool %398, label %399, label %398

; <label>:399		; preds = %397, %398
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %258		; <ubyte*>:555 [#uses=1]
	load ubyte* %555		; <ubyte>:871 [#uses=1]
	seteq ubyte %871, 0		; <bool>:399 [#uses=1]
	br bool %399, label %401, label %400

; <label>:400		; preds = %399, %400
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %259		; <ubyte*>:556 [#uses=2]
	load ubyte* %556		; <ubyte>:872 [#uses=1]
	add ubyte %872, 1		; <ubyte>:873 [#uses=1]
	store ubyte %873, ubyte* %556
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %258		; <ubyte*>:557 [#uses=2]
	load ubyte* %557		; <ubyte>:874 [#uses=2]
	add ubyte %874, 255		; <ubyte>:875 [#uses=1]
	store ubyte %875, ubyte* %557
	seteq ubyte %874, 1		; <bool>:400 [#uses=1]
	br bool %400, label %401, label %400

; <label>:401		; preds = %399, %400
	add uint %257, 4294967289		; <uint>:263 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %263		; <ubyte*>:558 [#uses=2]
	load ubyte* %558		; <ubyte>:876 [#uses=1]
	add ubyte %876, 3		; <ubyte>:877 [#uses=1]
	store ubyte %877, ubyte* %558
	add uint %257, 4294967291		; <uint>:264 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %264		; <ubyte*>:559 [#uses=1]
	load ubyte* %559		; <ubyte>:878 [#uses=1]
	seteq ubyte %878, 0		; <bool>:401 [#uses=1]
	br bool %401, label %361, label %360

; <label>:402		; preds = %21, %455
	phi uint [ %31, %21 ], [ %289, %455 ]		; <uint>:265 [#uses=23]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %265		; <ubyte*>:560 [#uses=2]
	load ubyte* %560		; <ubyte>:879 [#uses=1]
	add ubyte %879, 255		; <ubyte>:880 [#uses=1]
	store ubyte %880, ubyte* %560
	add uint %265, 104		; <uint>:266 [#uses=17]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %266		; <ubyte*>:561 [#uses=1]
	load ubyte* %561		; <ubyte>:881 [#uses=1]
	seteq ubyte %881, 0		; <bool>:402 [#uses=1]
	br bool %402, label %405, label %404

; <label>:403		; preds = %21, %455
	phi uint [ %31, %21 ], [ %289, %455 ]		; <uint>:267 [#uses=1]
	add uint %267, 4294967295		; <uint>:268 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %268		; <ubyte*>:562 [#uses=1]
	load ubyte* %562		; <ubyte>:882 [#uses=1]
	seteq ubyte %882, 0		; <bool>:403 [#uses=1]
	br bool %403, label %19, label %18

; <label>:404		; preds = %402, %404
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %266		; <ubyte*>:563 [#uses=2]
	load ubyte* %563		; <ubyte>:883 [#uses=2]
	add ubyte %883, 255		; <ubyte>:884 [#uses=1]
	store ubyte %884, ubyte* %563
	seteq ubyte %883, 1		; <bool>:404 [#uses=1]
	br bool %404, label %405, label %404

; <label>:405		; preds = %402, %404
	add uint %265, 100		; <uint>:269 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %269		; <ubyte*>:564 [#uses=1]
	load ubyte* %564		; <ubyte>:885 [#uses=1]
	seteq ubyte %885, 0		; <bool>:405 [#uses=1]
	br bool %405, label %407, label %406

; <label>:406		; preds = %405, %406
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %269		; <ubyte*>:565 [#uses=2]
	load ubyte* %565		; <ubyte>:886 [#uses=1]
	add ubyte %886, 255		; <ubyte>:887 [#uses=1]
	store ubyte %887, ubyte* %565
	add uint %265, 101		; <uint>:270 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %270		; <ubyte*>:566 [#uses=2]
	load ubyte* %566		; <ubyte>:888 [#uses=1]
	add ubyte %888, 1		; <ubyte>:889 [#uses=1]
	store ubyte %889, ubyte* %566
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %266		; <ubyte*>:567 [#uses=2]
	load ubyte* %567		; <ubyte>:890 [#uses=1]
	add ubyte %890, 1		; <ubyte>:891 [#uses=1]
	store ubyte %891, ubyte* %567
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %269		; <ubyte*>:568 [#uses=1]
	load ubyte* %568		; <ubyte>:892 [#uses=1]
	seteq ubyte %892, 0		; <bool>:406 [#uses=1]
	br bool %406, label %407, label %406

; <label>:407		; preds = %405, %406
	add uint %265, 101		; <uint>:271 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %271		; <ubyte*>:569 [#uses=1]
	load ubyte* %569		; <ubyte>:893 [#uses=1]
	seteq ubyte %893, 0		; <bool>:407 [#uses=1]
	br bool %407, label %409, label %408

; <label>:408		; preds = %407, %408
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %269		; <ubyte*>:570 [#uses=2]
	load ubyte* %570		; <ubyte>:894 [#uses=1]
	add ubyte %894, 1		; <ubyte>:895 [#uses=1]
	store ubyte %895, ubyte* %570
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %271		; <ubyte*>:571 [#uses=2]
	load ubyte* %571		; <ubyte>:896 [#uses=2]
	add ubyte %896, 255		; <ubyte>:897 [#uses=1]
	store ubyte %897, ubyte* %571
	seteq ubyte %896, 1		; <bool>:408 [#uses=1]
	br bool %408, label %409, label %408

; <label>:409		; preds = %407, %408
	add uint %265, 106		; <uint>:272 [#uses=12]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %272		; <ubyte*>:572 [#uses=1]
	load ubyte* %572		; <ubyte>:898 [#uses=1]
	seteq ubyte %898, 0		; <bool>:409 [#uses=1]
	br bool %409, label %411, label %410

; <label>:410		; preds = %409, %410
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %272		; <ubyte*>:573 [#uses=2]
	load ubyte* %573		; <ubyte>:899 [#uses=2]
	add ubyte %899, 255		; <ubyte>:900 [#uses=1]
	store ubyte %900, ubyte* %573
	seteq ubyte %899, 1		; <bool>:410 [#uses=1]
	br bool %410, label %411, label %410

; <label>:411		; preds = %409, %410
	add uint %265, 98		; <uint>:273 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %273		; <ubyte*>:574 [#uses=1]
	load ubyte* %574		; <ubyte>:901 [#uses=1]
	seteq ubyte %901, 0		; <bool>:411 [#uses=1]
	br bool %411, label %413, label %412

; <label>:412		; preds = %411, %412
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %273		; <ubyte*>:575 [#uses=2]
	load ubyte* %575		; <ubyte>:902 [#uses=1]
	add ubyte %902, 255		; <ubyte>:903 [#uses=1]
	store ubyte %903, ubyte* %575
	add uint %265, 99		; <uint>:274 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %274		; <ubyte*>:576 [#uses=2]
	load ubyte* %576		; <ubyte>:904 [#uses=1]
	add ubyte %904, 1		; <ubyte>:905 [#uses=1]
	store ubyte %905, ubyte* %576
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %272		; <ubyte*>:577 [#uses=2]
	load ubyte* %577		; <ubyte>:906 [#uses=1]
	add ubyte %906, 1		; <ubyte>:907 [#uses=1]
	store ubyte %907, ubyte* %577
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %273		; <ubyte*>:578 [#uses=1]
	load ubyte* %578		; <ubyte>:908 [#uses=1]
	seteq ubyte %908, 0		; <bool>:412 [#uses=1]
	br bool %412, label %413, label %412

; <label>:413		; preds = %411, %412
	add uint %265, 99		; <uint>:275 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %275		; <ubyte*>:579 [#uses=1]
	load ubyte* %579		; <ubyte>:909 [#uses=1]
	seteq ubyte %909, 0		; <bool>:413 [#uses=1]
	br bool %413, label %415, label %414

; <label>:414		; preds = %413, %414
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %273		; <ubyte*>:580 [#uses=2]
	load ubyte* %580		; <ubyte>:910 [#uses=1]
	add ubyte %910, 1		; <ubyte>:911 [#uses=1]
	store ubyte %911, ubyte* %580
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %275		; <ubyte*>:581 [#uses=2]
	load ubyte* %581		; <ubyte>:912 [#uses=2]
	add ubyte %912, 255		; <ubyte>:913 [#uses=1]
	store ubyte %913, ubyte* %581
	seteq ubyte %912, 1		; <bool>:414 [#uses=1]
	br bool %414, label %415, label %414

; <label>:415		; preds = %413, %414
	add uint %265, 108		; <uint>:276 [#uses=9]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %276		; <ubyte*>:582 [#uses=2]
	load ubyte* %582		; <ubyte>:914 [#uses=2]
	add ubyte %914, 1		; <ubyte>:915 [#uses=1]
	store ubyte %915, ubyte* %582
	seteq ubyte %914, 255		; <bool>:415 [#uses=1]
	br bool %415, label %417, label %416

; <label>:416		; preds = %415, %439
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %276		; <ubyte*>:583 [#uses=2]
	load ubyte* %583		; <ubyte>:916 [#uses=1]
	add ubyte %916, 1		; <ubyte>:917 [#uses=1]
	store ubyte %917, ubyte* %583
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %266		; <ubyte*>:584 [#uses=1]
	load ubyte* %584		; <ubyte>:918 [#uses=1]
	seteq ubyte %918, 0		; <bool>:416 [#uses=1]
	br bool %416, label %419, label %418

; <label>:417		; preds = %415, %439
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %266		; <ubyte*>:585 [#uses=1]
	load ubyte* %585		; <ubyte>:919 [#uses=1]
	seteq ubyte %919, 0		; <bool>:417 [#uses=1]
	br bool %417, label %441, label %440

; <label>:418		; preds = %416, %418
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %266		; <ubyte*>:586 [#uses=2]
	load ubyte* %586		; <ubyte>:920 [#uses=1]
	add ubyte %920, 255		; <ubyte>:921 [#uses=1]
	store ubyte %921, ubyte* %586
	add uint %265, 105		; <uint>:277 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %277		; <ubyte*>:587 [#uses=2]
	load ubyte* %587		; <ubyte>:922 [#uses=1]
	add ubyte %922, 1		; <ubyte>:923 [#uses=1]
	store ubyte %923, ubyte* %587
	add uint %265, 109		; <uint>:278 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %278		; <ubyte*>:588 [#uses=2]
	load ubyte* %588		; <ubyte>:924 [#uses=1]
	add ubyte %924, 1		; <ubyte>:925 [#uses=1]
	store ubyte %925, ubyte* %588
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %266		; <ubyte*>:589 [#uses=1]
	load ubyte* %589		; <ubyte>:926 [#uses=1]
	seteq ubyte %926, 0		; <bool>:418 [#uses=1]
	br bool %418, label %419, label %418

; <label>:419		; preds = %416, %418
	add uint %265, 105		; <uint>:279 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %279		; <ubyte*>:590 [#uses=1]
	load ubyte* %590		; <ubyte>:927 [#uses=1]
	seteq ubyte %927, 0		; <bool>:419 [#uses=1]
	br bool %419, label %421, label %420

; <label>:420		; preds = %419, %420
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %266		; <ubyte*>:591 [#uses=2]
	load ubyte* %591		; <ubyte>:928 [#uses=1]
	add ubyte %928, 1		; <ubyte>:929 [#uses=1]
	store ubyte %929, ubyte* %591
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %279		; <ubyte*>:592 [#uses=2]
	load ubyte* %592		; <ubyte>:930 [#uses=2]
	add ubyte %930, 255		; <ubyte>:931 [#uses=1]
	store ubyte %931, ubyte* %592
	seteq ubyte %930, 1		; <bool>:420 [#uses=1]
	br bool %420, label %421, label %420

; <label>:421		; preds = %419, %420
	add uint %265, 109		; <uint>:280 [#uses=13]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %280		; <ubyte*>:593 [#uses=1]
	load ubyte* %593		; <ubyte>:932 [#uses=1]
	seteq ubyte %932, 0		; <bool>:421 [#uses=1]
	br bool %421, label %423, label %422

; <label>:422		; preds = %421, %425
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %280		; <ubyte*>:594 [#uses=1]
	load ubyte* %594		; <ubyte>:933 [#uses=1]
	seteq ubyte %933, 0		; <bool>:422 [#uses=1]
	br bool %422, label %425, label %424

; <label>:423		; preds = %421, %425
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %272		; <ubyte*>:595 [#uses=1]
	load ubyte* %595		; <ubyte>:934 [#uses=1]
	seteq ubyte %934, 0		; <bool>:423 [#uses=1]
	br bool %423, label %427, label %426

; <label>:424		; preds = %422, %424
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %280		; <ubyte*>:596 [#uses=2]
	load ubyte* %596		; <ubyte>:935 [#uses=2]
	add ubyte %935, 255		; <ubyte>:936 [#uses=1]
	store ubyte %936, ubyte* %596
	seteq ubyte %935, 1		; <bool>:424 [#uses=1]
	br bool %424, label %425, label %424

; <label>:425		; preds = %422, %424
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %276		; <ubyte*>:597 [#uses=2]
	load ubyte* %597		; <ubyte>:937 [#uses=1]
	add ubyte %937, 255		; <ubyte>:938 [#uses=1]
	store ubyte %938, ubyte* %597
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %280		; <ubyte*>:598 [#uses=1]
	load ubyte* %598		; <ubyte>:939 [#uses=1]
	seteq ubyte %939, 0		; <bool>:425 [#uses=1]
	br bool %425, label %423, label %422

; <label>:426		; preds = %423, %426
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %272		; <ubyte*>:599 [#uses=2]
	load ubyte* %599		; <ubyte>:940 [#uses=1]
	add ubyte %940, 255		; <ubyte>:941 [#uses=1]
	store ubyte %941, ubyte* %599
	add uint %265, 107		; <uint>:281 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %281		; <ubyte*>:600 [#uses=2]
	load ubyte* %600		; <ubyte>:942 [#uses=1]
	add ubyte %942, 1		; <ubyte>:943 [#uses=1]
	store ubyte %943, ubyte* %600
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %280		; <ubyte*>:601 [#uses=2]
	load ubyte* %601		; <ubyte>:944 [#uses=1]
	add ubyte %944, 1		; <ubyte>:945 [#uses=1]
	store ubyte %945, ubyte* %601
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %272		; <ubyte*>:602 [#uses=1]
	load ubyte* %602		; <ubyte>:946 [#uses=1]
	seteq ubyte %946, 0		; <bool>:426 [#uses=1]
	br bool %426, label %427, label %426

; <label>:427		; preds = %423, %426
	add uint %265, 107		; <uint>:282 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %282		; <ubyte*>:603 [#uses=1]
	load ubyte* %603		; <ubyte>:947 [#uses=1]
	seteq ubyte %947, 0		; <bool>:427 [#uses=1]
	br bool %427, label %429, label %428

; <label>:428		; preds = %427, %428
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %272		; <ubyte*>:604 [#uses=2]
	load ubyte* %604		; <ubyte>:948 [#uses=1]
	add ubyte %948, 1		; <ubyte>:949 [#uses=1]
	store ubyte %949, ubyte* %604
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %282		; <ubyte*>:605 [#uses=2]
	load ubyte* %605		; <ubyte>:950 [#uses=2]
	add ubyte %950, 255		; <ubyte>:951 [#uses=1]
	store ubyte %951, ubyte* %605
	seteq ubyte %950, 1		; <bool>:428 [#uses=1]
	br bool %428, label %429, label %428

; <label>:429		; preds = %427, %428
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %280		; <ubyte*>:606 [#uses=1]
	load ubyte* %606		; <ubyte>:952 [#uses=1]
	seteq ubyte %952, 0		; <bool>:429 [#uses=1]
	br bool %429, label %431, label %430

; <label>:430		; preds = %429, %433
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %280		; <ubyte*>:607 [#uses=1]
	load ubyte* %607		; <ubyte>:953 [#uses=1]
	seteq ubyte %953, 0		; <bool>:430 [#uses=1]
	br bool %430, label %433, label %432

; <label>:431		; preds = %429, %433
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %280		; <ubyte*>:608 [#uses=2]
	load ubyte* %608		; <ubyte>:954 [#uses=1]
	add ubyte %954, 1		; <ubyte>:955 [#uses=1]
	store ubyte %955, ubyte* %608
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %276		; <ubyte*>:609 [#uses=1]
	load ubyte* %609		; <ubyte>:956 [#uses=1]
	seteq ubyte %956, 0		; <bool>:431 [#uses=1]
	br bool %431, label %435, label %434

; <label>:432		; preds = %430, %432
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %280		; <ubyte*>:610 [#uses=2]
	load ubyte* %610		; <ubyte>:957 [#uses=2]
	add ubyte %957, 255		; <ubyte>:958 [#uses=1]
	store ubyte %958, ubyte* %610
	seteq ubyte %957, 1		; <bool>:432 [#uses=1]
	br bool %432, label %433, label %432

; <label>:433		; preds = %430, %432
	add uint %265, 108		; <uint>:283 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %283		; <ubyte*>:611 [#uses=2]
	load ubyte* %611		; <ubyte>:959 [#uses=1]
	add ubyte %959, 255		; <ubyte>:960 [#uses=1]
	store ubyte %960, ubyte* %611
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %280		; <ubyte*>:612 [#uses=1]
	load ubyte* %612		; <ubyte>:961 [#uses=1]
	seteq ubyte %961, 0		; <bool>:433 [#uses=1]
	br bool %433, label %431, label %430

; <label>:434		; preds = %431, %437
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %276		; <ubyte*>:613 [#uses=1]
	load ubyte* %613		; <ubyte>:962 [#uses=1]
	seteq ubyte %962, 0		; <bool>:434 [#uses=1]
	br bool %434, label %437, label %436

; <label>:435		; preds = %431, %437
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %280		; <ubyte*>:614 [#uses=1]
	load ubyte* %614		; <ubyte>:963 [#uses=1]
	seteq ubyte %963, 0		; <bool>:435 [#uses=1]
	br bool %435, label %439, label %438

; <label>:436		; preds = %434, %436
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %276		; <ubyte*>:615 [#uses=2]
	load ubyte* %615		; <ubyte>:964 [#uses=2]
	add ubyte %964, 255		; <ubyte>:965 [#uses=1]
	store ubyte %965, ubyte* %615
	seteq ubyte %964, 1		; <bool>:436 [#uses=1]
	br bool %436, label %437, label %436

; <label>:437		; preds = %434, %436
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %280		; <ubyte*>:616 [#uses=2]
	load ubyte* %616		; <ubyte>:966 [#uses=1]
	add ubyte %966, 255		; <ubyte>:967 [#uses=1]
	store ubyte %967, ubyte* %616
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %276		; <ubyte*>:617 [#uses=1]
	load ubyte* %617		; <ubyte>:968 [#uses=1]
	seteq ubyte %968, 0		; <bool>:437 [#uses=1]
	br bool %437, label %435, label %434

; <label>:438		; preds = %435, %438
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %266		; <ubyte*>:618 [#uses=2]
	load ubyte* %618		; <ubyte>:969 [#uses=1]
	add ubyte %969, 255		; <ubyte>:970 [#uses=1]
	store ubyte %970, ubyte* %618
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %272		; <ubyte*>:619 [#uses=2]
	load ubyte* %619		; <ubyte>:971 [#uses=1]
	add ubyte %971, 255		; <ubyte>:972 [#uses=1]
	store ubyte %972, ubyte* %619
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %276		; <ubyte*>:620 [#uses=2]
	load ubyte* %620		; <ubyte>:973 [#uses=1]
	add ubyte %973, 1		; <ubyte>:974 [#uses=1]
	store ubyte %974, ubyte* %620
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %280		; <ubyte*>:621 [#uses=2]
	load ubyte* %621		; <ubyte>:975 [#uses=2]
	add ubyte %975, 255		; <ubyte>:976 [#uses=1]
	store ubyte %976, ubyte* %621
	seteq ubyte %975, 1		; <bool>:438 [#uses=1]
	br bool %438, label %439, label %438

; <label>:439		; preds = %435, %438
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %276		; <ubyte*>:622 [#uses=1]
	load ubyte* %622		; <ubyte>:977 [#uses=1]
	seteq ubyte %977, 0		; <bool>:439 [#uses=1]
	br bool %439, label %417, label %416

; <label>:440		; preds = %417, %443
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %266		; <ubyte*>:623 [#uses=1]
	load ubyte* %623		; <ubyte>:978 [#uses=1]
	seteq ubyte %978, 0		; <bool>:440 [#uses=1]
	br bool %440, label %443, label %442

; <label>:441		; preds = %417, %443
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %272		; <ubyte*>:624 [#uses=1]
	load ubyte* %624		; <ubyte>:979 [#uses=1]
	seteq ubyte %979, 0		; <bool>:441 [#uses=1]
	br bool %441, label %445, label %444

; <label>:442		; preds = %440, %442
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %266		; <ubyte*>:625 [#uses=2]
	load ubyte* %625		; <ubyte>:980 [#uses=2]
	add ubyte %980, 255		; <ubyte>:981 [#uses=1]
	store ubyte %981, ubyte* %625
	seteq ubyte %980, 1		; <bool>:442 [#uses=1]
	br bool %442, label %443, label %442

; <label>:443		; preds = %440, %442
	add uint %265, 105		; <uint>:284 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %284		; <ubyte*>:626 [#uses=2]
	load ubyte* %626		; <ubyte>:982 [#uses=1]
	add ubyte %982, 1		; <ubyte>:983 [#uses=1]
	store ubyte %983, ubyte* %626
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %266		; <ubyte*>:627 [#uses=1]
	load ubyte* %627		; <ubyte>:984 [#uses=1]
	seteq ubyte %984, 0		; <bool>:443 [#uses=1]
	br bool %443, label %441, label %440

; <label>:444		; preds = %441, %447
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %272		; <ubyte*>:628 [#uses=1]
	load ubyte* %628		; <ubyte>:985 [#uses=1]
	seteq ubyte %985, 0		; <bool>:444 [#uses=1]
	br bool %444, label %447, label %446

; <label>:445		; preds = %441, %447
	add uint %265, 105		; <uint>:285 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %285		; <ubyte*>:629 [#uses=1]
	load ubyte* %629		; <ubyte>:986 [#uses=1]
	seteq ubyte %986, 0		; <bool>:445 [#uses=1]
	br bool %445, label %449, label %448

; <label>:446		; preds = %444, %446
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %272		; <ubyte*>:630 [#uses=2]
	load ubyte* %630		; <ubyte>:987 [#uses=2]
	add ubyte %987, 255		; <ubyte>:988 [#uses=1]
	store ubyte %988, ubyte* %630
	seteq ubyte %987, 1		; <bool>:446 [#uses=1]
	br bool %446, label %447, label %446

; <label>:447		; preds = %444, %446
	add uint %265, 107		; <uint>:286 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %286		; <ubyte*>:631 [#uses=2]
	load ubyte* %631		; <ubyte>:989 [#uses=1]
	add ubyte %989, 1		; <ubyte>:990 [#uses=1]
	store ubyte %990, ubyte* %631
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %272		; <ubyte*>:632 [#uses=1]
	load ubyte* %632		; <ubyte>:991 [#uses=1]
	seteq ubyte %991, 0		; <bool>:447 [#uses=1]
	br bool %447, label %445, label %444

; <label>:448		; preds = %445, %448
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %285		; <ubyte*>:633 [#uses=2]
	load ubyte* %633		; <ubyte>:992 [#uses=2]
	add ubyte %992, 255		; <ubyte>:993 [#uses=1]
	store ubyte %993, ubyte* %633
	seteq ubyte %992, 1		; <bool>:448 [#uses=1]
	br bool %448, label %449, label %448

; <label>:449		; preds = %445, %448
	add uint %265, 107		; <uint>:287 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %287		; <ubyte*>:634 [#uses=1]
	load ubyte* %634		; <ubyte>:994 [#uses=1]
	seteq ubyte %994, 0		; <bool>:449 [#uses=1]
	br bool %449, label %451, label %450

; <label>:450		; preds = %449, %453
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %287		; <ubyte*>:635 [#uses=1]
	load ubyte* %635		; <ubyte>:995 [#uses=1]
	seteq ubyte %995, 0		; <bool>:450 [#uses=1]
	br bool %450, label %453, label %452

; <label>:451		; preds = %449, %453
	add uint %265, 4294967295		; <uint>:288 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %288		; <ubyte*>:636 [#uses=2]
	load ubyte* %636		; <ubyte>:996 [#uses=1]
	add ubyte %996, 2		; <ubyte>:997 [#uses=1]
	store ubyte %997, ubyte* %636
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %266		; <ubyte*>:637 [#uses=1]
	load ubyte* %637		; <ubyte>:998 [#uses=1]
	seteq ubyte %998, 0		; <bool>:451 [#uses=1]
	br bool %451, label %455, label %454

; <label>:452		; preds = %450, %452
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %287		; <ubyte*>:638 [#uses=2]
	load ubyte* %638		; <ubyte>:999 [#uses=2]
	add ubyte %999, 255		; <ubyte>:1000 [#uses=1]
	store ubyte %1000, ubyte* %638
	seteq ubyte %999, 1		; <bool>:452 [#uses=1]
	br bool %452, label %453, label %452

; <label>:453		; preds = %450, %452
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %266		; <ubyte*>:639 [#uses=2]
	load ubyte* %639		; <ubyte>:1001 [#uses=1]
	add ubyte %1001, 1		; <ubyte>:1002 [#uses=1]
	store ubyte %1002, ubyte* %639
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %287		; <ubyte*>:640 [#uses=1]
	load ubyte* %640		; <ubyte>:1003 [#uses=1]
	seteq ubyte %1003, 0		; <bool>:453 [#uses=1]
	br bool %453, label %451, label %450

; <label>:454		; preds = %451, %457
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %266		; <ubyte*>:641 [#uses=1]
	load ubyte* %641		; <ubyte>:1004 [#uses=1]
	seteq ubyte %1004, 0		; <bool>:454 [#uses=1]
	br bool %454, label %457, label %456

; <label>:455		; preds = %451, %457
	add uint %265, 1		; <uint>:289 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %289		; <ubyte*>:642 [#uses=1]
	load ubyte* %642		; <ubyte>:1005 [#uses=1]
	seteq ubyte %1005, 0		; <bool>:455 [#uses=1]
	br bool %455, label %403, label %402

; <label>:456		; preds = %454, %456
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %266		; <ubyte*>:643 [#uses=2]
	load ubyte* %643		; <ubyte>:1006 [#uses=2]
	add ubyte %1006, 255		; <ubyte>:1007 [#uses=1]
	store ubyte %1007, ubyte* %643
	seteq ubyte %1006, 1		; <bool>:456 [#uses=1]
	br bool %456, label %457, label %456

; <label>:457		; preds = %454, %456
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %288		; <ubyte*>:644 [#uses=2]
	load ubyte* %644		; <ubyte>:1008 [#uses=1]
	add ubyte %1008, 2		; <ubyte>:1009 [#uses=1]
	store ubyte %1009, ubyte* %644
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %266		; <ubyte*>:645 [#uses=1]
	load ubyte* %645		; <ubyte>:1010 [#uses=1]
	seteq ubyte %1010, 0		; <bool>:457 [#uses=1]
	br bool %457, label %455, label %454

; <label>:458		; preds = %19, %555
	phi uint [ %28, %19 ], [ %361, %555 ]		; <uint>:290 [#uses=70]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %290		; <ubyte*>:646 [#uses=2]
	load ubyte* %646		; <ubyte>:1011 [#uses=1]
	add ubyte %1011, 255		; <ubyte>:1012 [#uses=1]
	store ubyte %1012, ubyte* %646
	add uint %290, 104		; <uint>:291 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %291		; <ubyte*>:647 [#uses=1]
	load ubyte* %647		; <ubyte>:1013 [#uses=1]
	seteq ubyte %1013, 0		; <bool>:458 [#uses=1]
	br bool %458, label %461, label %460

; <label>:459		; preds = %19, %555
	phi uint [ %28, %19 ], [ %361, %555 ]		; <uint>:292 [#uses=1]
	add uint %292, 4294967295		; <uint>:293 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %293		; <ubyte*>:648 [#uses=1]
	load ubyte* %648		; <ubyte>:1014 [#uses=1]
	seteq ubyte %1014, 0		; <bool>:459 [#uses=1]
	br bool %459, label %17, label %16

; <label>:460		; preds = %458, %460
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %291		; <ubyte*>:649 [#uses=2]
	load ubyte* %649		; <ubyte>:1015 [#uses=2]
	add ubyte %1015, 255		; <ubyte>:1016 [#uses=1]
	store ubyte %1016, ubyte* %649
	seteq ubyte %1015, 1		; <bool>:460 [#uses=1]
	br bool %460, label %461, label %460

; <label>:461		; preds = %458, %460
	add uint %290, 98		; <uint>:294 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %294		; <ubyte*>:650 [#uses=1]
	load ubyte* %650		; <ubyte>:1017 [#uses=1]
	seteq ubyte %1017, 0		; <bool>:461 [#uses=1]
	br bool %461, label %463, label %462

; <label>:462		; preds = %461, %462
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %294		; <ubyte*>:651 [#uses=2]
	load ubyte* %651		; <ubyte>:1018 [#uses=1]
	add ubyte %1018, 255		; <ubyte>:1019 [#uses=1]
	store ubyte %1019, ubyte* %651
	add uint %290, 99		; <uint>:295 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %295		; <ubyte*>:652 [#uses=2]
	load ubyte* %652		; <ubyte>:1020 [#uses=1]
	add ubyte %1020, 1		; <ubyte>:1021 [#uses=1]
	store ubyte %1021, ubyte* %652
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %291		; <ubyte*>:653 [#uses=2]
	load ubyte* %653		; <ubyte>:1022 [#uses=1]
	add ubyte %1022, 1		; <ubyte>:1023 [#uses=1]
	store ubyte %1023, ubyte* %653
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %294		; <ubyte*>:654 [#uses=1]
	load ubyte* %654		; <ubyte>:1024 [#uses=1]
	seteq ubyte %1024, 0		; <bool>:462 [#uses=1]
	br bool %462, label %463, label %462

; <label>:463		; preds = %461, %462
	add uint %290, 99		; <uint>:296 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %296		; <ubyte*>:655 [#uses=1]
	load ubyte* %655		; <ubyte>:1025 [#uses=1]
	seteq ubyte %1025, 0		; <bool>:463 [#uses=1]
	br bool %463, label %465, label %464

; <label>:464		; preds = %463, %464
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %294		; <ubyte*>:656 [#uses=2]
	load ubyte* %656		; <ubyte>:1026 [#uses=1]
	add ubyte %1026, 1		; <ubyte>:1027 [#uses=1]
	store ubyte %1027, ubyte* %656
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %296		; <ubyte*>:657 [#uses=2]
	load ubyte* %657		; <ubyte>:1028 [#uses=2]
	add ubyte %1028, 255		; <ubyte>:1029 [#uses=1]
	store ubyte %1029, ubyte* %657
	seteq ubyte %1028, 1		; <bool>:464 [#uses=1]
	br bool %464, label %465, label %464

; <label>:465		; preds = %463, %464
	add uint %290, 116		; <uint>:297 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %297		; <ubyte*>:658 [#uses=1]
	load ubyte* %658		; <ubyte>:1030 [#uses=1]
	seteq ubyte %1030, 0		; <bool>:465 [#uses=1]
	br bool %465, label %467, label %466

; <label>:466		; preds = %465, %466
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %297		; <ubyte*>:659 [#uses=2]
	load ubyte* %659		; <ubyte>:1031 [#uses=2]
	add ubyte %1031, 255		; <ubyte>:1032 [#uses=1]
	store ubyte %1032, ubyte* %659
	seteq ubyte %1031, 1		; <bool>:466 [#uses=1]
	br bool %466, label %467, label %466

; <label>:467		; preds = %465, %466
	add uint %290, 10		; <uint>:298 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %298		; <ubyte*>:660 [#uses=1]
	load ubyte* %660		; <ubyte>:1033 [#uses=1]
	seteq ubyte %1033, 0		; <bool>:467 [#uses=1]
	br bool %467, label %469, label %468

; <label>:468		; preds = %467, %468
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %298		; <ubyte*>:661 [#uses=2]
	load ubyte* %661		; <ubyte>:1034 [#uses=1]
	add ubyte %1034, 255		; <ubyte>:1035 [#uses=1]
	store ubyte %1035, ubyte* %661
	add uint %290, 11		; <uint>:299 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %299		; <ubyte*>:662 [#uses=2]
	load ubyte* %662		; <ubyte>:1036 [#uses=1]
	add ubyte %1036, 1		; <ubyte>:1037 [#uses=1]
	store ubyte %1037, ubyte* %662
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %297		; <ubyte*>:663 [#uses=2]
	load ubyte* %663		; <ubyte>:1038 [#uses=1]
	add ubyte %1038, 1		; <ubyte>:1039 [#uses=1]
	store ubyte %1039, ubyte* %663
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %298		; <ubyte*>:664 [#uses=1]
	load ubyte* %664		; <ubyte>:1040 [#uses=1]
	seteq ubyte %1040, 0		; <bool>:468 [#uses=1]
	br bool %468, label %469, label %468

; <label>:469		; preds = %467, %468
	add uint %290, 11		; <uint>:300 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %300		; <ubyte*>:665 [#uses=1]
	load ubyte* %665		; <ubyte>:1041 [#uses=1]
	seteq ubyte %1041, 0		; <bool>:469 [#uses=1]
	br bool %469, label %471, label %470

; <label>:470		; preds = %469, %470
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %298		; <ubyte*>:666 [#uses=2]
	load ubyte* %666		; <ubyte>:1042 [#uses=1]
	add ubyte %1042, 1		; <ubyte>:1043 [#uses=1]
	store ubyte %1043, ubyte* %666
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %300		; <ubyte*>:667 [#uses=2]
	load ubyte* %667		; <ubyte>:1044 [#uses=2]
	add ubyte %1044, 255		; <ubyte>:1045 [#uses=1]
	store ubyte %1045, ubyte* %667
	seteq ubyte %1044, 1		; <bool>:470 [#uses=1]
	br bool %470, label %471, label %470

; <label>:471		; preds = %469, %470
	add uint %290, 122		; <uint>:301 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %301		; <ubyte*>:668 [#uses=1]
	load ubyte* %668		; <ubyte>:1046 [#uses=1]
	seteq ubyte %1046, 0		; <bool>:471 [#uses=1]
	br bool %471, label %473, label %472

; <label>:472		; preds = %471, %472
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %301		; <ubyte*>:669 [#uses=2]
	load ubyte* %669		; <ubyte>:1047 [#uses=2]
	add ubyte %1047, 255		; <ubyte>:1048 [#uses=1]
	store ubyte %1048, ubyte* %669
	seteq ubyte %1047, 1		; <bool>:472 [#uses=1]
	br bool %472, label %473, label %472

; <label>:473		; preds = %471, %472
	add uint %290, 16		; <uint>:302 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %302		; <ubyte*>:670 [#uses=1]
	load ubyte* %670		; <ubyte>:1049 [#uses=1]
	seteq ubyte %1049, 0		; <bool>:473 [#uses=1]
	br bool %473, label %475, label %474

; <label>:474		; preds = %473, %474
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %302		; <ubyte*>:671 [#uses=2]
	load ubyte* %671		; <ubyte>:1050 [#uses=1]
	add ubyte %1050, 255		; <ubyte>:1051 [#uses=1]
	store ubyte %1051, ubyte* %671
	add uint %290, 17		; <uint>:303 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %303		; <ubyte*>:672 [#uses=2]
	load ubyte* %672		; <ubyte>:1052 [#uses=1]
	add ubyte %1052, 1		; <ubyte>:1053 [#uses=1]
	store ubyte %1053, ubyte* %672
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %301		; <ubyte*>:673 [#uses=2]
	load ubyte* %673		; <ubyte>:1054 [#uses=1]
	add ubyte %1054, 1		; <ubyte>:1055 [#uses=1]
	store ubyte %1055, ubyte* %673
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %302		; <ubyte*>:674 [#uses=1]
	load ubyte* %674		; <ubyte>:1056 [#uses=1]
	seteq ubyte %1056, 0		; <bool>:474 [#uses=1]
	br bool %474, label %475, label %474

; <label>:475		; preds = %473, %474
	add uint %290, 17		; <uint>:304 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %304		; <ubyte*>:675 [#uses=1]
	load ubyte* %675		; <ubyte>:1057 [#uses=1]
	seteq ubyte %1057, 0		; <bool>:475 [#uses=1]
	br bool %475, label %477, label %476

; <label>:476		; preds = %475, %476
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %302		; <ubyte*>:676 [#uses=2]
	load ubyte* %676		; <ubyte>:1058 [#uses=1]
	add ubyte %1058, 1		; <ubyte>:1059 [#uses=1]
	store ubyte %1059, ubyte* %676
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %304		; <ubyte*>:677 [#uses=2]
	load ubyte* %677		; <ubyte>:1060 [#uses=2]
	add ubyte %1060, 255		; <ubyte>:1061 [#uses=1]
	store ubyte %1061, ubyte* %677
	seteq ubyte %1060, 1		; <bool>:476 [#uses=1]
	br bool %476, label %477, label %476

; <label>:477		; preds = %475, %476
	add uint %290, 128		; <uint>:305 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %305		; <ubyte*>:678 [#uses=1]
	load ubyte* %678		; <ubyte>:1062 [#uses=1]
	seteq ubyte %1062, 0		; <bool>:477 [#uses=1]
	br bool %477, label %479, label %478

; <label>:478		; preds = %477, %478
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %305		; <ubyte*>:679 [#uses=2]
	load ubyte* %679		; <ubyte>:1063 [#uses=2]
	add ubyte %1063, 255		; <ubyte>:1064 [#uses=1]
	store ubyte %1064, ubyte* %679
	seteq ubyte %1063, 1		; <bool>:478 [#uses=1]
	br bool %478, label %479, label %478

; <label>:479		; preds = %477, %478
	add uint %290, 22		; <uint>:306 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %306		; <ubyte*>:680 [#uses=1]
	load ubyte* %680		; <ubyte>:1065 [#uses=1]
	seteq ubyte %1065, 0		; <bool>:479 [#uses=1]
	br bool %479, label %481, label %480

; <label>:480		; preds = %479, %480
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %306		; <ubyte*>:681 [#uses=2]
	load ubyte* %681		; <ubyte>:1066 [#uses=1]
	add ubyte %1066, 255		; <ubyte>:1067 [#uses=1]
	store ubyte %1067, ubyte* %681
	add uint %290, 23		; <uint>:307 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %307		; <ubyte*>:682 [#uses=2]
	load ubyte* %682		; <ubyte>:1068 [#uses=1]
	add ubyte %1068, 1		; <ubyte>:1069 [#uses=1]
	store ubyte %1069, ubyte* %682
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %305		; <ubyte*>:683 [#uses=2]
	load ubyte* %683		; <ubyte>:1070 [#uses=1]
	add ubyte %1070, 1		; <ubyte>:1071 [#uses=1]
	store ubyte %1071, ubyte* %683
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %306		; <ubyte*>:684 [#uses=1]
	load ubyte* %684		; <ubyte>:1072 [#uses=1]
	seteq ubyte %1072, 0		; <bool>:480 [#uses=1]
	br bool %480, label %481, label %480

; <label>:481		; preds = %479, %480
	add uint %290, 23		; <uint>:308 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %308		; <ubyte*>:685 [#uses=1]
	load ubyte* %685		; <ubyte>:1073 [#uses=1]
	seteq ubyte %1073, 0		; <bool>:481 [#uses=1]
	br bool %481, label %483, label %482

; <label>:482		; preds = %481, %482
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %306		; <ubyte*>:686 [#uses=2]
	load ubyte* %686		; <ubyte>:1074 [#uses=1]
	add ubyte %1074, 1		; <ubyte>:1075 [#uses=1]
	store ubyte %1075, ubyte* %686
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %308		; <ubyte*>:687 [#uses=2]
	load ubyte* %687		; <ubyte>:1076 [#uses=2]
	add ubyte %1076, 255		; <ubyte>:1077 [#uses=1]
	store ubyte %1077, ubyte* %687
	seteq ubyte %1076, 1		; <bool>:482 [#uses=1]
	br bool %482, label %483, label %482

; <label>:483		; preds = %481, %482
	add uint %290, 134		; <uint>:309 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %309		; <ubyte*>:688 [#uses=1]
	load ubyte* %688		; <ubyte>:1078 [#uses=1]
	seteq ubyte %1078, 0		; <bool>:483 [#uses=1]
	br bool %483, label %485, label %484

; <label>:484		; preds = %483, %484
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %309		; <ubyte*>:689 [#uses=2]
	load ubyte* %689		; <ubyte>:1079 [#uses=2]
	add ubyte %1079, 255		; <ubyte>:1080 [#uses=1]
	store ubyte %1080, ubyte* %689
	seteq ubyte %1079, 1		; <bool>:484 [#uses=1]
	br bool %484, label %485, label %484

; <label>:485		; preds = %483, %484
	add uint %290, 28		; <uint>:310 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %310		; <ubyte*>:690 [#uses=1]
	load ubyte* %690		; <ubyte>:1081 [#uses=1]
	seteq ubyte %1081, 0		; <bool>:485 [#uses=1]
	br bool %485, label %487, label %486

; <label>:486		; preds = %485, %486
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %310		; <ubyte*>:691 [#uses=2]
	load ubyte* %691		; <ubyte>:1082 [#uses=1]
	add ubyte %1082, 255		; <ubyte>:1083 [#uses=1]
	store ubyte %1083, ubyte* %691
	add uint %290, 29		; <uint>:311 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %311		; <ubyte*>:692 [#uses=2]
	load ubyte* %692		; <ubyte>:1084 [#uses=1]
	add ubyte %1084, 1		; <ubyte>:1085 [#uses=1]
	store ubyte %1085, ubyte* %692
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %309		; <ubyte*>:693 [#uses=2]
	load ubyte* %693		; <ubyte>:1086 [#uses=1]
	add ubyte %1086, 1		; <ubyte>:1087 [#uses=1]
	store ubyte %1087, ubyte* %693
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %310		; <ubyte*>:694 [#uses=1]
	load ubyte* %694		; <ubyte>:1088 [#uses=1]
	seteq ubyte %1088, 0		; <bool>:486 [#uses=1]
	br bool %486, label %487, label %486

; <label>:487		; preds = %485, %486
	add uint %290, 29		; <uint>:312 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %312		; <ubyte*>:695 [#uses=1]
	load ubyte* %695		; <ubyte>:1089 [#uses=1]
	seteq ubyte %1089, 0		; <bool>:487 [#uses=1]
	br bool %487, label %489, label %488

; <label>:488		; preds = %487, %488
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %310		; <ubyte*>:696 [#uses=2]
	load ubyte* %696		; <ubyte>:1090 [#uses=1]
	add ubyte %1090, 1		; <ubyte>:1091 [#uses=1]
	store ubyte %1091, ubyte* %696
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %312		; <ubyte*>:697 [#uses=2]
	load ubyte* %697		; <ubyte>:1092 [#uses=2]
	add ubyte %1092, 255		; <ubyte>:1093 [#uses=1]
	store ubyte %1093, ubyte* %697
	seteq ubyte %1092, 1		; <bool>:488 [#uses=1]
	br bool %488, label %489, label %488

; <label>:489		; preds = %487, %488
	add uint %290, 140		; <uint>:313 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %313		; <ubyte*>:698 [#uses=1]
	load ubyte* %698		; <ubyte>:1094 [#uses=1]
	seteq ubyte %1094, 0		; <bool>:489 [#uses=1]
	br bool %489, label %491, label %490

; <label>:490		; preds = %489, %490
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %313		; <ubyte*>:699 [#uses=2]
	load ubyte* %699		; <ubyte>:1095 [#uses=2]
	add ubyte %1095, 255		; <ubyte>:1096 [#uses=1]
	store ubyte %1096, ubyte* %699
	seteq ubyte %1095, 1		; <bool>:490 [#uses=1]
	br bool %490, label %491, label %490

; <label>:491		; preds = %489, %490
	add uint %290, 34		; <uint>:314 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %314		; <ubyte*>:700 [#uses=1]
	load ubyte* %700		; <ubyte>:1097 [#uses=1]
	seteq ubyte %1097, 0		; <bool>:491 [#uses=1]
	br bool %491, label %493, label %492

; <label>:492		; preds = %491, %492
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %314		; <ubyte*>:701 [#uses=2]
	load ubyte* %701		; <ubyte>:1098 [#uses=1]
	add ubyte %1098, 255		; <ubyte>:1099 [#uses=1]
	store ubyte %1099, ubyte* %701
	add uint %290, 35		; <uint>:315 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %315		; <ubyte*>:702 [#uses=2]
	load ubyte* %702		; <ubyte>:1100 [#uses=1]
	add ubyte %1100, 1		; <ubyte>:1101 [#uses=1]
	store ubyte %1101, ubyte* %702
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %313		; <ubyte*>:703 [#uses=2]
	load ubyte* %703		; <ubyte>:1102 [#uses=1]
	add ubyte %1102, 1		; <ubyte>:1103 [#uses=1]
	store ubyte %1103, ubyte* %703
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %314		; <ubyte*>:704 [#uses=1]
	load ubyte* %704		; <ubyte>:1104 [#uses=1]
	seteq ubyte %1104, 0		; <bool>:492 [#uses=1]
	br bool %492, label %493, label %492

; <label>:493		; preds = %491, %492
	add uint %290, 35		; <uint>:316 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %316		; <ubyte*>:705 [#uses=1]
	load ubyte* %705		; <ubyte>:1105 [#uses=1]
	seteq ubyte %1105, 0		; <bool>:493 [#uses=1]
	br bool %493, label %495, label %494

; <label>:494		; preds = %493, %494
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %314		; <ubyte*>:706 [#uses=2]
	load ubyte* %706		; <ubyte>:1106 [#uses=1]
	add ubyte %1106, 1		; <ubyte>:1107 [#uses=1]
	store ubyte %1107, ubyte* %706
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %316		; <ubyte*>:707 [#uses=2]
	load ubyte* %707		; <ubyte>:1108 [#uses=2]
	add ubyte %1108, 255		; <ubyte>:1109 [#uses=1]
	store ubyte %1109, ubyte* %707
	seteq ubyte %1108, 1		; <bool>:494 [#uses=1]
	br bool %494, label %495, label %494

; <label>:495		; preds = %493, %494
	add uint %290, 146		; <uint>:317 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %317		; <ubyte*>:708 [#uses=1]
	load ubyte* %708		; <ubyte>:1110 [#uses=1]
	seteq ubyte %1110, 0		; <bool>:495 [#uses=1]
	br bool %495, label %497, label %496

; <label>:496		; preds = %495, %496
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %317		; <ubyte*>:709 [#uses=2]
	load ubyte* %709		; <ubyte>:1111 [#uses=2]
	add ubyte %1111, 255		; <ubyte>:1112 [#uses=1]
	store ubyte %1112, ubyte* %709
	seteq ubyte %1111, 1		; <bool>:496 [#uses=1]
	br bool %496, label %497, label %496

; <label>:497		; preds = %495, %496
	add uint %290, 40		; <uint>:318 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %318		; <ubyte*>:710 [#uses=1]
	load ubyte* %710		; <ubyte>:1113 [#uses=1]
	seteq ubyte %1113, 0		; <bool>:497 [#uses=1]
	br bool %497, label %499, label %498

; <label>:498		; preds = %497, %498
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %318		; <ubyte*>:711 [#uses=2]
	load ubyte* %711		; <ubyte>:1114 [#uses=1]
	add ubyte %1114, 255		; <ubyte>:1115 [#uses=1]
	store ubyte %1115, ubyte* %711
	add uint %290, 41		; <uint>:319 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %319		; <ubyte*>:712 [#uses=2]
	load ubyte* %712		; <ubyte>:1116 [#uses=1]
	add ubyte %1116, 1		; <ubyte>:1117 [#uses=1]
	store ubyte %1117, ubyte* %712
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %317		; <ubyte*>:713 [#uses=2]
	load ubyte* %713		; <ubyte>:1118 [#uses=1]
	add ubyte %1118, 1		; <ubyte>:1119 [#uses=1]
	store ubyte %1119, ubyte* %713
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %318		; <ubyte*>:714 [#uses=1]
	load ubyte* %714		; <ubyte>:1120 [#uses=1]
	seteq ubyte %1120, 0		; <bool>:498 [#uses=1]
	br bool %498, label %499, label %498

; <label>:499		; preds = %497, %498
	add uint %290, 41		; <uint>:320 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %320		; <ubyte*>:715 [#uses=1]
	load ubyte* %715		; <ubyte>:1121 [#uses=1]
	seteq ubyte %1121, 0		; <bool>:499 [#uses=1]
	br bool %499, label %501, label %500

; <label>:500		; preds = %499, %500
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %318		; <ubyte*>:716 [#uses=2]
	load ubyte* %716		; <ubyte>:1122 [#uses=1]
	add ubyte %1122, 1		; <ubyte>:1123 [#uses=1]
	store ubyte %1123, ubyte* %716
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %320		; <ubyte*>:717 [#uses=2]
	load ubyte* %717		; <ubyte>:1124 [#uses=2]
	add ubyte %1124, 255		; <ubyte>:1125 [#uses=1]
	store ubyte %1125, ubyte* %717
	seteq ubyte %1124, 1		; <bool>:500 [#uses=1]
	br bool %500, label %501, label %500

; <label>:501		; preds = %499, %500
	add uint %290, 152		; <uint>:321 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %321		; <ubyte*>:718 [#uses=1]
	load ubyte* %718		; <ubyte>:1126 [#uses=1]
	seteq ubyte %1126, 0		; <bool>:501 [#uses=1]
	br bool %501, label %503, label %502

; <label>:502		; preds = %501, %502
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %321		; <ubyte*>:719 [#uses=2]
	load ubyte* %719		; <ubyte>:1127 [#uses=2]
	add ubyte %1127, 255		; <ubyte>:1128 [#uses=1]
	store ubyte %1128, ubyte* %719
	seteq ubyte %1127, 1		; <bool>:502 [#uses=1]
	br bool %502, label %503, label %502

; <label>:503		; preds = %501, %502
	add uint %290, 46		; <uint>:322 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %322		; <ubyte*>:720 [#uses=1]
	load ubyte* %720		; <ubyte>:1129 [#uses=1]
	seteq ubyte %1129, 0		; <bool>:503 [#uses=1]
	br bool %503, label %505, label %504

; <label>:504		; preds = %503, %504
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %322		; <ubyte*>:721 [#uses=2]
	load ubyte* %721		; <ubyte>:1130 [#uses=1]
	add ubyte %1130, 255		; <ubyte>:1131 [#uses=1]
	store ubyte %1131, ubyte* %721
	add uint %290, 47		; <uint>:323 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %323		; <ubyte*>:722 [#uses=2]
	load ubyte* %722		; <ubyte>:1132 [#uses=1]
	add ubyte %1132, 1		; <ubyte>:1133 [#uses=1]
	store ubyte %1133, ubyte* %722
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %321		; <ubyte*>:723 [#uses=2]
	load ubyte* %723		; <ubyte>:1134 [#uses=1]
	add ubyte %1134, 1		; <ubyte>:1135 [#uses=1]
	store ubyte %1135, ubyte* %723
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %322		; <ubyte*>:724 [#uses=1]
	load ubyte* %724		; <ubyte>:1136 [#uses=1]
	seteq ubyte %1136, 0		; <bool>:504 [#uses=1]
	br bool %504, label %505, label %504

; <label>:505		; preds = %503, %504
	add uint %290, 47		; <uint>:324 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %324		; <ubyte*>:725 [#uses=1]
	load ubyte* %725		; <ubyte>:1137 [#uses=1]
	seteq ubyte %1137, 0		; <bool>:505 [#uses=1]
	br bool %505, label %507, label %506

; <label>:506		; preds = %505, %506
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %322		; <ubyte*>:726 [#uses=2]
	load ubyte* %726		; <ubyte>:1138 [#uses=1]
	add ubyte %1138, 1		; <ubyte>:1139 [#uses=1]
	store ubyte %1139, ubyte* %726
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %324		; <ubyte*>:727 [#uses=2]
	load ubyte* %727		; <ubyte>:1140 [#uses=2]
	add ubyte %1140, 255		; <ubyte>:1141 [#uses=1]
	store ubyte %1141, ubyte* %727
	seteq ubyte %1140, 1		; <bool>:506 [#uses=1]
	br bool %506, label %507, label %506

; <label>:507		; preds = %505, %506
	add uint %290, 158		; <uint>:325 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %325		; <ubyte*>:728 [#uses=1]
	load ubyte* %728		; <ubyte>:1142 [#uses=1]
	seteq ubyte %1142, 0		; <bool>:507 [#uses=1]
	br bool %507, label %509, label %508

; <label>:508		; preds = %507, %508
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %325		; <ubyte*>:729 [#uses=2]
	load ubyte* %729		; <ubyte>:1143 [#uses=2]
	add ubyte %1143, 255		; <ubyte>:1144 [#uses=1]
	store ubyte %1144, ubyte* %729
	seteq ubyte %1143, 1		; <bool>:508 [#uses=1]
	br bool %508, label %509, label %508

; <label>:509		; preds = %507, %508
	add uint %290, 52		; <uint>:326 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %326		; <ubyte*>:730 [#uses=1]
	load ubyte* %730		; <ubyte>:1145 [#uses=1]
	seteq ubyte %1145, 0		; <bool>:509 [#uses=1]
	br bool %509, label %511, label %510

; <label>:510		; preds = %509, %510
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %326		; <ubyte*>:731 [#uses=2]
	load ubyte* %731		; <ubyte>:1146 [#uses=1]
	add ubyte %1146, 255		; <ubyte>:1147 [#uses=1]
	store ubyte %1147, ubyte* %731
	add uint %290, 53		; <uint>:327 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %327		; <ubyte*>:732 [#uses=2]
	load ubyte* %732		; <ubyte>:1148 [#uses=1]
	add ubyte %1148, 1		; <ubyte>:1149 [#uses=1]
	store ubyte %1149, ubyte* %732
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %325		; <ubyte*>:733 [#uses=2]
	load ubyte* %733		; <ubyte>:1150 [#uses=1]
	add ubyte %1150, 1		; <ubyte>:1151 [#uses=1]
	store ubyte %1151, ubyte* %733
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %326		; <ubyte*>:734 [#uses=1]
	load ubyte* %734		; <ubyte>:1152 [#uses=1]
	seteq ubyte %1152, 0		; <bool>:510 [#uses=1]
	br bool %510, label %511, label %510

; <label>:511		; preds = %509, %510
	add uint %290, 53		; <uint>:328 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %328		; <ubyte*>:735 [#uses=1]
	load ubyte* %735		; <ubyte>:1153 [#uses=1]
	seteq ubyte %1153, 0		; <bool>:511 [#uses=1]
	br bool %511, label %513, label %512

; <label>:512		; preds = %511, %512
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %326		; <ubyte*>:736 [#uses=2]
	load ubyte* %736		; <ubyte>:1154 [#uses=1]
	add ubyte %1154, 1		; <ubyte>:1155 [#uses=1]
	store ubyte %1155, ubyte* %736
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %328		; <ubyte*>:737 [#uses=2]
	load ubyte* %737		; <ubyte>:1156 [#uses=2]
	add ubyte %1156, 255		; <ubyte>:1157 [#uses=1]
	store ubyte %1157, ubyte* %737
	seteq ubyte %1156, 1		; <bool>:512 [#uses=1]
	br bool %512, label %513, label %512

; <label>:513		; preds = %511, %512
	add uint %290, 164		; <uint>:329 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %329		; <ubyte*>:738 [#uses=1]
	load ubyte* %738		; <ubyte>:1158 [#uses=1]
	seteq ubyte %1158, 0		; <bool>:513 [#uses=1]
	br bool %513, label %515, label %514

; <label>:514		; preds = %513, %514
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %329		; <ubyte*>:739 [#uses=2]
	load ubyte* %739		; <ubyte>:1159 [#uses=2]
	add ubyte %1159, 255		; <ubyte>:1160 [#uses=1]
	store ubyte %1160, ubyte* %739
	seteq ubyte %1159, 1		; <bool>:514 [#uses=1]
	br bool %514, label %515, label %514

; <label>:515		; preds = %513, %514
	add uint %290, 58		; <uint>:330 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %330		; <ubyte*>:740 [#uses=1]
	load ubyte* %740		; <ubyte>:1161 [#uses=1]
	seteq ubyte %1161, 0		; <bool>:515 [#uses=1]
	br bool %515, label %517, label %516

; <label>:516		; preds = %515, %516
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %330		; <ubyte*>:741 [#uses=2]
	load ubyte* %741		; <ubyte>:1162 [#uses=1]
	add ubyte %1162, 255		; <ubyte>:1163 [#uses=1]
	store ubyte %1163, ubyte* %741
	add uint %290, 59		; <uint>:331 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %331		; <ubyte*>:742 [#uses=2]
	load ubyte* %742		; <ubyte>:1164 [#uses=1]
	add ubyte %1164, 1		; <ubyte>:1165 [#uses=1]
	store ubyte %1165, ubyte* %742
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %329		; <ubyte*>:743 [#uses=2]
	load ubyte* %743		; <ubyte>:1166 [#uses=1]
	add ubyte %1166, 1		; <ubyte>:1167 [#uses=1]
	store ubyte %1167, ubyte* %743
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %330		; <ubyte*>:744 [#uses=1]
	load ubyte* %744		; <ubyte>:1168 [#uses=1]
	seteq ubyte %1168, 0		; <bool>:516 [#uses=1]
	br bool %516, label %517, label %516

; <label>:517		; preds = %515, %516
	add uint %290, 59		; <uint>:332 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %332		; <ubyte*>:745 [#uses=1]
	load ubyte* %745		; <ubyte>:1169 [#uses=1]
	seteq ubyte %1169, 0		; <bool>:517 [#uses=1]
	br bool %517, label %519, label %518

; <label>:518		; preds = %517, %518
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %330		; <ubyte*>:746 [#uses=2]
	load ubyte* %746		; <ubyte>:1170 [#uses=1]
	add ubyte %1170, 1		; <ubyte>:1171 [#uses=1]
	store ubyte %1171, ubyte* %746
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %332		; <ubyte*>:747 [#uses=2]
	load ubyte* %747		; <ubyte>:1172 [#uses=2]
	add ubyte %1172, 255		; <ubyte>:1173 [#uses=1]
	store ubyte %1173, ubyte* %747
	seteq ubyte %1172, 1		; <bool>:518 [#uses=1]
	br bool %518, label %519, label %518

; <label>:519		; preds = %517, %518
	add uint %290, 170		; <uint>:333 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %333		; <ubyte*>:748 [#uses=1]
	load ubyte* %748		; <ubyte>:1174 [#uses=1]
	seteq ubyte %1174, 0		; <bool>:519 [#uses=1]
	br bool %519, label %521, label %520

; <label>:520		; preds = %519, %520
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %333		; <ubyte*>:749 [#uses=2]
	load ubyte* %749		; <ubyte>:1175 [#uses=2]
	add ubyte %1175, 255		; <ubyte>:1176 [#uses=1]
	store ubyte %1176, ubyte* %749
	seteq ubyte %1175, 1		; <bool>:520 [#uses=1]
	br bool %520, label %521, label %520

; <label>:521		; preds = %519, %520
	add uint %290, 64		; <uint>:334 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %334		; <ubyte*>:750 [#uses=1]
	load ubyte* %750		; <ubyte>:1177 [#uses=1]
	seteq ubyte %1177, 0		; <bool>:521 [#uses=1]
	br bool %521, label %523, label %522

; <label>:522		; preds = %521, %522
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %334		; <ubyte*>:751 [#uses=2]
	load ubyte* %751		; <ubyte>:1178 [#uses=1]
	add ubyte %1178, 255		; <ubyte>:1179 [#uses=1]
	store ubyte %1179, ubyte* %751
	add uint %290, 65		; <uint>:335 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %335		; <ubyte*>:752 [#uses=2]
	load ubyte* %752		; <ubyte>:1180 [#uses=1]
	add ubyte %1180, 1		; <ubyte>:1181 [#uses=1]
	store ubyte %1181, ubyte* %752
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %333		; <ubyte*>:753 [#uses=2]
	load ubyte* %753		; <ubyte>:1182 [#uses=1]
	add ubyte %1182, 1		; <ubyte>:1183 [#uses=1]
	store ubyte %1183, ubyte* %753
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %334		; <ubyte*>:754 [#uses=1]
	load ubyte* %754		; <ubyte>:1184 [#uses=1]
	seteq ubyte %1184, 0		; <bool>:522 [#uses=1]
	br bool %522, label %523, label %522

; <label>:523		; preds = %521, %522
	add uint %290, 65		; <uint>:336 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %336		; <ubyte*>:755 [#uses=1]
	load ubyte* %755		; <ubyte>:1185 [#uses=1]
	seteq ubyte %1185, 0		; <bool>:523 [#uses=1]
	br bool %523, label %525, label %524

; <label>:524		; preds = %523, %524
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %334		; <ubyte*>:756 [#uses=2]
	load ubyte* %756		; <ubyte>:1186 [#uses=1]
	add ubyte %1186, 1		; <ubyte>:1187 [#uses=1]
	store ubyte %1187, ubyte* %756
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %336		; <ubyte*>:757 [#uses=2]
	load ubyte* %757		; <ubyte>:1188 [#uses=2]
	add ubyte %1188, 255		; <ubyte>:1189 [#uses=1]
	store ubyte %1189, ubyte* %757
	seteq ubyte %1188, 1		; <bool>:524 [#uses=1]
	br bool %524, label %525, label %524

; <label>:525		; preds = %523, %524
	add uint %290, 176		; <uint>:337 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %337		; <ubyte*>:758 [#uses=1]
	load ubyte* %758		; <ubyte>:1190 [#uses=1]
	seteq ubyte %1190, 0		; <bool>:525 [#uses=1]
	br bool %525, label %527, label %526

; <label>:526		; preds = %525, %526
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %337		; <ubyte*>:759 [#uses=2]
	load ubyte* %759		; <ubyte>:1191 [#uses=2]
	add ubyte %1191, 255		; <ubyte>:1192 [#uses=1]
	store ubyte %1192, ubyte* %759
	seteq ubyte %1191, 1		; <bool>:526 [#uses=1]
	br bool %526, label %527, label %526

; <label>:527		; preds = %525, %526
	add uint %290, 70		; <uint>:338 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %338		; <ubyte*>:760 [#uses=1]
	load ubyte* %760		; <ubyte>:1193 [#uses=1]
	seteq ubyte %1193, 0		; <bool>:527 [#uses=1]
	br bool %527, label %529, label %528

; <label>:528		; preds = %527, %528
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %338		; <ubyte*>:761 [#uses=2]
	load ubyte* %761		; <ubyte>:1194 [#uses=1]
	add ubyte %1194, 255		; <ubyte>:1195 [#uses=1]
	store ubyte %1195, ubyte* %761
	add uint %290, 71		; <uint>:339 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %339		; <ubyte*>:762 [#uses=2]
	load ubyte* %762		; <ubyte>:1196 [#uses=1]
	add ubyte %1196, 1		; <ubyte>:1197 [#uses=1]
	store ubyte %1197, ubyte* %762
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %337		; <ubyte*>:763 [#uses=2]
	load ubyte* %763		; <ubyte>:1198 [#uses=1]
	add ubyte %1198, 1		; <ubyte>:1199 [#uses=1]
	store ubyte %1199, ubyte* %763
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %338		; <ubyte*>:764 [#uses=1]
	load ubyte* %764		; <ubyte>:1200 [#uses=1]
	seteq ubyte %1200, 0		; <bool>:528 [#uses=1]
	br bool %528, label %529, label %528

; <label>:529		; preds = %527, %528
	add uint %290, 71		; <uint>:340 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %340		; <ubyte*>:765 [#uses=1]
	load ubyte* %765		; <ubyte>:1201 [#uses=1]
	seteq ubyte %1201, 0		; <bool>:529 [#uses=1]
	br bool %529, label %531, label %530

; <label>:530		; preds = %529, %530
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %338		; <ubyte*>:766 [#uses=2]
	load ubyte* %766		; <ubyte>:1202 [#uses=1]
	add ubyte %1202, 1		; <ubyte>:1203 [#uses=1]
	store ubyte %1203, ubyte* %766
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %340		; <ubyte*>:767 [#uses=2]
	load ubyte* %767		; <ubyte>:1204 [#uses=2]
	add ubyte %1204, 255		; <ubyte>:1205 [#uses=1]
	store ubyte %1205, ubyte* %767
	seteq ubyte %1204, 1		; <bool>:530 [#uses=1]
	br bool %530, label %531, label %530

; <label>:531		; preds = %529, %530
	add uint %290, 182		; <uint>:341 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %341		; <ubyte*>:768 [#uses=1]
	load ubyte* %768		; <ubyte>:1206 [#uses=1]
	seteq ubyte %1206, 0		; <bool>:531 [#uses=1]
	br bool %531, label %533, label %532

; <label>:532		; preds = %531, %532
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %341		; <ubyte*>:769 [#uses=2]
	load ubyte* %769		; <ubyte>:1207 [#uses=2]
	add ubyte %1207, 255		; <ubyte>:1208 [#uses=1]
	store ubyte %1208, ubyte* %769
	seteq ubyte %1207, 1		; <bool>:532 [#uses=1]
	br bool %532, label %533, label %532

; <label>:533		; preds = %531, %532
	add uint %290, 76		; <uint>:342 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %342		; <ubyte*>:770 [#uses=1]
	load ubyte* %770		; <ubyte>:1209 [#uses=1]
	seteq ubyte %1209, 0		; <bool>:533 [#uses=1]
	br bool %533, label %535, label %534

; <label>:534		; preds = %533, %534
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %342		; <ubyte*>:771 [#uses=2]
	load ubyte* %771		; <ubyte>:1210 [#uses=1]
	add ubyte %1210, 255		; <ubyte>:1211 [#uses=1]
	store ubyte %1211, ubyte* %771
	add uint %290, 77		; <uint>:343 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %343		; <ubyte*>:772 [#uses=2]
	load ubyte* %772		; <ubyte>:1212 [#uses=1]
	add ubyte %1212, 1		; <ubyte>:1213 [#uses=1]
	store ubyte %1213, ubyte* %772
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %341		; <ubyte*>:773 [#uses=2]
	load ubyte* %773		; <ubyte>:1214 [#uses=1]
	add ubyte %1214, 1		; <ubyte>:1215 [#uses=1]
	store ubyte %1215, ubyte* %773
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %342		; <ubyte*>:774 [#uses=1]
	load ubyte* %774		; <ubyte>:1216 [#uses=1]
	seteq ubyte %1216, 0		; <bool>:534 [#uses=1]
	br bool %534, label %535, label %534

; <label>:535		; preds = %533, %534
	add uint %290, 77		; <uint>:344 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %344		; <ubyte*>:775 [#uses=1]
	load ubyte* %775		; <ubyte>:1217 [#uses=1]
	seteq ubyte %1217, 0		; <bool>:535 [#uses=1]
	br bool %535, label %537, label %536

; <label>:536		; preds = %535, %536
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %342		; <ubyte*>:776 [#uses=2]
	load ubyte* %776		; <ubyte>:1218 [#uses=1]
	add ubyte %1218, 1		; <ubyte>:1219 [#uses=1]
	store ubyte %1219, ubyte* %776
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %344		; <ubyte*>:777 [#uses=2]
	load ubyte* %777		; <ubyte>:1220 [#uses=2]
	add ubyte %1220, 255		; <ubyte>:1221 [#uses=1]
	store ubyte %1221, ubyte* %777
	seteq ubyte %1220, 1		; <bool>:536 [#uses=1]
	br bool %536, label %537, label %536

; <label>:537		; preds = %535, %536
	add uint %290, 188		; <uint>:345 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %345		; <ubyte*>:778 [#uses=1]
	load ubyte* %778		; <ubyte>:1222 [#uses=1]
	seteq ubyte %1222, 0		; <bool>:537 [#uses=1]
	br bool %537, label %539, label %538

; <label>:538		; preds = %537, %538
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %345		; <ubyte*>:779 [#uses=2]
	load ubyte* %779		; <ubyte>:1223 [#uses=2]
	add ubyte %1223, 255		; <ubyte>:1224 [#uses=1]
	store ubyte %1224, ubyte* %779
	seteq ubyte %1223, 1		; <bool>:538 [#uses=1]
	br bool %538, label %539, label %538

; <label>:539		; preds = %537, %538
	add uint %290, 82		; <uint>:346 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %346		; <ubyte*>:780 [#uses=1]
	load ubyte* %780		; <ubyte>:1225 [#uses=1]
	seteq ubyte %1225, 0		; <bool>:539 [#uses=1]
	br bool %539, label %541, label %540

; <label>:540		; preds = %539, %540
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %346		; <ubyte*>:781 [#uses=2]
	load ubyte* %781		; <ubyte>:1226 [#uses=1]
	add ubyte %1226, 255		; <ubyte>:1227 [#uses=1]
	store ubyte %1227, ubyte* %781
	add uint %290, 83		; <uint>:347 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %347		; <ubyte*>:782 [#uses=2]
	load ubyte* %782		; <ubyte>:1228 [#uses=1]
	add ubyte %1228, 1		; <ubyte>:1229 [#uses=1]
	store ubyte %1229, ubyte* %782
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %345		; <ubyte*>:783 [#uses=2]
	load ubyte* %783		; <ubyte>:1230 [#uses=1]
	add ubyte %1230, 1		; <ubyte>:1231 [#uses=1]
	store ubyte %1231, ubyte* %783
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %346		; <ubyte*>:784 [#uses=1]
	load ubyte* %784		; <ubyte>:1232 [#uses=1]
	seteq ubyte %1232, 0		; <bool>:540 [#uses=1]
	br bool %540, label %541, label %540

; <label>:541		; preds = %539, %540
	add uint %290, 83		; <uint>:348 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %348		; <ubyte*>:785 [#uses=1]
	load ubyte* %785		; <ubyte>:1233 [#uses=1]
	seteq ubyte %1233, 0		; <bool>:541 [#uses=1]
	br bool %541, label %543, label %542

; <label>:542		; preds = %541, %542
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %346		; <ubyte*>:786 [#uses=2]
	load ubyte* %786		; <ubyte>:1234 [#uses=1]
	add ubyte %1234, 1		; <ubyte>:1235 [#uses=1]
	store ubyte %1235, ubyte* %786
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %348		; <ubyte*>:787 [#uses=2]
	load ubyte* %787		; <ubyte>:1236 [#uses=2]
	add ubyte %1236, 255		; <ubyte>:1237 [#uses=1]
	store ubyte %1237, ubyte* %787
	seteq ubyte %1236, 1		; <bool>:542 [#uses=1]
	br bool %542, label %543, label %542

; <label>:543		; preds = %541, %542
	add uint %290, 194		; <uint>:349 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %349		; <ubyte*>:788 [#uses=1]
	load ubyte* %788		; <ubyte>:1238 [#uses=1]
	seteq ubyte %1238, 0		; <bool>:543 [#uses=1]
	br bool %543, label %545, label %544

; <label>:544		; preds = %543, %544
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %349		; <ubyte*>:789 [#uses=2]
	load ubyte* %789		; <ubyte>:1239 [#uses=2]
	add ubyte %1239, 255		; <ubyte>:1240 [#uses=1]
	store ubyte %1240, ubyte* %789
	seteq ubyte %1239, 1		; <bool>:544 [#uses=1]
	br bool %544, label %545, label %544

; <label>:545		; preds = %543, %544
	add uint %290, 88		; <uint>:350 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %350		; <ubyte*>:790 [#uses=1]
	load ubyte* %790		; <ubyte>:1241 [#uses=1]
	seteq ubyte %1241, 0		; <bool>:545 [#uses=1]
	br bool %545, label %547, label %546

; <label>:546		; preds = %545, %546
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %350		; <ubyte*>:791 [#uses=2]
	load ubyte* %791		; <ubyte>:1242 [#uses=1]
	add ubyte %1242, 255		; <ubyte>:1243 [#uses=1]
	store ubyte %1243, ubyte* %791
	add uint %290, 89		; <uint>:351 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %351		; <ubyte*>:792 [#uses=2]
	load ubyte* %792		; <ubyte>:1244 [#uses=1]
	add ubyte %1244, 1		; <ubyte>:1245 [#uses=1]
	store ubyte %1245, ubyte* %792
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %349		; <ubyte*>:793 [#uses=2]
	load ubyte* %793		; <ubyte>:1246 [#uses=1]
	add ubyte %1246, 1		; <ubyte>:1247 [#uses=1]
	store ubyte %1247, ubyte* %793
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %350		; <ubyte*>:794 [#uses=1]
	load ubyte* %794		; <ubyte>:1248 [#uses=1]
	seteq ubyte %1248, 0		; <bool>:546 [#uses=1]
	br bool %546, label %547, label %546

; <label>:547		; preds = %545, %546
	add uint %290, 89		; <uint>:352 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %352		; <ubyte*>:795 [#uses=1]
	load ubyte* %795		; <ubyte>:1249 [#uses=1]
	seteq ubyte %1249, 0		; <bool>:547 [#uses=1]
	br bool %547, label %549, label %548

; <label>:548		; preds = %547, %548
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %350		; <ubyte*>:796 [#uses=2]
	load ubyte* %796		; <ubyte>:1250 [#uses=1]
	add ubyte %1250, 1		; <ubyte>:1251 [#uses=1]
	store ubyte %1251, ubyte* %796
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %352		; <ubyte*>:797 [#uses=2]
	load ubyte* %797		; <ubyte>:1252 [#uses=2]
	add ubyte %1252, 255		; <ubyte>:1253 [#uses=1]
	store ubyte %1253, ubyte* %797
	seteq ubyte %1252, 1		; <bool>:548 [#uses=1]
	br bool %548, label %549, label %548

; <label>:549		; preds = %547, %548
	add uint %290, 200		; <uint>:353 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %353		; <ubyte*>:798 [#uses=1]
	load ubyte* %798		; <ubyte>:1254 [#uses=1]
	seteq ubyte %1254, 0		; <bool>:549 [#uses=1]
	br bool %549, label %551, label %550

; <label>:550		; preds = %549, %550
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %353		; <ubyte*>:799 [#uses=2]
	load ubyte* %799		; <ubyte>:1255 [#uses=2]
	add ubyte %1255, 255		; <ubyte>:1256 [#uses=1]
	store ubyte %1256, ubyte* %799
	seteq ubyte %1255, 1		; <bool>:550 [#uses=1]
	br bool %550, label %551, label %550

; <label>:551		; preds = %549, %550
	add uint %290, 94		; <uint>:354 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %354		; <ubyte*>:800 [#uses=1]
	load ubyte* %800		; <ubyte>:1257 [#uses=1]
	seteq ubyte %1257, 0		; <bool>:551 [#uses=1]
	br bool %551, label %553, label %552

; <label>:552		; preds = %551, %552
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %354		; <ubyte*>:801 [#uses=2]
	load ubyte* %801		; <ubyte>:1258 [#uses=1]
	add ubyte %1258, 255		; <ubyte>:1259 [#uses=1]
	store ubyte %1259, ubyte* %801
	add uint %290, 95		; <uint>:355 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %355		; <ubyte*>:802 [#uses=2]
	load ubyte* %802		; <ubyte>:1260 [#uses=1]
	add ubyte %1260, 1		; <ubyte>:1261 [#uses=1]
	store ubyte %1261, ubyte* %802
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %353		; <ubyte*>:803 [#uses=2]
	load ubyte* %803		; <ubyte>:1262 [#uses=1]
	add ubyte %1262, 1		; <ubyte>:1263 [#uses=1]
	store ubyte %1263, ubyte* %803
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %354		; <ubyte*>:804 [#uses=1]
	load ubyte* %804		; <ubyte>:1264 [#uses=1]
	seteq ubyte %1264, 0		; <bool>:552 [#uses=1]
	br bool %552, label %553, label %552

; <label>:553		; preds = %551, %552
	add uint %290, 95		; <uint>:356 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %356		; <ubyte*>:805 [#uses=1]
	load ubyte* %805		; <ubyte>:1265 [#uses=1]
	seteq ubyte %1265, 0		; <bool>:553 [#uses=1]
	br bool %553, label %555, label %554

; <label>:554		; preds = %553, %554
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %354		; <ubyte*>:806 [#uses=2]
	load ubyte* %806		; <ubyte>:1266 [#uses=1]
	add ubyte %1266, 1		; <ubyte>:1267 [#uses=1]
	store ubyte %1267, ubyte* %806
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %356		; <ubyte*>:807 [#uses=2]
	load ubyte* %807		; <ubyte>:1268 [#uses=2]
	add ubyte %1268, 255		; <ubyte>:1269 [#uses=1]
	store ubyte %1269, ubyte* %807
	seteq ubyte %1268, 1		; <bool>:554 [#uses=1]
	br bool %554, label %555, label %554

; <label>:555		; preds = %553, %554
	add uint %290, 204		; <uint>:357 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %357		; <ubyte*>:808 [#uses=2]
	load ubyte* %808		; <ubyte>:1270 [#uses=1]
	add ubyte %1270, 5		; <ubyte>:1271 [#uses=1]
	store ubyte %1271, ubyte* %808
	add uint %290, 207		; <uint>:358 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %358		; <ubyte*>:809 [#uses=2]
	load ubyte* %809		; <ubyte>:1272 [#uses=1]
	add ubyte %1272, 2		; <ubyte>:1273 [#uses=1]
	store ubyte %1273, ubyte* %809
	add uint %290, 210		; <uint>:359 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %359		; <ubyte*>:810 [#uses=2]
	load ubyte* %810		; <ubyte>:1274 [#uses=1]
	add ubyte %1274, 1		; <ubyte>:1275 [#uses=1]
	store ubyte %1275, ubyte* %810
	add uint %290, 213		; <uint>:360 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %360		; <ubyte*>:811 [#uses=2]
	load ubyte* %811		; <ubyte>:1276 [#uses=1]
	add ubyte %1276, 1		; <ubyte>:1277 [#uses=1]
	store ubyte %1277, ubyte* %811
	add uint %290, 218		; <uint>:361 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %361		; <ubyte*>:812 [#uses=1]
	load ubyte* %812		; <ubyte>:1278 [#uses=1]
	seteq ubyte %1278, 0		; <bool>:555 [#uses=1]
	br bool %555, label %459, label %458

; <label>:556		; preds = %17, %565
	phi uint [ %25, %17 ], [ %373, %565 ]		; <uint>:362 [#uses=10]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %362		; <ubyte*>:813 [#uses=2]
	load ubyte* %813		; <ubyte>:1279 [#uses=1]
	add ubyte %1279, 255		; <ubyte>:1280 [#uses=1]
	store ubyte %1280, ubyte* %813
	add uint %362, 104		; <uint>:363 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %363		; <ubyte*>:814 [#uses=1]
	call ubyte %inputcell( )		; <ubyte>:1281 [#uses=1]
	store ubyte %1281, ubyte* %814
	add uint %362, 107		; <uint>:364 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %364		; <ubyte*>:815 [#uses=2]
	load ubyte* %815		; <ubyte>:1282 [#uses=2]
	add ubyte %1282, 7		; <ubyte>:1283 [#uses=1]
	store ubyte %1283, ubyte* %815
	seteq ubyte %1282, 249		; <bool>:556 [#uses=1]
	br bool %556, label %559, label %558

; <label>:557		; preds = %17, %565
	phi uint [ %25, %17 ], [ %373, %565 ]		; <uint>:365 [#uses=1]
	add uint %365, 4294967295		; <uint>:366 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %366		; <ubyte*>:816 [#uses=1]
	load ubyte* %816		; <ubyte>:1284 [#uses=1]
	seteq ubyte %1284, 0		; <bool>:557 [#uses=1]
	br bool %557, label %15, label %14

; <label>:558		; preds = %556, %558
	add uint %362, 106		; <uint>:367 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %367		; <ubyte*>:817 [#uses=2]
	load ubyte* %817		; <ubyte>:1285 [#uses=1]
	add ubyte %1285, 7		; <ubyte>:1286 [#uses=1]
	store ubyte %1286, ubyte* %817
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %364		; <ubyte*>:818 [#uses=2]
	load ubyte* %818		; <ubyte>:1287 [#uses=2]
	add ubyte %1287, 255		; <ubyte>:1288 [#uses=1]
	store ubyte %1288, ubyte* %818
	seteq ubyte %1287, 1		; <bool>:558 [#uses=1]
	br bool %558, label %559, label %558

; <label>:559		; preds = %556, %558
	add uint %362, 106		; <uint>:368 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %368		; <ubyte*>:819 [#uses=2]
	load ubyte* %819		; <ubyte>:1289 [#uses=2]
	add ubyte %1289, 255		; <ubyte>:1290 [#uses=1]
	store ubyte %1290, ubyte* %819
	seteq ubyte %1289, 1		; <bool>:559 [#uses=1]
	br bool %559, label %561, label %560

; <label>:560		; preds = %559, %560
	add uint %362, 104		; <uint>:369 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %369		; <ubyte*>:820 [#uses=2]
	load ubyte* %820		; <ubyte>:1291 [#uses=1]
	add ubyte %1291, 255		; <ubyte>:1292 [#uses=1]
	store ubyte %1292, ubyte* %820
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %368		; <ubyte*>:821 [#uses=2]
	load ubyte* %821		; <ubyte>:1293 [#uses=2]
	add ubyte %1293, 255		; <ubyte>:1294 [#uses=1]
	store ubyte %1294, ubyte* %821
	seteq ubyte %1293, 1		; <bool>:560 [#uses=1]
	br bool %560, label %561, label %560

; <label>:561		; preds = %559, %560
	add uint %362, 98		; <uint>:370 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %370		; <ubyte*>:822 [#uses=1]
	load ubyte* %822		; <ubyte>:1295 [#uses=1]
	seteq ubyte %1295, 0		; <bool>:561 [#uses=1]
	br bool %561, label %563, label %562

; <label>:562		; preds = %561, %562
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %370		; <ubyte*>:823 [#uses=2]
	load ubyte* %823		; <ubyte>:1296 [#uses=2]
	add ubyte %1296, 255		; <ubyte>:1297 [#uses=1]
	store ubyte %1297, ubyte* %823
	seteq ubyte %1296, 1		; <bool>:562 [#uses=1]
	br bool %562, label %563, label %562

; <label>:563		; preds = %561, %562
	add uint %362, 104		; <uint>:371 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %371		; <ubyte*>:824 [#uses=1]
	load ubyte* %824		; <ubyte>:1298 [#uses=1]
	seteq ubyte %1298, 0		; <bool>:563 [#uses=1]
	br bool %563, label %565, label %564

; <label>:564		; preds = %563, %564
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %370		; <ubyte*>:825 [#uses=2]
	load ubyte* %825		; <ubyte>:1299 [#uses=1]
	add ubyte %1299, 1		; <ubyte>:1300 [#uses=1]
	store ubyte %1300, ubyte* %825
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %371		; <ubyte*>:826 [#uses=2]
	load ubyte* %826		; <ubyte>:1301 [#uses=2]
	add ubyte %1301, 255		; <ubyte>:1302 [#uses=1]
	store ubyte %1302, ubyte* %826
	seteq ubyte %1301, 1		; <bool>:564 [#uses=1]
	br bool %564, label %565, label %564

; <label>:565		; preds = %563, %564
	add uint %362, 4294967295		; <uint>:372 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %372		; <ubyte*>:827 [#uses=2]
	load ubyte* %827		; <ubyte>:1303 [#uses=1]
	add ubyte %1303, 3		; <ubyte>:1304 [#uses=1]
	store ubyte %1304, ubyte* %827
	add uint %362, 1		; <uint>:373 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %373		; <ubyte*>:828 [#uses=1]
	load ubyte* %828		; <ubyte>:1305 [#uses=1]
	seteq ubyte %1305, 0		; <bool>:565 [#uses=1]
	br bool %565, label %557, label %556

; <label>:566		; preds = %5, %569
	phi uint [ %7, %5 ], [ %381, %569 ]		; <uint>:374 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %374		; <ubyte*>:829 [#uses=2]
	load ubyte* %829		; <ubyte>:1306 [#uses=1]
	add ubyte %1306, 255		; <ubyte>:1307 [#uses=1]
	store ubyte %1307, ubyte* %829
	add uint %374, 4294967292		; <uint>:375 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %375		; <ubyte*>:830 [#uses=1]
	load ubyte* %830		; <ubyte>:1308 [#uses=1]
	seteq ubyte %1308, 0		; <bool>:566 [#uses=1]
	br bool %566, label %569, label %568

; <label>:567		; preds = %5, %569
	phi uint [ %7, %5 ], [ %381, %569 ]		; <uint>:376 [#uses=1]
	add uint %376, 4294967295		; <uint>:377 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %377		; <ubyte*>:831 [#uses=1]
	load ubyte* %831		; <ubyte>:1309 [#uses=1]
	seteq ubyte %1309, 0		; <bool>:567 [#uses=1]
	br bool %567, label %3, label %2

; <label>:568		; preds = %566, %571
	phi uint [ %375, %566 ], [ %384, %571 ]		; <uint>:378 [#uses=4]
	add uint %378, 1		; <uint>:379 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %379		; <ubyte*>:832 [#uses=2]
	load ubyte* %832		; <ubyte>:1310 [#uses=1]
	add ubyte %1310, 1		; <ubyte>:1311 [#uses=1]
	store ubyte %1311, ubyte* %832
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %378		; <ubyte*>:833 [#uses=1]
	load ubyte* %833		; <ubyte>:1312 [#uses=1]
	seteq ubyte %1312, 0		; <bool>:568 [#uses=1]
	br bool %568, label %571, label %570

; <label>:569		; preds = %566, %571
	phi uint [ %375, %566 ], [ %384, %571 ]		; <uint>:380 [#uses=1]
	add uint %380, 4294967295		; <uint>:381 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %381		; <ubyte*>:834 [#uses=1]
	load ubyte* %834		; <ubyte>:1313 [#uses=1]
	seteq ubyte %1313, 0		; <bool>:569 [#uses=1]
	br bool %569, label %567, label %566

; <label>:570		; preds = %568, %2077
	phi uint [ %378, %568 ], [ %1305, %2077 ]		; <uint>:382 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %382		; <ubyte*>:835 [#uses=2]
	load ubyte* %835		; <ubyte>:1314 [#uses=2]
	add ubyte %1314, 255		; <ubyte>:1315 [#uses=1]
	store ubyte %1315, ubyte* %835
	seteq ubyte %1314, 1		; <bool>:570 [#uses=1]
	br bool %570, label %573, label %572

; <label>:571		; preds = %568, %2077
	phi uint [ %378, %568 ], [ %1305, %2077 ]		; <uint>:383 [#uses=1]
	add uint %383, 4294967295		; <uint>:384 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %384		; <ubyte*>:836 [#uses=1]
	load ubyte* %836		; <ubyte>:1316 [#uses=1]
	seteq ubyte %1316, 0		; <bool>:571 [#uses=1]
	br bool %571, label %569, label %568

; <label>:572		; preds = %570, %1649
	phi uint [ %382, %570 ], [ %1063, %1649 ]		; <uint>:385 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %385		; <ubyte*>:837 [#uses=2]
	load ubyte* %837		; <ubyte>:1317 [#uses=2]
	add ubyte %1317, 255		; <ubyte>:1318 [#uses=1]
	store ubyte %1318, ubyte* %837
	seteq ubyte %1317, 1		; <bool>:572 [#uses=1]
	br bool %572, label %575, label %574

; <label>:573		; preds = %570, %1649
	phi uint [ %382, %570 ], [ %1063, %1649 ]		; <uint>:386 [#uses=1]
	add uint %386, 1		; <uint>:387 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %387		; <ubyte*>:838 [#uses=1]
	load ubyte* %838		; <ubyte>:1319 [#uses=1]
	seteq ubyte %1319, 0		; <bool>:573 [#uses=1]
	br bool %573, label %2077, label %2076

; <label>:574		; preds = %572, %1593
	phi uint [ %385, %572 ], [ %1038, %1593 ]		; <uint>:388 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %388		; <ubyte*>:839 [#uses=2]
	load ubyte* %839		; <ubyte>:1320 [#uses=2]
	add ubyte %1320, 255		; <ubyte>:1321 [#uses=1]
	store ubyte %1321, ubyte* %839
	seteq ubyte %1320, 1		; <bool>:574 [#uses=1]
	br bool %574, label %577, label %576

; <label>:575		; preds = %572, %1593
	phi uint [ %385, %572 ], [ %1038, %1593 ]		; <uint>:389 [#uses=1]
	add uint %389, 1		; <uint>:390 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %390		; <ubyte*>:840 [#uses=1]
	load ubyte* %840		; <ubyte>:1322 [#uses=1]
	seteq ubyte %1322, 0		; <bool>:575 [#uses=1]
	br bool %575, label %1649, label %1648

; <label>:576		; preds = %574, %1397
	phi uint [ %388, %574 ], [ %924, %1397 ]		; <uint>:391 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %391		; <ubyte*>:841 [#uses=2]
	load ubyte* %841		; <ubyte>:1323 [#uses=2]
	add ubyte %1323, 255		; <ubyte>:1324 [#uses=1]
	store ubyte %1324, ubyte* %841
	seteq ubyte %1323, 1		; <bool>:576 [#uses=1]
	br bool %576, label %579, label %578

; <label>:577		; preds = %574, %1397
	phi uint [ %388, %574 ], [ %924, %1397 ]		; <uint>:392 [#uses=1]
	add uint %392, 1		; <uint>:393 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %393		; <ubyte*>:842 [#uses=1]
	load ubyte* %842		; <ubyte>:1325 [#uses=1]
	seteq ubyte %1325, 0		; <bool>:577 [#uses=1]
	br bool %577, label %1593, label %1592

; <label>:578		; preds = %576, %1395
	phi uint [ %391, %576 ], [ %920, %1395 ]		; <uint>:394 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %394		; <ubyte*>:843 [#uses=2]
	load ubyte* %843		; <ubyte>:1326 [#uses=2]
	add ubyte %1326, 255		; <ubyte>:1327 [#uses=1]
	store ubyte %1327, ubyte* %843
	seteq ubyte %1326, 1		; <bool>:578 [#uses=1]
	br bool %578, label %581, label %580

; <label>:579		; preds = %576, %1395
	phi uint [ %391, %576 ], [ %920, %1395 ]		; <uint>:395 [#uses=1]
	add uint %395, 1		; <uint>:396 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %396		; <ubyte*>:844 [#uses=1]
	load ubyte* %844		; <ubyte>:1328 [#uses=1]
	seteq ubyte %1328, 0		; <bool>:579 [#uses=1]
	br bool %579, label %1397, label %1396

; <label>:580		; preds = %578, %1381
	phi uint [ %394, %578 ], [ %909, %1381 ]		; <uint>:397 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %397		; <ubyte*>:845 [#uses=2]
	load ubyte* %845		; <ubyte>:1329 [#uses=2]
	add ubyte %1329, 255		; <ubyte>:1330 [#uses=1]
	store ubyte %1330, ubyte* %845
	seteq ubyte %1329, 1		; <bool>:580 [#uses=1]
	br bool %580, label %583, label %582

; <label>:581		; preds = %578, %1381
	phi uint [ %394, %578 ], [ %909, %1381 ]		; <uint>:398 [#uses=1]
	add uint %398, 1		; <uint>:399 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %399		; <ubyte*>:846 [#uses=1]
	load ubyte* %846		; <ubyte>:1331 [#uses=1]
	seteq ubyte %1331, 0		; <bool>:581 [#uses=1]
	br bool %581, label %1395, label %1394

; <label>:582		; preds = %580, %1009
	phi uint [ %397, %580 ], [ %687, %1009 ]		; <uint>:400 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %400		; <ubyte*>:847 [#uses=2]
	load ubyte* %847		; <ubyte>:1332 [#uses=2]
	add ubyte %1332, 255		; <ubyte>:1333 [#uses=1]
	store ubyte %1333, ubyte* %847
	seteq ubyte %1332, 1		; <bool>:582 [#uses=1]
	br bool %582, label %585, label %584

; <label>:583		; preds = %580, %1009
	phi uint [ %397, %580 ], [ %687, %1009 ]		; <uint>:401 [#uses=1]
	add uint %401, 1		; <uint>:402 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %402		; <ubyte*>:848 [#uses=1]
	load ubyte* %848		; <ubyte>:1334 [#uses=1]
	seteq ubyte %1334, 0		; <bool>:583 [#uses=1]
	br bool %583, label %1381, label %1380

; <label>:584		; preds = %582, %951
	phi uint [ %400, %582 ], [ %662, %951 ]		; <uint>:403 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %403		; <ubyte*>:849 [#uses=2]
	load ubyte* %849		; <ubyte>:1335 [#uses=2]
	add ubyte %1335, 255		; <ubyte>:1336 [#uses=1]
	store ubyte %1336, ubyte* %849
	seteq ubyte %1335, 1		; <bool>:584 [#uses=1]
	br bool %584, label %587, label %586

; <label>:585		; preds = %582, %951
	phi uint [ %400, %582 ], [ %662, %951 ]		; <uint>:404 [#uses=1]
	add uint %404, 1		; <uint>:405 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %405		; <ubyte*>:850 [#uses=1]
	load ubyte* %850		; <ubyte>:1337 [#uses=1]
	seteq ubyte %1337, 0		; <bool>:585 [#uses=1]
	br bool %585, label %1009, label %1008

; <label>:586		; preds = %584, %847
	phi uint [ %403, %584 ], [ %586, %847 ]		; <uint>:406 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %406		; <ubyte*>:851 [#uses=2]
	load ubyte* %851		; <ubyte>:1338 [#uses=2]
	add ubyte %1338, 255		; <ubyte>:1339 [#uses=1]
	store ubyte %1339, ubyte* %851
	seteq ubyte %1338, 1		; <bool>:586 [#uses=1]
	br bool %586, label %589, label %588

; <label>:587		; preds = %584, %847
	phi uint [ %403, %584 ], [ %586, %847 ]		; <uint>:407 [#uses=1]
	add uint %407, 1		; <uint>:408 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %408		; <ubyte*>:852 [#uses=1]
	load ubyte* %852		; <ubyte>:1340 [#uses=1]
	seteq ubyte %1340, 0		; <bool>:587 [#uses=1]
	br bool %587, label %951, label %950

; <label>:588		; preds = %586, %781
	phi uint [ %406, %586 ], [ %549, %781 ]		; <uint>:409 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %409		; <ubyte*>:853 [#uses=2]
	load ubyte* %853		; <ubyte>:1341 [#uses=2]
	add ubyte %1341, 255		; <ubyte>:1342 [#uses=1]
	store ubyte %1342, ubyte* %853
	seteq ubyte %1341, 1		; <bool>:588 [#uses=1]
	br bool %588, label %591, label %590

; <label>:589		; preds = %586, %781
	phi uint [ %406, %586 ], [ %549, %781 ]		; <uint>:410 [#uses=1]
	add uint %410, 1		; <uint>:411 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %411		; <ubyte*>:854 [#uses=1]
	load ubyte* %854		; <ubyte>:1343 [#uses=1]
	seteq ubyte %1343, 0		; <bool>:589 [#uses=1]
	br bool %589, label %847, label %846

; <label>:590		; preds = %588, %771
	phi uint [ %409, %588 ], [ %541, %771 ]		; <uint>:412 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %412		; <ubyte*>:855 [#uses=2]
	load ubyte* %855		; <ubyte>:1344 [#uses=2]
	add ubyte %1344, 255		; <ubyte>:1345 [#uses=1]
	store ubyte %1345, ubyte* %855
	seteq ubyte %1344, 1		; <bool>:590 [#uses=1]
	br bool %590, label %593, label %592

; <label>:591		; preds = %588, %771
	phi uint [ %409, %588 ], [ %541, %771 ]		; <uint>:413 [#uses=1]
	add uint %413, 1		; <uint>:414 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %414		; <ubyte*>:856 [#uses=1]
	load ubyte* %856		; <ubyte>:1346 [#uses=1]
	seteq ubyte %1346, 0		; <bool>:591 [#uses=1]
	br bool %591, label %781, label %780

; <label>:592		; preds = %590, %667
	phi uint [ %412, %590 ], [ %465, %667 ]		; <uint>:415 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %415		; <ubyte*>:857 [#uses=2]
	load ubyte* %857		; <ubyte>:1347 [#uses=2]
	add ubyte %1347, 255		; <ubyte>:1348 [#uses=1]
	store ubyte %1348, ubyte* %857
	seteq ubyte %1347, 1		; <bool>:592 [#uses=1]
	br bool %592, label %595, label %594

; <label>:593		; preds = %590, %667
	phi uint [ %412, %590 ], [ %465, %667 ]		; <uint>:416 [#uses=1]
	add uint %416, 1		; <uint>:417 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %417		; <ubyte*>:858 [#uses=1]
	load ubyte* %858		; <ubyte>:1349 [#uses=1]
	seteq ubyte %1349, 0		; <bool>:593 [#uses=1]
	br bool %593, label %771, label %770

; <label>:594		; preds = %592, %601
	phi uint [ %415, %592 ], [ %428, %601 ]		; <uint>:418 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %418		; <ubyte*>:859 [#uses=2]
	load ubyte* %859		; <ubyte>:1350 [#uses=2]
	add ubyte %1350, 255		; <ubyte>:1351 [#uses=1]
	store ubyte %1351, ubyte* %859
	seteq ubyte %1350, 1		; <bool>:594 [#uses=1]
	br bool %594, label %597, label %596

; <label>:595		; preds = %592, %601
	phi uint [ %415, %592 ], [ %428, %601 ]		; <uint>:419 [#uses=1]
	add uint %419, 1		; <uint>:420 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %420		; <ubyte*>:860 [#uses=1]
	load ubyte* %860		; <ubyte>:1352 [#uses=1]
	seteq ubyte %1352, 0		; <bool>:595 [#uses=1]
	br bool %595, label %667, label %666

; <label>:596		; preds = %594, %599
	phi uint [ %418, %594 ], [ %424, %599 ]		; <uint>:421 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %421		; <ubyte*>:861 [#uses=1]
	load ubyte* %861		; <ubyte>:1353 [#uses=1]
	seteq ubyte %1353, 0		; <bool>:596 [#uses=1]
	br bool %596, label %599, label %598

; <label>:597		; preds = %594, %599
	phi uint [ %418, %594 ], [ %424, %599 ]		; <uint>:422 [#uses=1]
	add uint %422, 1		; <uint>:423 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %423		; <ubyte*>:862 [#uses=1]
	load ubyte* %862		; <ubyte>:1354 [#uses=1]
	seteq ubyte %1354, 0		; <bool>:597 [#uses=1]
	br bool %597, label %601, label %600

; <label>:598		; preds = %596, %598
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %421		; <ubyte*>:863 [#uses=2]
	load ubyte* %863		; <ubyte>:1355 [#uses=2]
	add ubyte %1355, 255		; <ubyte>:1356 [#uses=1]
	store ubyte %1356, ubyte* %863
	seteq ubyte %1355, 1		; <bool>:598 [#uses=1]
	br bool %598, label %599, label %598

; <label>:599		; preds = %596, %598
	add uint %421, 1		; <uint>:424 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %424		; <ubyte*>:864 [#uses=2]
	load ubyte* %864		; <ubyte>:1357 [#uses=2]
	add ubyte %1357, 255		; <ubyte>:1358 [#uses=1]
	store ubyte %1358, ubyte* %864
	seteq ubyte %1357, 1		; <bool>:599 [#uses=1]
	br bool %599, label %597, label %596

; <label>:600		; preds = %597, %665
	phi uint [ %423, %597 ], [ %461, %665 ]		; <uint>:425 [#uses=35]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %425		; <ubyte*>:865 [#uses=2]
	load ubyte* %865		; <ubyte>:1359 [#uses=1]
	add ubyte %1359, 255		; <ubyte>:1360 [#uses=1]
	store ubyte %1360, ubyte* %865
	add uint %425, 4294967090		; <uint>:426 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %426		; <ubyte*>:866 [#uses=1]
	load ubyte* %866		; <ubyte>:1361 [#uses=1]
	seteq ubyte %1361, 0		; <bool>:600 [#uses=1]
	br bool %600, label %603, label %602

; <label>:601		; preds = %597, %665
	phi uint [ %423, %597 ], [ %461, %665 ]		; <uint>:427 [#uses=1]
	add uint %427, 4294967295		; <uint>:428 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %428		; <ubyte*>:867 [#uses=1]
	load ubyte* %867		; <ubyte>:1362 [#uses=1]
	seteq ubyte %1362, 0		; <bool>:601 [#uses=1]
	br bool %601, label %595, label %594

; <label>:602		; preds = %600, %602
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %426		; <ubyte*>:868 [#uses=2]
	load ubyte* %868		; <ubyte>:1363 [#uses=2]
	add ubyte %1363, 255		; <ubyte>:1364 [#uses=1]
	store ubyte %1364, ubyte* %868
	seteq ubyte %1363, 1		; <bool>:602 [#uses=1]
	br bool %602, label %603, label %602

; <label>:603		; preds = %600, %602
	add uint %425, 4294967207		; <uint>:429 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %429		; <ubyte*>:869 [#uses=1]
	load ubyte* %869		; <ubyte>:1365 [#uses=1]
	seteq ubyte %1365, 0		; <bool>:603 [#uses=1]
	br bool %603, label %605, label %604

; <label>:604		; preds = %603, %604
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %426		; <ubyte*>:870 [#uses=2]
	load ubyte* %870		; <ubyte>:1366 [#uses=1]
	add ubyte %1366, 1		; <ubyte>:1367 [#uses=1]
	store ubyte %1367, ubyte* %870
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %429		; <ubyte*>:871 [#uses=2]
	load ubyte* %871		; <ubyte>:1368 [#uses=2]
	add ubyte %1368, 255		; <ubyte>:1369 [#uses=1]
	store ubyte %1369, ubyte* %871
	seteq ubyte %1368, 1		; <bool>:604 [#uses=1]
	br bool %604, label %605, label %604

; <label>:605		; preds = %603, %604
	add uint %425, 4294967096		; <uint>:430 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %430		; <ubyte*>:872 [#uses=1]
	load ubyte* %872		; <ubyte>:1370 [#uses=1]
	seteq ubyte %1370, 0		; <bool>:605 [#uses=1]
	br bool %605, label %607, label %606

; <label>:606		; preds = %605, %606
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %430		; <ubyte*>:873 [#uses=2]
	load ubyte* %873		; <ubyte>:1371 [#uses=2]
	add ubyte %1371, 255		; <ubyte>:1372 [#uses=1]
	store ubyte %1372, ubyte* %873
	seteq ubyte %1371, 1		; <bool>:606 [#uses=1]
	br bool %606, label %607, label %606

; <label>:607		; preds = %605, %606
	add uint %425, 4294967213		; <uint>:431 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %431		; <ubyte*>:874 [#uses=1]
	load ubyte* %874		; <ubyte>:1373 [#uses=1]
	seteq ubyte %1373, 0		; <bool>:607 [#uses=1]
	br bool %607, label %609, label %608

; <label>:608		; preds = %607, %608
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %430		; <ubyte*>:875 [#uses=2]
	load ubyte* %875		; <ubyte>:1374 [#uses=1]
	add ubyte %1374, 1		; <ubyte>:1375 [#uses=1]
	store ubyte %1375, ubyte* %875
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %431		; <ubyte*>:876 [#uses=2]
	load ubyte* %876		; <ubyte>:1376 [#uses=2]
	add ubyte %1376, 255		; <ubyte>:1377 [#uses=1]
	store ubyte %1377, ubyte* %876
	seteq ubyte %1376, 1		; <bool>:608 [#uses=1]
	br bool %608, label %609, label %608

; <label>:609		; preds = %607, %608
	add uint %425, 4294967102		; <uint>:432 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %432		; <ubyte*>:877 [#uses=1]
	load ubyte* %877		; <ubyte>:1378 [#uses=1]
	seteq ubyte %1378, 0		; <bool>:609 [#uses=1]
	br bool %609, label %611, label %610

; <label>:610		; preds = %609, %610
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %432		; <ubyte*>:878 [#uses=2]
	load ubyte* %878		; <ubyte>:1379 [#uses=2]
	add ubyte %1379, 255		; <ubyte>:1380 [#uses=1]
	store ubyte %1380, ubyte* %878
	seteq ubyte %1379, 1		; <bool>:610 [#uses=1]
	br bool %610, label %611, label %610

; <label>:611		; preds = %609, %610
	add uint %425, 4294967219		; <uint>:433 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %433		; <ubyte*>:879 [#uses=1]
	load ubyte* %879		; <ubyte>:1381 [#uses=1]
	seteq ubyte %1381, 0		; <bool>:611 [#uses=1]
	br bool %611, label %613, label %612

; <label>:612		; preds = %611, %612
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %432		; <ubyte*>:880 [#uses=2]
	load ubyte* %880		; <ubyte>:1382 [#uses=1]
	add ubyte %1382, 1		; <ubyte>:1383 [#uses=1]
	store ubyte %1383, ubyte* %880
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %433		; <ubyte*>:881 [#uses=2]
	load ubyte* %881		; <ubyte>:1384 [#uses=2]
	add ubyte %1384, 255		; <ubyte>:1385 [#uses=1]
	store ubyte %1385, ubyte* %881
	seteq ubyte %1384, 1		; <bool>:612 [#uses=1]
	br bool %612, label %613, label %612

; <label>:613		; preds = %611, %612
	add uint %425, 4294967108		; <uint>:434 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %434		; <ubyte*>:882 [#uses=1]
	load ubyte* %882		; <ubyte>:1386 [#uses=1]
	seteq ubyte %1386, 0		; <bool>:613 [#uses=1]
	br bool %613, label %615, label %614

; <label>:614		; preds = %613, %614
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %434		; <ubyte*>:883 [#uses=2]
	load ubyte* %883		; <ubyte>:1387 [#uses=2]
	add ubyte %1387, 255		; <ubyte>:1388 [#uses=1]
	store ubyte %1388, ubyte* %883
	seteq ubyte %1387, 1		; <bool>:614 [#uses=1]
	br bool %614, label %615, label %614

; <label>:615		; preds = %613, %614
	add uint %425, 4294967225		; <uint>:435 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %435		; <ubyte*>:884 [#uses=1]
	load ubyte* %884		; <ubyte>:1389 [#uses=1]
	seteq ubyte %1389, 0		; <bool>:615 [#uses=1]
	br bool %615, label %617, label %616

; <label>:616		; preds = %615, %616
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %434		; <ubyte*>:885 [#uses=2]
	load ubyte* %885		; <ubyte>:1390 [#uses=1]
	add ubyte %1390, 1		; <ubyte>:1391 [#uses=1]
	store ubyte %1391, ubyte* %885
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %435		; <ubyte*>:886 [#uses=2]
	load ubyte* %886		; <ubyte>:1392 [#uses=2]
	add ubyte %1392, 255		; <ubyte>:1393 [#uses=1]
	store ubyte %1393, ubyte* %886
	seteq ubyte %1392, 1		; <bool>:616 [#uses=1]
	br bool %616, label %617, label %616

; <label>:617		; preds = %615, %616
	add uint %425, 4294967114		; <uint>:436 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %436		; <ubyte*>:887 [#uses=1]
	load ubyte* %887		; <ubyte>:1394 [#uses=1]
	seteq ubyte %1394, 0		; <bool>:617 [#uses=1]
	br bool %617, label %619, label %618

; <label>:618		; preds = %617, %618
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %436		; <ubyte*>:888 [#uses=2]
	load ubyte* %888		; <ubyte>:1395 [#uses=2]
	add ubyte %1395, 255		; <ubyte>:1396 [#uses=1]
	store ubyte %1396, ubyte* %888
	seteq ubyte %1395, 1		; <bool>:618 [#uses=1]
	br bool %618, label %619, label %618

; <label>:619		; preds = %617, %618
	add uint %425, 4294967231		; <uint>:437 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %437		; <ubyte*>:889 [#uses=1]
	load ubyte* %889		; <ubyte>:1397 [#uses=1]
	seteq ubyte %1397, 0		; <bool>:619 [#uses=1]
	br bool %619, label %621, label %620

; <label>:620		; preds = %619, %620
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %436		; <ubyte*>:890 [#uses=2]
	load ubyte* %890		; <ubyte>:1398 [#uses=1]
	add ubyte %1398, 1		; <ubyte>:1399 [#uses=1]
	store ubyte %1399, ubyte* %890
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %437		; <ubyte*>:891 [#uses=2]
	load ubyte* %891		; <ubyte>:1400 [#uses=2]
	add ubyte %1400, 255		; <ubyte>:1401 [#uses=1]
	store ubyte %1401, ubyte* %891
	seteq ubyte %1400, 1		; <bool>:620 [#uses=1]
	br bool %620, label %621, label %620

; <label>:621		; preds = %619, %620
	add uint %425, 4294967120		; <uint>:438 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %438		; <ubyte*>:892 [#uses=1]
	load ubyte* %892		; <ubyte>:1402 [#uses=1]
	seteq ubyte %1402, 0		; <bool>:621 [#uses=1]
	br bool %621, label %623, label %622

; <label>:622		; preds = %621, %622
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %438		; <ubyte*>:893 [#uses=2]
	load ubyte* %893		; <ubyte>:1403 [#uses=2]
	add ubyte %1403, 255		; <ubyte>:1404 [#uses=1]
	store ubyte %1404, ubyte* %893
	seteq ubyte %1403, 1		; <bool>:622 [#uses=1]
	br bool %622, label %623, label %622

; <label>:623		; preds = %621, %622
	add uint %425, 4294967237		; <uint>:439 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %439		; <ubyte*>:894 [#uses=1]
	load ubyte* %894		; <ubyte>:1405 [#uses=1]
	seteq ubyte %1405, 0		; <bool>:623 [#uses=1]
	br bool %623, label %625, label %624

; <label>:624		; preds = %623, %624
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %438		; <ubyte*>:895 [#uses=2]
	load ubyte* %895		; <ubyte>:1406 [#uses=1]
	add ubyte %1406, 1		; <ubyte>:1407 [#uses=1]
	store ubyte %1407, ubyte* %895
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %439		; <ubyte*>:896 [#uses=2]
	load ubyte* %896		; <ubyte>:1408 [#uses=2]
	add ubyte %1408, 255		; <ubyte>:1409 [#uses=1]
	store ubyte %1409, ubyte* %896
	seteq ubyte %1408, 1		; <bool>:624 [#uses=1]
	br bool %624, label %625, label %624

; <label>:625		; preds = %623, %624
	add uint %425, 4294967126		; <uint>:440 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %440		; <ubyte*>:897 [#uses=1]
	load ubyte* %897		; <ubyte>:1410 [#uses=1]
	seteq ubyte %1410, 0		; <bool>:625 [#uses=1]
	br bool %625, label %627, label %626

; <label>:626		; preds = %625, %626
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %440		; <ubyte*>:898 [#uses=2]
	load ubyte* %898		; <ubyte>:1411 [#uses=2]
	add ubyte %1411, 255		; <ubyte>:1412 [#uses=1]
	store ubyte %1412, ubyte* %898
	seteq ubyte %1411, 1		; <bool>:626 [#uses=1]
	br bool %626, label %627, label %626

; <label>:627		; preds = %625, %626
	add uint %425, 4294967243		; <uint>:441 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %441		; <ubyte*>:899 [#uses=1]
	load ubyte* %899		; <ubyte>:1413 [#uses=1]
	seteq ubyte %1413, 0		; <bool>:627 [#uses=1]
	br bool %627, label %629, label %628

; <label>:628		; preds = %627, %628
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %440		; <ubyte*>:900 [#uses=2]
	load ubyte* %900		; <ubyte>:1414 [#uses=1]
	add ubyte %1414, 1		; <ubyte>:1415 [#uses=1]
	store ubyte %1415, ubyte* %900
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %441		; <ubyte*>:901 [#uses=2]
	load ubyte* %901		; <ubyte>:1416 [#uses=2]
	add ubyte %1416, 255		; <ubyte>:1417 [#uses=1]
	store ubyte %1417, ubyte* %901
	seteq ubyte %1416, 1		; <bool>:628 [#uses=1]
	br bool %628, label %629, label %628

; <label>:629		; preds = %627, %628
	add uint %425, 4294967132		; <uint>:442 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %442		; <ubyte*>:902 [#uses=1]
	load ubyte* %902		; <ubyte>:1418 [#uses=1]
	seteq ubyte %1418, 0		; <bool>:629 [#uses=1]
	br bool %629, label %631, label %630

; <label>:630		; preds = %629, %630
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %442		; <ubyte*>:903 [#uses=2]
	load ubyte* %903		; <ubyte>:1419 [#uses=2]
	add ubyte %1419, 255		; <ubyte>:1420 [#uses=1]
	store ubyte %1420, ubyte* %903
	seteq ubyte %1419, 1		; <bool>:630 [#uses=1]
	br bool %630, label %631, label %630

; <label>:631		; preds = %629, %630
	add uint %425, 4294967249		; <uint>:443 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %443		; <ubyte*>:904 [#uses=1]
	load ubyte* %904		; <ubyte>:1421 [#uses=1]
	seteq ubyte %1421, 0		; <bool>:631 [#uses=1]
	br bool %631, label %633, label %632

; <label>:632		; preds = %631, %632
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %442		; <ubyte*>:905 [#uses=2]
	load ubyte* %905		; <ubyte>:1422 [#uses=1]
	add ubyte %1422, 1		; <ubyte>:1423 [#uses=1]
	store ubyte %1423, ubyte* %905
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %443		; <ubyte*>:906 [#uses=2]
	load ubyte* %906		; <ubyte>:1424 [#uses=2]
	add ubyte %1424, 255		; <ubyte>:1425 [#uses=1]
	store ubyte %1425, ubyte* %906
	seteq ubyte %1424, 1		; <bool>:632 [#uses=1]
	br bool %632, label %633, label %632

; <label>:633		; preds = %631, %632
	add uint %425, 4294967138		; <uint>:444 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %444		; <ubyte*>:907 [#uses=1]
	load ubyte* %907		; <ubyte>:1426 [#uses=1]
	seteq ubyte %1426, 0		; <bool>:633 [#uses=1]
	br bool %633, label %635, label %634

; <label>:634		; preds = %633, %634
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %444		; <ubyte*>:908 [#uses=2]
	load ubyte* %908		; <ubyte>:1427 [#uses=2]
	add ubyte %1427, 255		; <ubyte>:1428 [#uses=1]
	store ubyte %1428, ubyte* %908
	seteq ubyte %1427, 1		; <bool>:634 [#uses=1]
	br bool %634, label %635, label %634

; <label>:635		; preds = %633, %634
	add uint %425, 4294967255		; <uint>:445 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %445		; <ubyte*>:909 [#uses=1]
	load ubyte* %909		; <ubyte>:1429 [#uses=1]
	seteq ubyte %1429, 0		; <bool>:635 [#uses=1]
	br bool %635, label %637, label %636

; <label>:636		; preds = %635, %636
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %444		; <ubyte*>:910 [#uses=2]
	load ubyte* %910		; <ubyte>:1430 [#uses=1]
	add ubyte %1430, 1		; <ubyte>:1431 [#uses=1]
	store ubyte %1431, ubyte* %910
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %445		; <ubyte*>:911 [#uses=2]
	load ubyte* %911		; <ubyte>:1432 [#uses=2]
	add ubyte %1432, 255		; <ubyte>:1433 [#uses=1]
	store ubyte %1433, ubyte* %911
	seteq ubyte %1432, 1		; <bool>:636 [#uses=1]
	br bool %636, label %637, label %636

; <label>:637		; preds = %635, %636
	add uint %425, 4294967144		; <uint>:446 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %446		; <ubyte*>:912 [#uses=1]
	load ubyte* %912		; <ubyte>:1434 [#uses=1]
	seteq ubyte %1434, 0		; <bool>:637 [#uses=1]
	br bool %637, label %639, label %638

; <label>:638		; preds = %637, %638
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %446		; <ubyte*>:913 [#uses=2]
	load ubyte* %913		; <ubyte>:1435 [#uses=2]
	add ubyte %1435, 255		; <ubyte>:1436 [#uses=1]
	store ubyte %1436, ubyte* %913
	seteq ubyte %1435, 1		; <bool>:638 [#uses=1]
	br bool %638, label %639, label %638

; <label>:639		; preds = %637, %638
	add uint %425, 4294967261		; <uint>:447 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %447		; <ubyte*>:914 [#uses=1]
	load ubyte* %914		; <ubyte>:1437 [#uses=1]
	seteq ubyte %1437, 0		; <bool>:639 [#uses=1]
	br bool %639, label %641, label %640

; <label>:640		; preds = %639, %640
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %446		; <ubyte*>:915 [#uses=2]
	load ubyte* %915		; <ubyte>:1438 [#uses=1]
	add ubyte %1438, 1		; <ubyte>:1439 [#uses=1]
	store ubyte %1439, ubyte* %915
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %447		; <ubyte*>:916 [#uses=2]
	load ubyte* %916		; <ubyte>:1440 [#uses=2]
	add ubyte %1440, 255		; <ubyte>:1441 [#uses=1]
	store ubyte %1441, ubyte* %916
	seteq ubyte %1440, 1		; <bool>:640 [#uses=1]
	br bool %640, label %641, label %640

; <label>:641		; preds = %639, %640
	add uint %425, 4294967150		; <uint>:448 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %448		; <ubyte*>:917 [#uses=1]
	load ubyte* %917		; <ubyte>:1442 [#uses=1]
	seteq ubyte %1442, 0		; <bool>:641 [#uses=1]
	br bool %641, label %643, label %642

; <label>:642		; preds = %641, %642
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %448		; <ubyte*>:918 [#uses=2]
	load ubyte* %918		; <ubyte>:1443 [#uses=2]
	add ubyte %1443, 255		; <ubyte>:1444 [#uses=1]
	store ubyte %1444, ubyte* %918
	seteq ubyte %1443, 1		; <bool>:642 [#uses=1]
	br bool %642, label %643, label %642

; <label>:643		; preds = %641, %642
	add uint %425, 4294967267		; <uint>:449 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %449		; <ubyte*>:919 [#uses=1]
	load ubyte* %919		; <ubyte>:1445 [#uses=1]
	seteq ubyte %1445, 0		; <bool>:643 [#uses=1]
	br bool %643, label %645, label %644

; <label>:644		; preds = %643, %644
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %448		; <ubyte*>:920 [#uses=2]
	load ubyte* %920		; <ubyte>:1446 [#uses=1]
	add ubyte %1446, 1		; <ubyte>:1447 [#uses=1]
	store ubyte %1447, ubyte* %920
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %449		; <ubyte*>:921 [#uses=2]
	load ubyte* %921		; <ubyte>:1448 [#uses=2]
	add ubyte %1448, 255		; <ubyte>:1449 [#uses=1]
	store ubyte %1449, ubyte* %921
	seteq ubyte %1448, 1		; <bool>:644 [#uses=1]
	br bool %644, label %645, label %644

; <label>:645		; preds = %643, %644
	add uint %425, 4294967156		; <uint>:450 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %450		; <ubyte*>:922 [#uses=1]
	load ubyte* %922		; <ubyte>:1450 [#uses=1]
	seteq ubyte %1450, 0		; <bool>:645 [#uses=1]
	br bool %645, label %647, label %646

; <label>:646		; preds = %645, %646
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %450		; <ubyte*>:923 [#uses=2]
	load ubyte* %923		; <ubyte>:1451 [#uses=2]
	add ubyte %1451, 255		; <ubyte>:1452 [#uses=1]
	store ubyte %1452, ubyte* %923
	seteq ubyte %1451, 1		; <bool>:646 [#uses=1]
	br bool %646, label %647, label %646

; <label>:647		; preds = %645, %646
	add uint %425, 4294967273		; <uint>:451 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %451		; <ubyte*>:924 [#uses=1]
	load ubyte* %924		; <ubyte>:1453 [#uses=1]
	seteq ubyte %1453, 0		; <bool>:647 [#uses=1]
	br bool %647, label %649, label %648

; <label>:648		; preds = %647, %648
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %450		; <ubyte*>:925 [#uses=2]
	load ubyte* %925		; <ubyte>:1454 [#uses=1]
	add ubyte %1454, 1		; <ubyte>:1455 [#uses=1]
	store ubyte %1455, ubyte* %925
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %451		; <ubyte*>:926 [#uses=2]
	load ubyte* %926		; <ubyte>:1456 [#uses=2]
	add ubyte %1456, 255		; <ubyte>:1457 [#uses=1]
	store ubyte %1457, ubyte* %926
	seteq ubyte %1456, 1		; <bool>:648 [#uses=1]
	br bool %648, label %649, label %648

; <label>:649		; preds = %647, %648
	add uint %425, 4294967162		; <uint>:452 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %452		; <ubyte*>:927 [#uses=1]
	load ubyte* %927		; <ubyte>:1458 [#uses=1]
	seteq ubyte %1458, 0		; <bool>:649 [#uses=1]
	br bool %649, label %651, label %650

; <label>:650		; preds = %649, %650
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %452		; <ubyte*>:928 [#uses=2]
	load ubyte* %928		; <ubyte>:1459 [#uses=2]
	add ubyte %1459, 255		; <ubyte>:1460 [#uses=1]
	store ubyte %1460, ubyte* %928
	seteq ubyte %1459, 1		; <bool>:650 [#uses=1]
	br bool %650, label %651, label %650

; <label>:651		; preds = %649, %650
	add uint %425, 4294967279		; <uint>:453 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %453		; <ubyte*>:929 [#uses=1]
	load ubyte* %929		; <ubyte>:1461 [#uses=1]
	seteq ubyte %1461, 0		; <bool>:651 [#uses=1]
	br bool %651, label %653, label %652

; <label>:652		; preds = %651, %652
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %452		; <ubyte*>:930 [#uses=2]
	load ubyte* %930		; <ubyte>:1462 [#uses=1]
	add ubyte %1462, 1		; <ubyte>:1463 [#uses=1]
	store ubyte %1463, ubyte* %930
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %453		; <ubyte*>:931 [#uses=2]
	load ubyte* %931		; <ubyte>:1464 [#uses=2]
	add ubyte %1464, 255		; <ubyte>:1465 [#uses=1]
	store ubyte %1465, ubyte* %931
	seteq ubyte %1464, 1		; <bool>:652 [#uses=1]
	br bool %652, label %653, label %652

; <label>:653		; preds = %651, %652
	add uint %425, 4294967168		; <uint>:454 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %454		; <ubyte*>:932 [#uses=1]
	load ubyte* %932		; <ubyte>:1466 [#uses=1]
	seteq ubyte %1466, 0		; <bool>:653 [#uses=1]
	br bool %653, label %655, label %654

; <label>:654		; preds = %653, %654
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %454		; <ubyte*>:933 [#uses=2]
	load ubyte* %933		; <ubyte>:1467 [#uses=2]
	add ubyte %1467, 255		; <ubyte>:1468 [#uses=1]
	store ubyte %1468, ubyte* %933
	seteq ubyte %1467, 1		; <bool>:654 [#uses=1]
	br bool %654, label %655, label %654

; <label>:655		; preds = %653, %654
	add uint %425, 4294967285		; <uint>:455 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %455		; <ubyte*>:934 [#uses=1]
	load ubyte* %934		; <ubyte>:1469 [#uses=1]
	seteq ubyte %1469, 0		; <bool>:655 [#uses=1]
	br bool %655, label %657, label %656

; <label>:656		; preds = %655, %656
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %454		; <ubyte*>:935 [#uses=2]
	load ubyte* %935		; <ubyte>:1470 [#uses=1]
	add ubyte %1470, 1		; <ubyte>:1471 [#uses=1]
	store ubyte %1471, ubyte* %935
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %455		; <ubyte*>:936 [#uses=2]
	load ubyte* %936		; <ubyte>:1472 [#uses=2]
	add ubyte %1472, 255		; <ubyte>:1473 [#uses=1]
	store ubyte %1473, ubyte* %936
	seteq ubyte %1472, 1		; <bool>:656 [#uses=1]
	br bool %656, label %657, label %656

; <label>:657		; preds = %655, %656
	add uint %425, 4294967174		; <uint>:456 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %456		; <ubyte*>:937 [#uses=1]
	load ubyte* %937		; <ubyte>:1474 [#uses=1]
	seteq ubyte %1474, 0		; <bool>:657 [#uses=1]
	br bool %657, label %659, label %658

; <label>:658		; preds = %657, %658
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %456		; <ubyte*>:938 [#uses=2]
	load ubyte* %938		; <ubyte>:1475 [#uses=2]
	add ubyte %1475, 255		; <ubyte>:1476 [#uses=1]
	store ubyte %1476, ubyte* %938
	seteq ubyte %1475, 1		; <bool>:658 [#uses=1]
	br bool %658, label %659, label %658

; <label>:659		; preds = %657, %658
	add uint %425, 4294967291		; <uint>:457 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %457		; <ubyte*>:939 [#uses=1]
	load ubyte* %939		; <ubyte>:1477 [#uses=1]
	seteq ubyte %1477, 0		; <bool>:659 [#uses=1]
	br bool %659, label %661, label %660

; <label>:660		; preds = %659, %660
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %456		; <ubyte*>:940 [#uses=2]
	load ubyte* %940		; <ubyte>:1478 [#uses=1]
	add ubyte %1478, 1		; <ubyte>:1479 [#uses=1]
	store ubyte %1479, ubyte* %940
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %457		; <ubyte*>:941 [#uses=2]
	load ubyte* %941		; <ubyte>:1480 [#uses=2]
	add ubyte %1480, 255		; <ubyte>:1481 [#uses=1]
	store ubyte %1481, ubyte* %941
	seteq ubyte %1480, 1		; <bool>:660 [#uses=1]
	br bool %660, label %661, label %660

; <label>:661		; preds = %659, %660
	add uint %425, 4294967197		; <uint>:458 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %458		; <ubyte*>:942 [#uses=1]
	load ubyte* %942		; <ubyte>:1482 [#uses=1]
	seteq ubyte %1482, 0		; <bool>:661 [#uses=1]
	br bool %661, label %663, label %662

; <label>:662		; preds = %661, %662
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %458		; <ubyte*>:943 [#uses=2]
	load ubyte* %943		; <ubyte>:1483 [#uses=2]
	add ubyte %1483, 255		; <ubyte>:1484 [#uses=1]
	store ubyte %1484, ubyte* %943
	seteq ubyte %1483, 1		; <bool>:662 [#uses=1]
	br bool %662, label %663, label %662

; <label>:663		; preds = %661, %662
	add uint %425, 4294967195		; <uint>:459 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %459		; <ubyte*>:944 [#uses=1]
	load ubyte* %944		; <ubyte>:1485 [#uses=1]
	seteq ubyte %1485, 0		; <bool>:663 [#uses=1]
	br bool %663, label %665, label %664

; <label>:664		; preds = %663, %664
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %459		; <ubyte*>:945 [#uses=2]
	load ubyte* %945		; <ubyte>:1486 [#uses=2]
	add ubyte %1486, 255		; <ubyte>:1487 [#uses=1]
	store ubyte %1487, ubyte* %945
	seteq ubyte %1486, 1		; <bool>:664 [#uses=1]
	br bool %664, label %665, label %664

; <label>:665		; preds = %663, %664
	add uint %425, 4294967184		; <uint>:460 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %460		; <ubyte*>:946 [#uses=2]
	load ubyte* %946		; <ubyte>:1488 [#uses=1]
	add ubyte %1488, 11		; <ubyte>:1489 [#uses=1]
	store ubyte %1489, ubyte* %946
	add uint %425, 4294967186		; <uint>:461 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %461		; <ubyte*>:947 [#uses=1]
	load ubyte* %947		; <ubyte>:1490 [#uses=1]
	seteq ubyte %1490, 0		; <bool>:665 [#uses=1]
	br bool %665, label %601, label %600

; <label>:666		; preds = %595, %769
	phi uint [ %420, %595 ], [ %537, %769 ]		; <uint>:462 [#uses=74]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %462		; <ubyte*>:948 [#uses=2]
	load ubyte* %948		; <ubyte>:1491 [#uses=1]
	add ubyte %1491, 255		; <ubyte>:1492 [#uses=1]
	store ubyte %1492, ubyte* %948
	add uint %462, 10		; <uint>:463 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %463		; <ubyte*>:949 [#uses=1]
	load ubyte* %949		; <ubyte>:1493 [#uses=1]
	seteq ubyte %1493, 0		; <bool>:666 [#uses=1]
	br bool %666, label %669, label %668

; <label>:667		; preds = %595, %769
	phi uint [ %420, %595 ], [ %537, %769 ]		; <uint>:464 [#uses=1]
	add uint %464, 4294967295		; <uint>:465 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %465		; <ubyte*>:950 [#uses=1]
	load ubyte* %950		; <ubyte>:1494 [#uses=1]
	seteq ubyte %1494, 0		; <bool>:667 [#uses=1]
	br bool %667, label %593, label %592

; <label>:668		; preds = %666, %668
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %463		; <ubyte*>:951 [#uses=2]
	load ubyte* %951		; <ubyte>:1495 [#uses=2]
	add ubyte %1495, 255		; <ubyte>:1496 [#uses=1]
	store ubyte %1496, ubyte* %951
	seteq ubyte %1495, 1		; <bool>:668 [#uses=1]
	br bool %668, label %669, label %668

; <label>:669		; preds = %666, %668
	add uint %462, 4294967189		; <uint>:466 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %466		; <ubyte*>:952 [#uses=1]
	load ubyte* %952		; <ubyte>:1497 [#uses=1]
	seteq ubyte %1497, 0		; <bool>:669 [#uses=1]
	br bool %669, label %671, label %670

; <label>:670		; preds = %669, %670
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %466		; <ubyte*>:953 [#uses=2]
	load ubyte* %953		; <ubyte>:1498 [#uses=1]
	add ubyte %1498, 255		; <ubyte>:1499 [#uses=1]
	store ubyte %1499, ubyte* %953
	add uint %462, 4294967190		; <uint>:467 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %467		; <ubyte*>:954 [#uses=2]
	load ubyte* %954		; <ubyte>:1500 [#uses=1]
	add ubyte %1500, 1		; <ubyte>:1501 [#uses=1]
	store ubyte %1501, ubyte* %954
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %463		; <ubyte*>:955 [#uses=2]
	load ubyte* %955		; <ubyte>:1502 [#uses=1]
	add ubyte %1502, 1		; <ubyte>:1503 [#uses=1]
	store ubyte %1503, ubyte* %955
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %466		; <ubyte*>:956 [#uses=1]
	load ubyte* %956		; <ubyte>:1504 [#uses=1]
	seteq ubyte %1504, 0		; <bool>:670 [#uses=1]
	br bool %670, label %671, label %670

; <label>:671		; preds = %669, %670
	add uint %462, 4294967190		; <uint>:468 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %468		; <ubyte*>:957 [#uses=1]
	load ubyte* %957		; <ubyte>:1505 [#uses=1]
	seteq ubyte %1505, 0		; <bool>:671 [#uses=1]
	br bool %671, label %673, label %672

; <label>:672		; preds = %671, %672
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %466		; <ubyte*>:958 [#uses=2]
	load ubyte* %958		; <ubyte>:1506 [#uses=1]
	add ubyte %1506, 1		; <ubyte>:1507 [#uses=1]
	store ubyte %1507, ubyte* %958
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %468		; <ubyte*>:959 [#uses=2]
	load ubyte* %959		; <ubyte>:1508 [#uses=2]
	add ubyte %1508, 255		; <ubyte>:1509 [#uses=1]
	store ubyte %1509, ubyte* %959
	seteq ubyte %1508, 1		; <bool>:672 [#uses=1]
	br bool %672, label %673, label %672

; <label>:673		; preds = %671, %672
	add uint %462, 12		; <uint>:469 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %469		; <ubyte*>:960 [#uses=1]
	load ubyte* %960		; <ubyte>:1510 [#uses=1]
	seteq ubyte %1510, 0		; <bool>:673 [#uses=1]
	br bool %673, label %675, label %674

; <label>:674		; preds = %673, %674
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %469		; <ubyte*>:961 [#uses=2]
	load ubyte* %961		; <ubyte>:1511 [#uses=2]
	add ubyte %1511, 255		; <ubyte>:1512 [#uses=1]
	store ubyte %1512, ubyte* %961
	seteq ubyte %1511, 1		; <bool>:674 [#uses=1]
	br bool %674, label %675, label %674

; <label>:675		; preds = %673, %674
	add uint %462, 6		; <uint>:470 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %470		; <ubyte*>:962 [#uses=1]
	load ubyte* %962		; <ubyte>:1513 [#uses=1]
	seteq ubyte %1513, 0		; <bool>:675 [#uses=1]
	br bool %675, label %677, label %676

; <label>:676		; preds = %675, %676
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %470		; <ubyte*>:963 [#uses=2]
	load ubyte* %963		; <ubyte>:1514 [#uses=1]
	add ubyte %1514, 255		; <ubyte>:1515 [#uses=1]
	store ubyte %1515, ubyte* %963
	add uint %462, 7		; <uint>:471 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %471		; <ubyte*>:964 [#uses=2]
	load ubyte* %964		; <ubyte>:1516 [#uses=1]
	add ubyte %1516, 1		; <ubyte>:1517 [#uses=1]
	store ubyte %1517, ubyte* %964
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %469		; <ubyte*>:965 [#uses=2]
	load ubyte* %965		; <ubyte>:1518 [#uses=1]
	add ubyte %1518, 1		; <ubyte>:1519 [#uses=1]
	store ubyte %1519, ubyte* %965
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %470		; <ubyte*>:966 [#uses=1]
	load ubyte* %966		; <ubyte>:1520 [#uses=1]
	seteq ubyte %1520, 0		; <bool>:676 [#uses=1]
	br bool %676, label %677, label %676

; <label>:677		; preds = %675, %676
	add uint %462, 7		; <uint>:472 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %472		; <ubyte*>:967 [#uses=1]
	load ubyte* %967		; <ubyte>:1521 [#uses=1]
	seteq ubyte %1521, 0		; <bool>:677 [#uses=1]
	br bool %677, label %679, label %678

; <label>:678		; preds = %677, %678
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %470		; <ubyte*>:968 [#uses=2]
	load ubyte* %968		; <ubyte>:1522 [#uses=1]
	add ubyte %1522, 1		; <ubyte>:1523 [#uses=1]
	store ubyte %1523, ubyte* %968
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %472		; <ubyte*>:969 [#uses=2]
	load ubyte* %969		; <ubyte>:1524 [#uses=2]
	add ubyte %1524, 255		; <ubyte>:1525 [#uses=1]
	store ubyte %1525, ubyte* %969
	seteq ubyte %1524, 1		; <bool>:678 [#uses=1]
	br bool %678, label %679, label %678

; <label>:679		; preds = %677, %678
	add uint %462, 22		; <uint>:473 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %473		; <ubyte*>:970 [#uses=1]
	load ubyte* %970		; <ubyte>:1526 [#uses=1]
	seteq ubyte %1526, 0		; <bool>:679 [#uses=1]
	br bool %679, label %681, label %680

; <label>:680		; preds = %679, %680
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %473		; <ubyte*>:971 [#uses=2]
	load ubyte* %971		; <ubyte>:1527 [#uses=2]
	add ubyte %1527, 255		; <ubyte>:1528 [#uses=1]
	store ubyte %1528, ubyte* %971
	seteq ubyte %1527, 1		; <bool>:680 [#uses=1]
	br bool %680, label %681, label %680

; <label>:681		; preds = %679, %680
	add uint %462, 4294967201		; <uint>:474 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %474		; <ubyte*>:972 [#uses=1]
	load ubyte* %972		; <ubyte>:1529 [#uses=1]
	seteq ubyte %1529, 0		; <bool>:681 [#uses=1]
	br bool %681, label %683, label %682

; <label>:682		; preds = %681, %682
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %474		; <ubyte*>:973 [#uses=2]
	load ubyte* %973		; <ubyte>:1530 [#uses=1]
	add ubyte %1530, 255		; <ubyte>:1531 [#uses=1]
	store ubyte %1531, ubyte* %973
	add uint %462, 4294967202		; <uint>:475 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %475		; <ubyte*>:974 [#uses=2]
	load ubyte* %974		; <ubyte>:1532 [#uses=1]
	add ubyte %1532, 1		; <ubyte>:1533 [#uses=1]
	store ubyte %1533, ubyte* %974
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %473		; <ubyte*>:975 [#uses=2]
	load ubyte* %975		; <ubyte>:1534 [#uses=1]
	add ubyte %1534, 1		; <ubyte>:1535 [#uses=1]
	store ubyte %1535, ubyte* %975
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %474		; <ubyte*>:976 [#uses=1]
	load ubyte* %976		; <ubyte>:1536 [#uses=1]
	seteq ubyte %1536, 0		; <bool>:682 [#uses=1]
	br bool %682, label %683, label %682

; <label>:683		; preds = %681, %682
	add uint %462, 4294967202		; <uint>:476 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %476		; <ubyte*>:977 [#uses=1]
	load ubyte* %977		; <ubyte>:1537 [#uses=1]
	seteq ubyte %1537, 0		; <bool>:683 [#uses=1]
	br bool %683, label %685, label %684

; <label>:684		; preds = %683, %684
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %474		; <ubyte*>:978 [#uses=2]
	load ubyte* %978		; <ubyte>:1538 [#uses=1]
	add ubyte %1538, 1		; <ubyte>:1539 [#uses=1]
	store ubyte %1539, ubyte* %978
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %476		; <ubyte*>:979 [#uses=2]
	load ubyte* %979		; <ubyte>:1540 [#uses=2]
	add ubyte %1540, 255		; <ubyte>:1541 [#uses=1]
	store ubyte %1541, ubyte* %979
	seteq ubyte %1540, 1		; <bool>:684 [#uses=1]
	br bool %684, label %685, label %684

; <label>:685		; preds = %683, %684
	add uint %462, 28		; <uint>:477 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %477		; <ubyte*>:980 [#uses=1]
	load ubyte* %980		; <ubyte>:1542 [#uses=1]
	seteq ubyte %1542, 0		; <bool>:685 [#uses=1]
	br bool %685, label %687, label %686

; <label>:686		; preds = %685, %686
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %477		; <ubyte*>:981 [#uses=2]
	load ubyte* %981		; <ubyte>:1543 [#uses=2]
	add ubyte %1543, 255		; <ubyte>:1544 [#uses=1]
	store ubyte %1544, ubyte* %981
	seteq ubyte %1543, 1		; <bool>:686 [#uses=1]
	br bool %686, label %687, label %686

; <label>:687		; preds = %685, %686
	add uint %462, 4294967207		; <uint>:478 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %478		; <ubyte*>:982 [#uses=1]
	load ubyte* %982		; <ubyte>:1545 [#uses=1]
	seteq ubyte %1545, 0		; <bool>:687 [#uses=1]
	br bool %687, label %689, label %688

; <label>:688		; preds = %687, %688
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %478		; <ubyte*>:983 [#uses=2]
	load ubyte* %983		; <ubyte>:1546 [#uses=1]
	add ubyte %1546, 255		; <ubyte>:1547 [#uses=1]
	store ubyte %1547, ubyte* %983
	add uint %462, 4294967208		; <uint>:479 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %479		; <ubyte*>:984 [#uses=2]
	load ubyte* %984		; <ubyte>:1548 [#uses=1]
	add ubyte %1548, 1		; <ubyte>:1549 [#uses=1]
	store ubyte %1549, ubyte* %984
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %477		; <ubyte*>:985 [#uses=2]
	load ubyte* %985		; <ubyte>:1550 [#uses=1]
	add ubyte %1550, 1		; <ubyte>:1551 [#uses=1]
	store ubyte %1551, ubyte* %985
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %478		; <ubyte*>:986 [#uses=1]
	load ubyte* %986		; <ubyte>:1552 [#uses=1]
	seteq ubyte %1552, 0		; <bool>:688 [#uses=1]
	br bool %688, label %689, label %688

; <label>:689		; preds = %687, %688
	add uint %462, 4294967208		; <uint>:480 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %480		; <ubyte*>:987 [#uses=1]
	load ubyte* %987		; <ubyte>:1553 [#uses=1]
	seteq ubyte %1553, 0		; <bool>:689 [#uses=1]
	br bool %689, label %691, label %690

; <label>:690		; preds = %689, %690
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %478		; <ubyte*>:988 [#uses=2]
	load ubyte* %988		; <ubyte>:1554 [#uses=1]
	add ubyte %1554, 1		; <ubyte>:1555 [#uses=1]
	store ubyte %1555, ubyte* %988
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %480		; <ubyte*>:989 [#uses=2]
	load ubyte* %989		; <ubyte>:1556 [#uses=2]
	add ubyte %1556, 255		; <ubyte>:1557 [#uses=1]
	store ubyte %1557, ubyte* %989
	seteq ubyte %1556, 1		; <bool>:690 [#uses=1]
	br bool %690, label %691, label %690

; <label>:691		; preds = %689, %690
	add uint %462, 34		; <uint>:481 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %481		; <ubyte*>:990 [#uses=1]
	load ubyte* %990		; <ubyte>:1558 [#uses=1]
	seteq ubyte %1558, 0		; <bool>:691 [#uses=1]
	br bool %691, label %693, label %692

; <label>:692		; preds = %691, %692
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %481		; <ubyte*>:991 [#uses=2]
	load ubyte* %991		; <ubyte>:1559 [#uses=2]
	add ubyte %1559, 255		; <ubyte>:1560 [#uses=1]
	store ubyte %1560, ubyte* %991
	seteq ubyte %1559, 1		; <bool>:692 [#uses=1]
	br bool %692, label %693, label %692

; <label>:693		; preds = %691, %692
	add uint %462, 4294967213		; <uint>:482 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %482		; <ubyte*>:992 [#uses=1]
	load ubyte* %992		; <ubyte>:1561 [#uses=1]
	seteq ubyte %1561, 0		; <bool>:693 [#uses=1]
	br bool %693, label %695, label %694

; <label>:694		; preds = %693, %694
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %482		; <ubyte*>:993 [#uses=2]
	load ubyte* %993		; <ubyte>:1562 [#uses=1]
	add ubyte %1562, 255		; <ubyte>:1563 [#uses=1]
	store ubyte %1563, ubyte* %993
	add uint %462, 4294967214		; <uint>:483 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %483		; <ubyte*>:994 [#uses=2]
	load ubyte* %994		; <ubyte>:1564 [#uses=1]
	add ubyte %1564, 1		; <ubyte>:1565 [#uses=1]
	store ubyte %1565, ubyte* %994
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %481		; <ubyte*>:995 [#uses=2]
	load ubyte* %995		; <ubyte>:1566 [#uses=1]
	add ubyte %1566, 1		; <ubyte>:1567 [#uses=1]
	store ubyte %1567, ubyte* %995
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %482		; <ubyte*>:996 [#uses=1]
	load ubyte* %996		; <ubyte>:1568 [#uses=1]
	seteq ubyte %1568, 0		; <bool>:694 [#uses=1]
	br bool %694, label %695, label %694

; <label>:695		; preds = %693, %694
	add uint %462, 4294967214		; <uint>:484 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %484		; <ubyte*>:997 [#uses=1]
	load ubyte* %997		; <ubyte>:1569 [#uses=1]
	seteq ubyte %1569, 0		; <bool>:695 [#uses=1]
	br bool %695, label %697, label %696

; <label>:696		; preds = %695, %696
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %482		; <ubyte*>:998 [#uses=2]
	load ubyte* %998		; <ubyte>:1570 [#uses=1]
	add ubyte %1570, 1		; <ubyte>:1571 [#uses=1]
	store ubyte %1571, ubyte* %998
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %484		; <ubyte*>:999 [#uses=2]
	load ubyte* %999		; <ubyte>:1572 [#uses=2]
	add ubyte %1572, 255		; <ubyte>:1573 [#uses=1]
	store ubyte %1573, ubyte* %999
	seteq ubyte %1572, 1		; <bool>:696 [#uses=1]
	br bool %696, label %697, label %696

; <label>:697		; preds = %695, %696
	add uint %462, 40		; <uint>:485 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %485		; <ubyte*>:1000 [#uses=1]
	load ubyte* %1000		; <ubyte>:1574 [#uses=1]
	seteq ubyte %1574, 0		; <bool>:697 [#uses=1]
	br bool %697, label %699, label %698

; <label>:698		; preds = %697, %698
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %485		; <ubyte*>:1001 [#uses=2]
	load ubyte* %1001		; <ubyte>:1575 [#uses=2]
	add ubyte %1575, 255		; <ubyte>:1576 [#uses=1]
	store ubyte %1576, ubyte* %1001
	seteq ubyte %1575, 1		; <bool>:698 [#uses=1]
	br bool %698, label %699, label %698

; <label>:699		; preds = %697, %698
	add uint %462, 4294967219		; <uint>:486 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %486		; <ubyte*>:1002 [#uses=1]
	load ubyte* %1002		; <ubyte>:1577 [#uses=1]
	seteq ubyte %1577, 0		; <bool>:699 [#uses=1]
	br bool %699, label %701, label %700

; <label>:700		; preds = %699, %700
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %486		; <ubyte*>:1003 [#uses=2]
	load ubyte* %1003		; <ubyte>:1578 [#uses=1]
	add ubyte %1578, 255		; <ubyte>:1579 [#uses=1]
	store ubyte %1579, ubyte* %1003
	add uint %462, 4294967220		; <uint>:487 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %487		; <ubyte*>:1004 [#uses=2]
	load ubyte* %1004		; <ubyte>:1580 [#uses=1]
	add ubyte %1580, 1		; <ubyte>:1581 [#uses=1]
	store ubyte %1581, ubyte* %1004
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %485		; <ubyte*>:1005 [#uses=2]
	load ubyte* %1005		; <ubyte>:1582 [#uses=1]
	add ubyte %1582, 1		; <ubyte>:1583 [#uses=1]
	store ubyte %1583, ubyte* %1005
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %486		; <ubyte*>:1006 [#uses=1]
	load ubyte* %1006		; <ubyte>:1584 [#uses=1]
	seteq ubyte %1584, 0		; <bool>:700 [#uses=1]
	br bool %700, label %701, label %700

; <label>:701		; preds = %699, %700
	add uint %462, 4294967220		; <uint>:488 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %488		; <ubyte*>:1007 [#uses=1]
	load ubyte* %1007		; <ubyte>:1585 [#uses=1]
	seteq ubyte %1585, 0		; <bool>:701 [#uses=1]
	br bool %701, label %703, label %702

; <label>:702		; preds = %701, %702
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %486		; <ubyte*>:1008 [#uses=2]
	load ubyte* %1008		; <ubyte>:1586 [#uses=1]
	add ubyte %1586, 1		; <ubyte>:1587 [#uses=1]
	store ubyte %1587, ubyte* %1008
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %488		; <ubyte*>:1009 [#uses=2]
	load ubyte* %1009		; <ubyte>:1588 [#uses=2]
	add ubyte %1588, 255		; <ubyte>:1589 [#uses=1]
	store ubyte %1589, ubyte* %1009
	seteq ubyte %1588, 1		; <bool>:702 [#uses=1]
	br bool %702, label %703, label %702

; <label>:703		; preds = %701, %702
	add uint %462, 46		; <uint>:489 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %489		; <ubyte*>:1010 [#uses=1]
	load ubyte* %1010		; <ubyte>:1590 [#uses=1]
	seteq ubyte %1590, 0		; <bool>:703 [#uses=1]
	br bool %703, label %705, label %704

; <label>:704		; preds = %703, %704
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %489		; <ubyte*>:1011 [#uses=2]
	load ubyte* %1011		; <ubyte>:1591 [#uses=2]
	add ubyte %1591, 255		; <ubyte>:1592 [#uses=1]
	store ubyte %1592, ubyte* %1011
	seteq ubyte %1591, 1		; <bool>:704 [#uses=1]
	br bool %704, label %705, label %704

; <label>:705		; preds = %703, %704
	add uint %462, 4294967225		; <uint>:490 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %490		; <ubyte*>:1012 [#uses=1]
	load ubyte* %1012		; <ubyte>:1593 [#uses=1]
	seteq ubyte %1593, 0		; <bool>:705 [#uses=1]
	br bool %705, label %707, label %706

; <label>:706		; preds = %705, %706
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %490		; <ubyte*>:1013 [#uses=2]
	load ubyte* %1013		; <ubyte>:1594 [#uses=1]
	add ubyte %1594, 255		; <ubyte>:1595 [#uses=1]
	store ubyte %1595, ubyte* %1013
	add uint %462, 4294967226		; <uint>:491 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %491		; <ubyte*>:1014 [#uses=2]
	load ubyte* %1014		; <ubyte>:1596 [#uses=1]
	add ubyte %1596, 1		; <ubyte>:1597 [#uses=1]
	store ubyte %1597, ubyte* %1014
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %489		; <ubyte*>:1015 [#uses=2]
	load ubyte* %1015		; <ubyte>:1598 [#uses=1]
	add ubyte %1598, 1		; <ubyte>:1599 [#uses=1]
	store ubyte %1599, ubyte* %1015
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %490		; <ubyte*>:1016 [#uses=1]
	load ubyte* %1016		; <ubyte>:1600 [#uses=1]
	seteq ubyte %1600, 0		; <bool>:706 [#uses=1]
	br bool %706, label %707, label %706

; <label>:707		; preds = %705, %706
	add uint %462, 4294967226		; <uint>:492 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %492		; <ubyte*>:1017 [#uses=1]
	load ubyte* %1017		; <ubyte>:1601 [#uses=1]
	seteq ubyte %1601, 0		; <bool>:707 [#uses=1]
	br bool %707, label %709, label %708

; <label>:708		; preds = %707, %708
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %490		; <ubyte*>:1018 [#uses=2]
	load ubyte* %1018		; <ubyte>:1602 [#uses=1]
	add ubyte %1602, 1		; <ubyte>:1603 [#uses=1]
	store ubyte %1603, ubyte* %1018
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %492		; <ubyte*>:1019 [#uses=2]
	load ubyte* %1019		; <ubyte>:1604 [#uses=2]
	add ubyte %1604, 255		; <ubyte>:1605 [#uses=1]
	store ubyte %1605, ubyte* %1019
	seteq ubyte %1604, 1		; <bool>:708 [#uses=1]
	br bool %708, label %709, label %708

; <label>:709		; preds = %707, %708
	add uint %462, 52		; <uint>:493 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %493		; <ubyte*>:1020 [#uses=1]
	load ubyte* %1020		; <ubyte>:1606 [#uses=1]
	seteq ubyte %1606, 0		; <bool>:709 [#uses=1]
	br bool %709, label %711, label %710

; <label>:710		; preds = %709, %710
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %493		; <ubyte*>:1021 [#uses=2]
	load ubyte* %1021		; <ubyte>:1607 [#uses=2]
	add ubyte %1607, 255		; <ubyte>:1608 [#uses=1]
	store ubyte %1608, ubyte* %1021
	seteq ubyte %1607, 1		; <bool>:710 [#uses=1]
	br bool %710, label %711, label %710

; <label>:711		; preds = %709, %710
	add uint %462, 4294967231		; <uint>:494 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %494		; <ubyte*>:1022 [#uses=1]
	load ubyte* %1022		; <ubyte>:1609 [#uses=1]
	seteq ubyte %1609, 0		; <bool>:711 [#uses=1]
	br bool %711, label %713, label %712

; <label>:712		; preds = %711, %712
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %494		; <ubyte*>:1023 [#uses=2]
	load ubyte* %1023		; <ubyte>:1610 [#uses=1]
	add ubyte %1610, 255		; <ubyte>:1611 [#uses=1]
	store ubyte %1611, ubyte* %1023
	add uint %462, 4294967232		; <uint>:495 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %495		; <ubyte*>:1024 [#uses=2]
	load ubyte* %1024		; <ubyte>:1612 [#uses=1]
	add ubyte %1612, 1		; <ubyte>:1613 [#uses=1]
	store ubyte %1613, ubyte* %1024
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %493		; <ubyte*>:1025 [#uses=2]
	load ubyte* %1025		; <ubyte>:1614 [#uses=1]
	add ubyte %1614, 1		; <ubyte>:1615 [#uses=1]
	store ubyte %1615, ubyte* %1025
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %494		; <ubyte*>:1026 [#uses=1]
	load ubyte* %1026		; <ubyte>:1616 [#uses=1]
	seteq ubyte %1616, 0		; <bool>:712 [#uses=1]
	br bool %712, label %713, label %712

; <label>:713		; preds = %711, %712
	add uint %462, 4294967232		; <uint>:496 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %496		; <ubyte*>:1027 [#uses=1]
	load ubyte* %1027		; <ubyte>:1617 [#uses=1]
	seteq ubyte %1617, 0		; <bool>:713 [#uses=1]
	br bool %713, label %715, label %714

; <label>:714		; preds = %713, %714
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %494		; <ubyte*>:1028 [#uses=2]
	load ubyte* %1028		; <ubyte>:1618 [#uses=1]
	add ubyte %1618, 1		; <ubyte>:1619 [#uses=1]
	store ubyte %1619, ubyte* %1028
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %496		; <ubyte*>:1029 [#uses=2]
	load ubyte* %1029		; <ubyte>:1620 [#uses=2]
	add ubyte %1620, 255		; <ubyte>:1621 [#uses=1]
	store ubyte %1621, ubyte* %1029
	seteq ubyte %1620, 1		; <bool>:714 [#uses=1]
	br bool %714, label %715, label %714

; <label>:715		; preds = %713, %714
	add uint %462, 58		; <uint>:497 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %497		; <ubyte*>:1030 [#uses=1]
	load ubyte* %1030		; <ubyte>:1622 [#uses=1]
	seteq ubyte %1622, 0		; <bool>:715 [#uses=1]
	br bool %715, label %717, label %716

; <label>:716		; preds = %715, %716
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %497		; <ubyte*>:1031 [#uses=2]
	load ubyte* %1031		; <ubyte>:1623 [#uses=2]
	add ubyte %1623, 255		; <ubyte>:1624 [#uses=1]
	store ubyte %1624, ubyte* %1031
	seteq ubyte %1623, 1		; <bool>:716 [#uses=1]
	br bool %716, label %717, label %716

; <label>:717		; preds = %715, %716
	add uint %462, 4294967237		; <uint>:498 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %498		; <ubyte*>:1032 [#uses=1]
	load ubyte* %1032		; <ubyte>:1625 [#uses=1]
	seteq ubyte %1625, 0		; <bool>:717 [#uses=1]
	br bool %717, label %719, label %718

; <label>:718		; preds = %717, %718
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %498		; <ubyte*>:1033 [#uses=2]
	load ubyte* %1033		; <ubyte>:1626 [#uses=1]
	add ubyte %1626, 255		; <ubyte>:1627 [#uses=1]
	store ubyte %1627, ubyte* %1033
	add uint %462, 4294967238		; <uint>:499 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %499		; <ubyte*>:1034 [#uses=2]
	load ubyte* %1034		; <ubyte>:1628 [#uses=1]
	add ubyte %1628, 1		; <ubyte>:1629 [#uses=1]
	store ubyte %1629, ubyte* %1034
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %497		; <ubyte*>:1035 [#uses=2]
	load ubyte* %1035		; <ubyte>:1630 [#uses=1]
	add ubyte %1630, 1		; <ubyte>:1631 [#uses=1]
	store ubyte %1631, ubyte* %1035
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %498		; <ubyte*>:1036 [#uses=1]
	load ubyte* %1036		; <ubyte>:1632 [#uses=1]
	seteq ubyte %1632, 0		; <bool>:718 [#uses=1]
	br bool %718, label %719, label %718

; <label>:719		; preds = %717, %718
	add uint %462, 4294967238		; <uint>:500 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %500		; <ubyte*>:1037 [#uses=1]
	load ubyte* %1037		; <ubyte>:1633 [#uses=1]
	seteq ubyte %1633, 0		; <bool>:719 [#uses=1]
	br bool %719, label %721, label %720

; <label>:720		; preds = %719, %720
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %498		; <ubyte*>:1038 [#uses=2]
	load ubyte* %1038		; <ubyte>:1634 [#uses=1]
	add ubyte %1634, 1		; <ubyte>:1635 [#uses=1]
	store ubyte %1635, ubyte* %1038
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %500		; <ubyte*>:1039 [#uses=2]
	load ubyte* %1039		; <ubyte>:1636 [#uses=2]
	add ubyte %1636, 255		; <ubyte>:1637 [#uses=1]
	store ubyte %1637, ubyte* %1039
	seteq ubyte %1636, 1		; <bool>:720 [#uses=1]
	br bool %720, label %721, label %720

; <label>:721		; preds = %719, %720
	add uint %462, 64		; <uint>:501 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %501		; <ubyte*>:1040 [#uses=1]
	load ubyte* %1040		; <ubyte>:1638 [#uses=1]
	seteq ubyte %1638, 0		; <bool>:721 [#uses=1]
	br bool %721, label %723, label %722

; <label>:722		; preds = %721, %722
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %501		; <ubyte*>:1041 [#uses=2]
	load ubyte* %1041		; <ubyte>:1639 [#uses=2]
	add ubyte %1639, 255		; <ubyte>:1640 [#uses=1]
	store ubyte %1640, ubyte* %1041
	seteq ubyte %1639, 1		; <bool>:722 [#uses=1]
	br bool %722, label %723, label %722

; <label>:723		; preds = %721, %722
	add uint %462, 4294967243		; <uint>:502 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %502		; <ubyte*>:1042 [#uses=1]
	load ubyte* %1042		; <ubyte>:1641 [#uses=1]
	seteq ubyte %1641, 0		; <bool>:723 [#uses=1]
	br bool %723, label %725, label %724

; <label>:724		; preds = %723, %724
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %502		; <ubyte*>:1043 [#uses=2]
	load ubyte* %1043		; <ubyte>:1642 [#uses=1]
	add ubyte %1642, 255		; <ubyte>:1643 [#uses=1]
	store ubyte %1643, ubyte* %1043
	add uint %462, 4294967244		; <uint>:503 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %503		; <ubyte*>:1044 [#uses=2]
	load ubyte* %1044		; <ubyte>:1644 [#uses=1]
	add ubyte %1644, 1		; <ubyte>:1645 [#uses=1]
	store ubyte %1645, ubyte* %1044
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %501		; <ubyte*>:1045 [#uses=2]
	load ubyte* %1045		; <ubyte>:1646 [#uses=1]
	add ubyte %1646, 1		; <ubyte>:1647 [#uses=1]
	store ubyte %1647, ubyte* %1045
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %502		; <ubyte*>:1046 [#uses=1]
	load ubyte* %1046		; <ubyte>:1648 [#uses=1]
	seteq ubyte %1648, 0		; <bool>:724 [#uses=1]
	br bool %724, label %725, label %724

; <label>:725		; preds = %723, %724
	add uint %462, 4294967244		; <uint>:504 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %504		; <ubyte*>:1047 [#uses=1]
	load ubyte* %1047		; <ubyte>:1649 [#uses=1]
	seteq ubyte %1649, 0		; <bool>:725 [#uses=1]
	br bool %725, label %727, label %726

; <label>:726		; preds = %725, %726
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %502		; <ubyte*>:1048 [#uses=2]
	load ubyte* %1048		; <ubyte>:1650 [#uses=1]
	add ubyte %1650, 1		; <ubyte>:1651 [#uses=1]
	store ubyte %1651, ubyte* %1048
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %504		; <ubyte*>:1049 [#uses=2]
	load ubyte* %1049		; <ubyte>:1652 [#uses=2]
	add ubyte %1652, 255		; <ubyte>:1653 [#uses=1]
	store ubyte %1653, ubyte* %1049
	seteq ubyte %1652, 1		; <bool>:726 [#uses=1]
	br bool %726, label %727, label %726

; <label>:727		; preds = %725, %726
	add uint %462, 70		; <uint>:505 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %505		; <ubyte*>:1050 [#uses=1]
	load ubyte* %1050		; <ubyte>:1654 [#uses=1]
	seteq ubyte %1654, 0		; <bool>:727 [#uses=1]
	br bool %727, label %729, label %728

; <label>:728		; preds = %727, %728
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %505		; <ubyte*>:1051 [#uses=2]
	load ubyte* %1051		; <ubyte>:1655 [#uses=2]
	add ubyte %1655, 255		; <ubyte>:1656 [#uses=1]
	store ubyte %1656, ubyte* %1051
	seteq ubyte %1655, 1		; <bool>:728 [#uses=1]
	br bool %728, label %729, label %728

; <label>:729		; preds = %727, %728
	add uint %462, 4294967249		; <uint>:506 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %506		; <ubyte*>:1052 [#uses=1]
	load ubyte* %1052		; <ubyte>:1657 [#uses=1]
	seteq ubyte %1657, 0		; <bool>:729 [#uses=1]
	br bool %729, label %731, label %730

; <label>:730		; preds = %729, %730
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %506		; <ubyte*>:1053 [#uses=2]
	load ubyte* %1053		; <ubyte>:1658 [#uses=1]
	add ubyte %1658, 255		; <ubyte>:1659 [#uses=1]
	store ubyte %1659, ubyte* %1053
	add uint %462, 4294967250		; <uint>:507 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %507		; <ubyte*>:1054 [#uses=2]
	load ubyte* %1054		; <ubyte>:1660 [#uses=1]
	add ubyte %1660, 1		; <ubyte>:1661 [#uses=1]
	store ubyte %1661, ubyte* %1054
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %505		; <ubyte*>:1055 [#uses=2]
	load ubyte* %1055		; <ubyte>:1662 [#uses=1]
	add ubyte %1662, 1		; <ubyte>:1663 [#uses=1]
	store ubyte %1663, ubyte* %1055
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %506		; <ubyte*>:1056 [#uses=1]
	load ubyte* %1056		; <ubyte>:1664 [#uses=1]
	seteq ubyte %1664, 0		; <bool>:730 [#uses=1]
	br bool %730, label %731, label %730

; <label>:731		; preds = %729, %730
	add uint %462, 4294967250		; <uint>:508 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %508		; <ubyte*>:1057 [#uses=1]
	load ubyte* %1057		; <ubyte>:1665 [#uses=1]
	seteq ubyte %1665, 0		; <bool>:731 [#uses=1]
	br bool %731, label %733, label %732

; <label>:732		; preds = %731, %732
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %506		; <ubyte*>:1058 [#uses=2]
	load ubyte* %1058		; <ubyte>:1666 [#uses=1]
	add ubyte %1666, 1		; <ubyte>:1667 [#uses=1]
	store ubyte %1667, ubyte* %1058
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %508		; <ubyte*>:1059 [#uses=2]
	load ubyte* %1059		; <ubyte>:1668 [#uses=2]
	add ubyte %1668, 255		; <ubyte>:1669 [#uses=1]
	store ubyte %1669, ubyte* %1059
	seteq ubyte %1668, 1		; <bool>:732 [#uses=1]
	br bool %732, label %733, label %732

; <label>:733		; preds = %731, %732
	add uint %462, 76		; <uint>:509 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %509		; <ubyte*>:1060 [#uses=1]
	load ubyte* %1060		; <ubyte>:1670 [#uses=1]
	seteq ubyte %1670, 0		; <bool>:733 [#uses=1]
	br bool %733, label %735, label %734

; <label>:734		; preds = %733, %734
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %509		; <ubyte*>:1061 [#uses=2]
	load ubyte* %1061		; <ubyte>:1671 [#uses=2]
	add ubyte %1671, 255		; <ubyte>:1672 [#uses=1]
	store ubyte %1672, ubyte* %1061
	seteq ubyte %1671, 1		; <bool>:734 [#uses=1]
	br bool %734, label %735, label %734

; <label>:735		; preds = %733, %734
	add uint %462, 4294967255		; <uint>:510 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %510		; <ubyte*>:1062 [#uses=1]
	load ubyte* %1062		; <ubyte>:1673 [#uses=1]
	seteq ubyte %1673, 0		; <bool>:735 [#uses=1]
	br bool %735, label %737, label %736

; <label>:736		; preds = %735, %736
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %510		; <ubyte*>:1063 [#uses=2]
	load ubyte* %1063		; <ubyte>:1674 [#uses=1]
	add ubyte %1674, 255		; <ubyte>:1675 [#uses=1]
	store ubyte %1675, ubyte* %1063
	add uint %462, 4294967256		; <uint>:511 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %511		; <ubyte*>:1064 [#uses=2]
	load ubyte* %1064		; <ubyte>:1676 [#uses=1]
	add ubyte %1676, 1		; <ubyte>:1677 [#uses=1]
	store ubyte %1677, ubyte* %1064
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %509		; <ubyte*>:1065 [#uses=2]
	load ubyte* %1065		; <ubyte>:1678 [#uses=1]
	add ubyte %1678, 1		; <ubyte>:1679 [#uses=1]
	store ubyte %1679, ubyte* %1065
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %510		; <ubyte*>:1066 [#uses=1]
	load ubyte* %1066		; <ubyte>:1680 [#uses=1]
	seteq ubyte %1680, 0		; <bool>:736 [#uses=1]
	br bool %736, label %737, label %736

; <label>:737		; preds = %735, %736
	add uint %462, 4294967256		; <uint>:512 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %512		; <ubyte*>:1067 [#uses=1]
	load ubyte* %1067		; <ubyte>:1681 [#uses=1]
	seteq ubyte %1681, 0		; <bool>:737 [#uses=1]
	br bool %737, label %739, label %738

; <label>:738		; preds = %737, %738
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %510		; <ubyte*>:1068 [#uses=2]
	load ubyte* %1068		; <ubyte>:1682 [#uses=1]
	add ubyte %1682, 1		; <ubyte>:1683 [#uses=1]
	store ubyte %1683, ubyte* %1068
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %512		; <ubyte*>:1069 [#uses=2]
	load ubyte* %1069		; <ubyte>:1684 [#uses=2]
	add ubyte %1684, 255		; <ubyte>:1685 [#uses=1]
	store ubyte %1685, ubyte* %1069
	seteq ubyte %1684, 1		; <bool>:738 [#uses=1]
	br bool %738, label %739, label %738

; <label>:739		; preds = %737, %738
	add uint %462, 82		; <uint>:513 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %513		; <ubyte*>:1070 [#uses=1]
	load ubyte* %1070		; <ubyte>:1686 [#uses=1]
	seteq ubyte %1686, 0		; <bool>:739 [#uses=1]
	br bool %739, label %741, label %740

; <label>:740		; preds = %739, %740
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %513		; <ubyte*>:1071 [#uses=2]
	load ubyte* %1071		; <ubyte>:1687 [#uses=2]
	add ubyte %1687, 255		; <ubyte>:1688 [#uses=1]
	store ubyte %1688, ubyte* %1071
	seteq ubyte %1687, 1		; <bool>:740 [#uses=1]
	br bool %740, label %741, label %740

; <label>:741		; preds = %739, %740
	add uint %462, 4294967261		; <uint>:514 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %514		; <ubyte*>:1072 [#uses=1]
	load ubyte* %1072		; <ubyte>:1689 [#uses=1]
	seteq ubyte %1689, 0		; <bool>:741 [#uses=1]
	br bool %741, label %743, label %742

; <label>:742		; preds = %741, %742
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %514		; <ubyte*>:1073 [#uses=2]
	load ubyte* %1073		; <ubyte>:1690 [#uses=1]
	add ubyte %1690, 255		; <ubyte>:1691 [#uses=1]
	store ubyte %1691, ubyte* %1073
	add uint %462, 4294967262		; <uint>:515 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %515		; <ubyte*>:1074 [#uses=2]
	load ubyte* %1074		; <ubyte>:1692 [#uses=1]
	add ubyte %1692, 1		; <ubyte>:1693 [#uses=1]
	store ubyte %1693, ubyte* %1074
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %513		; <ubyte*>:1075 [#uses=2]
	load ubyte* %1075		; <ubyte>:1694 [#uses=1]
	add ubyte %1694, 1		; <ubyte>:1695 [#uses=1]
	store ubyte %1695, ubyte* %1075
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %514		; <ubyte*>:1076 [#uses=1]
	load ubyte* %1076		; <ubyte>:1696 [#uses=1]
	seteq ubyte %1696, 0		; <bool>:742 [#uses=1]
	br bool %742, label %743, label %742

; <label>:743		; preds = %741, %742
	add uint %462, 4294967262		; <uint>:516 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %516		; <ubyte*>:1077 [#uses=1]
	load ubyte* %1077		; <ubyte>:1697 [#uses=1]
	seteq ubyte %1697, 0		; <bool>:743 [#uses=1]
	br bool %743, label %745, label %744

; <label>:744		; preds = %743, %744
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %514		; <ubyte*>:1078 [#uses=2]
	load ubyte* %1078		; <ubyte>:1698 [#uses=1]
	add ubyte %1698, 1		; <ubyte>:1699 [#uses=1]
	store ubyte %1699, ubyte* %1078
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %516		; <ubyte*>:1079 [#uses=2]
	load ubyte* %1079		; <ubyte>:1700 [#uses=2]
	add ubyte %1700, 255		; <ubyte>:1701 [#uses=1]
	store ubyte %1701, ubyte* %1079
	seteq ubyte %1700, 1		; <bool>:744 [#uses=1]
	br bool %744, label %745, label %744

; <label>:745		; preds = %743, %744
	add uint %462, 88		; <uint>:517 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %517		; <ubyte*>:1080 [#uses=1]
	load ubyte* %1080		; <ubyte>:1702 [#uses=1]
	seteq ubyte %1702, 0		; <bool>:745 [#uses=1]
	br bool %745, label %747, label %746

; <label>:746		; preds = %745, %746
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %517		; <ubyte*>:1081 [#uses=2]
	load ubyte* %1081		; <ubyte>:1703 [#uses=2]
	add ubyte %1703, 255		; <ubyte>:1704 [#uses=1]
	store ubyte %1704, ubyte* %1081
	seteq ubyte %1703, 1		; <bool>:746 [#uses=1]
	br bool %746, label %747, label %746

; <label>:747		; preds = %745, %746
	add uint %462, 4294967267		; <uint>:518 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %518		; <ubyte*>:1082 [#uses=1]
	load ubyte* %1082		; <ubyte>:1705 [#uses=1]
	seteq ubyte %1705, 0		; <bool>:747 [#uses=1]
	br bool %747, label %749, label %748

; <label>:748		; preds = %747, %748
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %518		; <ubyte*>:1083 [#uses=2]
	load ubyte* %1083		; <ubyte>:1706 [#uses=1]
	add ubyte %1706, 255		; <ubyte>:1707 [#uses=1]
	store ubyte %1707, ubyte* %1083
	add uint %462, 4294967268		; <uint>:519 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %519		; <ubyte*>:1084 [#uses=2]
	load ubyte* %1084		; <ubyte>:1708 [#uses=1]
	add ubyte %1708, 1		; <ubyte>:1709 [#uses=1]
	store ubyte %1709, ubyte* %1084
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %517		; <ubyte*>:1085 [#uses=2]
	load ubyte* %1085		; <ubyte>:1710 [#uses=1]
	add ubyte %1710, 1		; <ubyte>:1711 [#uses=1]
	store ubyte %1711, ubyte* %1085
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %518		; <ubyte*>:1086 [#uses=1]
	load ubyte* %1086		; <ubyte>:1712 [#uses=1]
	seteq ubyte %1712, 0		; <bool>:748 [#uses=1]
	br bool %748, label %749, label %748

; <label>:749		; preds = %747, %748
	add uint %462, 4294967268		; <uint>:520 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %520		; <ubyte*>:1087 [#uses=1]
	load ubyte* %1087		; <ubyte>:1713 [#uses=1]
	seteq ubyte %1713, 0		; <bool>:749 [#uses=1]
	br bool %749, label %751, label %750

; <label>:750		; preds = %749, %750
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %518		; <ubyte*>:1088 [#uses=2]
	load ubyte* %1088		; <ubyte>:1714 [#uses=1]
	add ubyte %1714, 1		; <ubyte>:1715 [#uses=1]
	store ubyte %1715, ubyte* %1088
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %520		; <ubyte*>:1089 [#uses=2]
	load ubyte* %1089		; <ubyte>:1716 [#uses=2]
	add ubyte %1716, 255		; <ubyte>:1717 [#uses=1]
	store ubyte %1717, ubyte* %1089
	seteq ubyte %1716, 1		; <bool>:750 [#uses=1]
	br bool %750, label %751, label %750

; <label>:751		; preds = %749, %750
	add uint %462, 94		; <uint>:521 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %521		; <ubyte*>:1090 [#uses=1]
	load ubyte* %1090		; <ubyte>:1718 [#uses=1]
	seteq ubyte %1718, 0		; <bool>:751 [#uses=1]
	br bool %751, label %753, label %752

; <label>:752		; preds = %751, %752
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %521		; <ubyte*>:1091 [#uses=2]
	load ubyte* %1091		; <ubyte>:1719 [#uses=2]
	add ubyte %1719, 255		; <ubyte>:1720 [#uses=1]
	store ubyte %1720, ubyte* %1091
	seteq ubyte %1719, 1		; <bool>:752 [#uses=1]
	br bool %752, label %753, label %752

; <label>:753		; preds = %751, %752
	add uint %462, 4294967273		; <uint>:522 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %522		; <ubyte*>:1092 [#uses=1]
	load ubyte* %1092		; <ubyte>:1721 [#uses=1]
	seteq ubyte %1721, 0		; <bool>:753 [#uses=1]
	br bool %753, label %755, label %754

; <label>:754		; preds = %753, %754
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %522		; <ubyte*>:1093 [#uses=2]
	load ubyte* %1093		; <ubyte>:1722 [#uses=1]
	add ubyte %1722, 255		; <ubyte>:1723 [#uses=1]
	store ubyte %1723, ubyte* %1093
	add uint %462, 4294967274		; <uint>:523 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %523		; <ubyte*>:1094 [#uses=2]
	load ubyte* %1094		; <ubyte>:1724 [#uses=1]
	add ubyte %1724, 1		; <ubyte>:1725 [#uses=1]
	store ubyte %1725, ubyte* %1094
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %521		; <ubyte*>:1095 [#uses=2]
	load ubyte* %1095		; <ubyte>:1726 [#uses=1]
	add ubyte %1726, 1		; <ubyte>:1727 [#uses=1]
	store ubyte %1727, ubyte* %1095
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %522		; <ubyte*>:1096 [#uses=1]
	load ubyte* %1096		; <ubyte>:1728 [#uses=1]
	seteq ubyte %1728, 0		; <bool>:754 [#uses=1]
	br bool %754, label %755, label %754

; <label>:755		; preds = %753, %754
	add uint %462, 4294967274		; <uint>:524 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %524		; <ubyte*>:1097 [#uses=1]
	load ubyte* %1097		; <ubyte>:1729 [#uses=1]
	seteq ubyte %1729, 0		; <bool>:755 [#uses=1]
	br bool %755, label %757, label %756

; <label>:756		; preds = %755, %756
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %522		; <ubyte*>:1098 [#uses=2]
	load ubyte* %1098		; <ubyte>:1730 [#uses=1]
	add ubyte %1730, 1		; <ubyte>:1731 [#uses=1]
	store ubyte %1731, ubyte* %1098
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %524		; <ubyte*>:1099 [#uses=2]
	load ubyte* %1099		; <ubyte>:1732 [#uses=2]
	add ubyte %1732, 255		; <ubyte>:1733 [#uses=1]
	store ubyte %1733, ubyte* %1099
	seteq ubyte %1732, 1		; <bool>:756 [#uses=1]
	br bool %756, label %757, label %756

; <label>:757		; preds = %755, %756
	add uint %462, 100		; <uint>:525 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %525		; <ubyte*>:1100 [#uses=1]
	load ubyte* %1100		; <ubyte>:1734 [#uses=1]
	seteq ubyte %1734, 0		; <bool>:757 [#uses=1]
	br bool %757, label %759, label %758

; <label>:758		; preds = %757, %758
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %525		; <ubyte*>:1101 [#uses=2]
	load ubyte* %1101		; <ubyte>:1735 [#uses=2]
	add ubyte %1735, 255		; <ubyte>:1736 [#uses=1]
	store ubyte %1736, ubyte* %1101
	seteq ubyte %1735, 1		; <bool>:758 [#uses=1]
	br bool %758, label %759, label %758

; <label>:759		; preds = %757, %758
	add uint %462, 4294967279		; <uint>:526 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %526		; <ubyte*>:1102 [#uses=1]
	load ubyte* %1102		; <ubyte>:1737 [#uses=1]
	seteq ubyte %1737, 0		; <bool>:759 [#uses=1]
	br bool %759, label %761, label %760

; <label>:760		; preds = %759, %760
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %526		; <ubyte*>:1103 [#uses=2]
	load ubyte* %1103		; <ubyte>:1738 [#uses=1]
	add ubyte %1738, 255		; <ubyte>:1739 [#uses=1]
	store ubyte %1739, ubyte* %1103
	add uint %462, 4294967280		; <uint>:527 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %527		; <ubyte*>:1104 [#uses=2]
	load ubyte* %1104		; <ubyte>:1740 [#uses=1]
	add ubyte %1740, 1		; <ubyte>:1741 [#uses=1]
	store ubyte %1741, ubyte* %1104
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %525		; <ubyte*>:1105 [#uses=2]
	load ubyte* %1105		; <ubyte>:1742 [#uses=1]
	add ubyte %1742, 1		; <ubyte>:1743 [#uses=1]
	store ubyte %1743, ubyte* %1105
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %526		; <ubyte*>:1106 [#uses=1]
	load ubyte* %1106		; <ubyte>:1744 [#uses=1]
	seteq ubyte %1744, 0		; <bool>:760 [#uses=1]
	br bool %760, label %761, label %760

; <label>:761		; preds = %759, %760
	add uint %462, 4294967280		; <uint>:528 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %528		; <ubyte*>:1107 [#uses=1]
	load ubyte* %1107		; <ubyte>:1745 [#uses=1]
	seteq ubyte %1745, 0		; <bool>:761 [#uses=1]
	br bool %761, label %763, label %762

; <label>:762		; preds = %761, %762
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %526		; <ubyte*>:1108 [#uses=2]
	load ubyte* %1108		; <ubyte>:1746 [#uses=1]
	add ubyte %1746, 1		; <ubyte>:1747 [#uses=1]
	store ubyte %1747, ubyte* %1108
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %528		; <ubyte*>:1109 [#uses=2]
	load ubyte* %1109		; <ubyte>:1748 [#uses=2]
	add ubyte %1748, 255		; <ubyte>:1749 [#uses=1]
	store ubyte %1749, ubyte* %1109
	seteq ubyte %1748, 1		; <bool>:762 [#uses=1]
	br bool %762, label %763, label %762

; <label>:763		; preds = %761, %762
	add uint %462, 106		; <uint>:529 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %529		; <ubyte*>:1110 [#uses=1]
	load ubyte* %1110		; <ubyte>:1750 [#uses=1]
	seteq ubyte %1750, 0		; <bool>:763 [#uses=1]
	br bool %763, label %765, label %764

; <label>:764		; preds = %763, %764
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %529		; <ubyte*>:1111 [#uses=2]
	load ubyte* %1111		; <ubyte>:1751 [#uses=2]
	add ubyte %1751, 255		; <ubyte>:1752 [#uses=1]
	store ubyte %1752, ubyte* %1111
	seteq ubyte %1751, 1		; <bool>:764 [#uses=1]
	br bool %764, label %765, label %764

; <label>:765		; preds = %763, %764
	add uint %462, 4294967285		; <uint>:530 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %530		; <ubyte*>:1112 [#uses=1]
	load ubyte* %1112		; <ubyte>:1753 [#uses=1]
	seteq ubyte %1753, 0		; <bool>:765 [#uses=1]
	br bool %765, label %767, label %766

; <label>:766		; preds = %765, %766
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %530		; <ubyte*>:1113 [#uses=2]
	load ubyte* %1113		; <ubyte>:1754 [#uses=1]
	add ubyte %1754, 255		; <ubyte>:1755 [#uses=1]
	store ubyte %1755, ubyte* %1113
	add uint %462, 4294967286		; <uint>:531 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %531		; <ubyte*>:1114 [#uses=2]
	load ubyte* %1114		; <ubyte>:1756 [#uses=1]
	add ubyte %1756, 1		; <ubyte>:1757 [#uses=1]
	store ubyte %1757, ubyte* %1114
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %529		; <ubyte*>:1115 [#uses=2]
	load ubyte* %1115		; <ubyte>:1758 [#uses=1]
	add ubyte %1758, 1		; <ubyte>:1759 [#uses=1]
	store ubyte %1759, ubyte* %1115
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %530		; <ubyte*>:1116 [#uses=1]
	load ubyte* %1116		; <ubyte>:1760 [#uses=1]
	seteq ubyte %1760, 0		; <bool>:766 [#uses=1]
	br bool %766, label %767, label %766

; <label>:767		; preds = %765, %766
	add uint %462, 4294967286		; <uint>:532 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %532		; <ubyte*>:1117 [#uses=1]
	load ubyte* %1117		; <ubyte>:1761 [#uses=1]
	seteq ubyte %1761, 0		; <bool>:767 [#uses=1]
	br bool %767, label %769, label %768

; <label>:768		; preds = %767, %768
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %530		; <ubyte*>:1118 [#uses=2]
	load ubyte* %1118		; <ubyte>:1762 [#uses=1]
	add ubyte %1762, 1		; <ubyte>:1763 [#uses=1]
	store ubyte %1763, ubyte* %1118
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %532		; <ubyte*>:1119 [#uses=2]
	load ubyte* %1119		; <ubyte>:1764 [#uses=2]
	add ubyte %1764, 255		; <ubyte>:1765 [#uses=1]
	store ubyte %1765, ubyte* %1119
	seteq ubyte %1764, 1		; <bool>:768 [#uses=1]
	br bool %768, label %769, label %768

; <label>:769		; preds = %767, %768
	add uint %462, 110		; <uint>:533 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %533		; <ubyte*>:1120 [#uses=2]
	load ubyte* %1120		; <ubyte>:1766 [#uses=1]
	add ubyte %1766, 13		; <ubyte>:1767 [#uses=1]
	store ubyte %1767, ubyte* %1120
	add uint %462, 113		; <uint>:534 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %534		; <ubyte*>:1121 [#uses=2]
	load ubyte* %1121		; <ubyte>:1768 [#uses=1]
	add ubyte %1768, 1		; <ubyte>:1769 [#uses=1]
	store ubyte %1769, ubyte* %1121
	add uint %462, 116		; <uint>:535 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %535		; <ubyte*>:1122 [#uses=2]
	load ubyte* %1122		; <ubyte>:1770 [#uses=1]
	add ubyte %1770, 1		; <ubyte>:1771 [#uses=1]
	store ubyte %1771, ubyte* %1122
	add uint %462, 119		; <uint>:536 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %536		; <ubyte*>:1123 [#uses=2]
	load ubyte* %1123		; <ubyte>:1772 [#uses=1]
	add ubyte %1772, 1		; <ubyte>:1773 [#uses=1]
	store ubyte %1773, ubyte* %1123
	add uint %462, 124		; <uint>:537 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %537		; <ubyte*>:1124 [#uses=1]
	load ubyte* %1124		; <ubyte>:1774 [#uses=1]
	seteq ubyte %1774, 0		; <bool>:769 [#uses=1]
	br bool %769, label %667, label %666

; <label>:770		; preds = %593, %779
	phi uint [ %417, %593 ], [ %545, %779 ]		; <uint>:538 [#uses=6]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %538		; <ubyte*>:1125 [#uses=2]
	load ubyte* %1125		; <ubyte>:1775 [#uses=1]
	add ubyte %1775, 255		; <ubyte>:1776 [#uses=1]
	store ubyte %1776, ubyte* %1125
	add uint %538, 2		; <uint>:539 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %539		; <ubyte*>:1126 [#uses=1]
	load ubyte* %1126		; <ubyte>:1777 [#uses=1]
	seteq ubyte %1777, 0		; <bool>:770 [#uses=1]
	br bool %770, label %773, label %772

; <label>:771		; preds = %593, %779
	phi uint [ %417, %593 ], [ %545, %779 ]		; <uint>:540 [#uses=1]
	add uint %540, 4294967295		; <uint>:541 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %541		; <ubyte*>:1127 [#uses=1]
	load ubyte* %1127		; <ubyte>:1778 [#uses=1]
	seteq ubyte %1778, 0		; <bool>:771 [#uses=1]
	br bool %771, label %591, label %590

; <label>:772		; preds = %770, %772
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %539		; <ubyte*>:1128 [#uses=2]
	load ubyte* %1128		; <ubyte>:1779 [#uses=2]
	add ubyte %1779, 255		; <ubyte>:1780 [#uses=1]
	store ubyte %1780, ubyte* %1128
	seteq ubyte %1779, 1		; <bool>:772 [#uses=1]
	br bool %772, label %773, label %772

; <label>:773		; preds = %770, %772
	add uint %538, 4		; <uint>:542 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %542		; <ubyte*>:1129 [#uses=1]
	load ubyte* %1129		; <ubyte>:1781 [#uses=1]
	seteq ubyte %1781, 0		; <bool>:773 [#uses=1]
	br bool %773, label %775, label %774

; <label>:774		; preds = %773, %774
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %542		; <ubyte*>:1130 [#uses=2]
	load ubyte* %1130		; <ubyte>:1782 [#uses=2]
	add ubyte %1782, 255		; <ubyte>:1783 [#uses=1]
	store ubyte %1783, ubyte* %1130
	seteq ubyte %1782, 1		; <bool>:774 [#uses=1]
	br bool %774, label %775, label %774

; <label>:775		; preds = %773, %774
	add uint %538, 6		; <uint>:543 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %543		; <ubyte*>:1131 [#uses=1]
	load ubyte* %1131		; <ubyte>:1784 [#uses=1]
	seteq ubyte %1784, 0		; <bool>:775 [#uses=1]
	br bool %775, label %777, label %776

; <label>:776		; preds = %775, %776
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %543		; <ubyte*>:1132 [#uses=2]
	load ubyte* %1132		; <ubyte>:1785 [#uses=2]
	add ubyte %1785, 255		; <ubyte>:1786 [#uses=1]
	store ubyte %1786, ubyte* %1132
	seteq ubyte %1785, 1		; <bool>:776 [#uses=1]
	br bool %776, label %777, label %776

; <label>:777		; preds = %775, %776
	add uint %538, 8		; <uint>:544 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %544		; <ubyte*>:1133 [#uses=1]
	load ubyte* %1133		; <ubyte>:1787 [#uses=1]
	seteq ubyte %1787, 0		; <bool>:777 [#uses=1]
	br bool %777, label %779, label %778

; <label>:778		; preds = %777, %778
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %544		; <ubyte*>:1134 [#uses=2]
	load ubyte* %1134		; <ubyte>:1788 [#uses=2]
	add ubyte %1788, 255		; <ubyte>:1789 [#uses=1]
	store ubyte %1789, ubyte* %1134
	seteq ubyte %1788, 1		; <bool>:778 [#uses=1]
	br bool %778, label %779, label %778

; <label>:779		; preds = %777, %778
	add uint %538, 1		; <uint>:545 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %545		; <ubyte*>:1135 [#uses=1]
	load ubyte* %1135		; <ubyte>:1790 [#uses=1]
	seteq ubyte %1790, 0		; <bool>:779 [#uses=1]
	br bool %779, label %771, label %770

; <label>:780		; preds = %591, %845
	phi uint [ %414, %591 ], [ %582, %845 ]		; <uint>:546 [#uses=35]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %546		; <ubyte*>:1136 [#uses=2]
	load ubyte* %1136		; <ubyte>:1791 [#uses=1]
	add ubyte %1791, 255		; <ubyte>:1792 [#uses=1]
	store ubyte %1792, ubyte* %1136
	add uint %546, 4294967090		; <uint>:547 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %547		; <ubyte*>:1137 [#uses=1]
	load ubyte* %1137		; <ubyte>:1793 [#uses=1]
	seteq ubyte %1793, 0		; <bool>:780 [#uses=1]
	br bool %780, label %783, label %782

; <label>:781		; preds = %591, %845
	phi uint [ %414, %591 ], [ %582, %845 ]		; <uint>:548 [#uses=1]
	add uint %548, 4294967295		; <uint>:549 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %549		; <ubyte*>:1138 [#uses=1]
	load ubyte* %1138		; <ubyte>:1794 [#uses=1]
	seteq ubyte %1794, 0		; <bool>:781 [#uses=1]
	br bool %781, label %589, label %588

; <label>:782		; preds = %780, %782
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %547		; <ubyte*>:1139 [#uses=2]
	load ubyte* %1139		; <ubyte>:1795 [#uses=2]
	add ubyte %1795, 255		; <ubyte>:1796 [#uses=1]
	store ubyte %1796, ubyte* %1139
	seteq ubyte %1795, 1		; <bool>:782 [#uses=1]
	br bool %782, label %783, label %782

; <label>:783		; preds = %780, %782
	add uint %546, 4294967207		; <uint>:550 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %550		; <ubyte*>:1140 [#uses=1]
	load ubyte* %1140		; <ubyte>:1797 [#uses=1]
	seteq ubyte %1797, 0		; <bool>:783 [#uses=1]
	br bool %783, label %785, label %784

; <label>:784		; preds = %783, %784
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %547		; <ubyte*>:1141 [#uses=2]
	load ubyte* %1141		; <ubyte>:1798 [#uses=1]
	add ubyte %1798, 1		; <ubyte>:1799 [#uses=1]
	store ubyte %1799, ubyte* %1141
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %550		; <ubyte*>:1142 [#uses=2]
	load ubyte* %1142		; <ubyte>:1800 [#uses=2]
	add ubyte %1800, 255		; <ubyte>:1801 [#uses=1]
	store ubyte %1801, ubyte* %1142
	seteq ubyte %1800, 1		; <bool>:784 [#uses=1]
	br bool %784, label %785, label %784

; <label>:785		; preds = %783, %784
	add uint %546, 4294967096		; <uint>:551 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %551		; <ubyte*>:1143 [#uses=1]
	load ubyte* %1143		; <ubyte>:1802 [#uses=1]
	seteq ubyte %1802, 0		; <bool>:785 [#uses=1]
	br bool %785, label %787, label %786

; <label>:786		; preds = %785, %786
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %551		; <ubyte*>:1144 [#uses=2]
	load ubyte* %1144		; <ubyte>:1803 [#uses=2]
	add ubyte %1803, 255		; <ubyte>:1804 [#uses=1]
	store ubyte %1804, ubyte* %1144
	seteq ubyte %1803, 1		; <bool>:786 [#uses=1]
	br bool %786, label %787, label %786

; <label>:787		; preds = %785, %786
	add uint %546, 4294967213		; <uint>:552 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %552		; <ubyte*>:1145 [#uses=1]
	load ubyte* %1145		; <ubyte>:1805 [#uses=1]
	seteq ubyte %1805, 0		; <bool>:787 [#uses=1]
	br bool %787, label %789, label %788

; <label>:788		; preds = %787, %788
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %551		; <ubyte*>:1146 [#uses=2]
	load ubyte* %1146		; <ubyte>:1806 [#uses=1]
	add ubyte %1806, 1		; <ubyte>:1807 [#uses=1]
	store ubyte %1807, ubyte* %1146
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %552		; <ubyte*>:1147 [#uses=2]
	load ubyte* %1147		; <ubyte>:1808 [#uses=2]
	add ubyte %1808, 255		; <ubyte>:1809 [#uses=1]
	store ubyte %1809, ubyte* %1147
	seteq ubyte %1808, 1		; <bool>:788 [#uses=1]
	br bool %788, label %789, label %788

; <label>:789		; preds = %787, %788
	add uint %546, 4294967102		; <uint>:553 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %553		; <ubyte*>:1148 [#uses=1]
	load ubyte* %1148		; <ubyte>:1810 [#uses=1]
	seteq ubyte %1810, 0		; <bool>:789 [#uses=1]
	br bool %789, label %791, label %790

; <label>:790		; preds = %789, %790
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %553		; <ubyte*>:1149 [#uses=2]
	load ubyte* %1149		; <ubyte>:1811 [#uses=2]
	add ubyte %1811, 255		; <ubyte>:1812 [#uses=1]
	store ubyte %1812, ubyte* %1149
	seteq ubyte %1811, 1		; <bool>:790 [#uses=1]
	br bool %790, label %791, label %790

; <label>:791		; preds = %789, %790
	add uint %546, 4294967219		; <uint>:554 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %554		; <ubyte*>:1150 [#uses=1]
	load ubyte* %1150		; <ubyte>:1813 [#uses=1]
	seteq ubyte %1813, 0		; <bool>:791 [#uses=1]
	br bool %791, label %793, label %792

; <label>:792		; preds = %791, %792
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %553		; <ubyte*>:1151 [#uses=2]
	load ubyte* %1151		; <ubyte>:1814 [#uses=1]
	add ubyte %1814, 1		; <ubyte>:1815 [#uses=1]
	store ubyte %1815, ubyte* %1151
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %554		; <ubyte*>:1152 [#uses=2]
	load ubyte* %1152		; <ubyte>:1816 [#uses=2]
	add ubyte %1816, 255		; <ubyte>:1817 [#uses=1]
	store ubyte %1817, ubyte* %1152
	seteq ubyte %1816, 1		; <bool>:792 [#uses=1]
	br bool %792, label %793, label %792

; <label>:793		; preds = %791, %792
	add uint %546, 4294967108		; <uint>:555 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %555		; <ubyte*>:1153 [#uses=1]
	load ubyte* %1153		; <ubyte>:1818 [#uses=1]
	seteq ubyte %1818, 0		; <bool>:793 [#uses=1]
	br bool %793, label %795, label %794

; <label>:794		; preds = %793, %794
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %555		; <ubyte*>:1154 [#uses=2]
	load ubyte* %1154		; <ubyte>:1819 [#uses=2]
	add ubyte %1819, 255		; <ubyte>:1820 [#uses=1]
	store ubyte %1820, ubyte* %1154
	seteq ubyte %1819, 1		; <bool>:794 [#uses=1]
	br bool %794, label %795, label %794

; <label>:795		; preds = %793, %794
	add uint %546, 4294967225		; <uint>:556 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %556		; <ubyte*>:1155 [#uses=1]
	load ubyte* %1155		; <ubyte>:1821 [#uses=1]
	seteq ubyte %1821, 0		; <bool>:795 [#uses=1]
	br bool %795, label %797, label %796

; <label>:796		; preds = %795, %796
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %555		; <ubyte*>:1156 [#uses=2]
	load ubyte* %1156		; <ubyte>:1822 [#uses=1]
	add ubyte %1822, 1		; <ubyte>:1823 [#uses=1]
	store ubyte %1823, ubyte* %1156
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %556		; <ubyte*>:1157 [#uses=2]
	load ubyte* %1157		; <ubyte>:1824 [#uses=2]
	add ubyte %1824, 255		; <ubyte>:1825 [#uses=1]
	store ubyte %1825, ubyte* %1157
	seteq ubyte %1824, 1		; <bool>:796 [#uses=1]
	br bool %796, label %797, label %796

; <label>:797		; preds = %795, %796
	add uint %546, 4294967114		; <uint>:557 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %557		; <ubyte*>:1158 [#uses=1]
	load ubyte* %1158		; <ubyte>:1826 [#uses=1]
	seteq ubyte %1826, 0		; <bool>:797 [#uses=1]
	br bool %797, label %799, label %798

; <label>:798		; preds = %797, %798
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %557		; <ubyte*>:1159 [#uses=2]
	load ubyte* %1159		; <ubyte>:1827 [#uses=2]
	add ubyte %1827, 255		; <ubyte>:1828 [#uses=1]
	store ubyte %1828, ubyte* %1159
	seteq ubyte %1827, 1		; <bool>:798 [#uses=1]
	br bool %798, label %799, label %798

; <label>:799		; preds = %797, %798
	add uint %546, 4294967231		; <uint>:558 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %558		; <ubyte*>:1160 [#uses=1]
	load ubyte* %1160		; <ubyte>:1829 [#uses=1]
	seteq ubyte %1829, 0		; <bool>:799 [#uses=1]
	br bool %799, label %801, label %800

; <label>:800		; preds = %799, %800
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %557		; <ubyte*>:1161 [#uses=2]
	load ubyte* %1161		; <ubyte>:1830 [#uses=1]
	add ubyte %1830, 1		; <ubyte>:1831 [#uses=1]
	store ubyte %1831, ubyte* %1161
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %558		; <ubyte*>:1162 [#uses=2]
	load ubyte* %1162		; <ubyte>:1832 [#uses=2]
	add ubyte %1832, 255		; <ubyte>:1833 [#uses=1]
	store ubyte %1833, ubyte* %1162
	seteq ubyte %1832, 1		; <bool>:800 [#uses=1]
	br bool %800, label %801, label %800

; <label>:801		; preds = %799, %800
	add uint %546, 4294967120		; <uint>:559 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %559		; <ubyte*>:1163 [#uses=1]
	load ubyte* %1163		; <ubyte>:1834 [#uses=1]
	seteq ubyte %1834, 0		; <bool>:801 [#uses=1]
	br bool %801, label %803, label %802

; <label>:802		; preds = %801, %802
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %559		; <ubyte*>:1164 [#uses=2]
	load ubyte* %1164		; <ubyte>:1835 [#uses=2]
	add ubyte %1835, 255		; <ubyte>:1836 [#uses=1]
	store ubyte %1836, ubyte* %1164
	seteq ubyte %1835, 1		; <bool>:802 [#uses=1]
	br bool %802, label %803, label %802

; <label>:803		; preds = %801, %802
	add uint %546, 4294967237		; <uint>:560 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %560		; <ubyte*>:1165 [#uses=1]
	load ubyte* %1165		; <ubyte>:1837 [#uses=1]
	seteq ubyte %1837, 0		; <bool>:803 [#uses=1]
	br bool %803, label %805, label %804

; <label>:804		; preds = %803, %804
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %559		; <ubyte*>:1166 [#uses=2]
	load ubyte* %1166		; <ubyte>:1838 [#uses=1]
	add ubyte %1838, 1		; <ubyte>:1839 [#uses=1]
	store ubyte %1839, ubyte* %1166
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %560		; <ubyte*>:1167 [#uses=2]
	load ubyte* %1167		; <ubyte>:1840 [#uses=2]
	add ubyte %1840, 255		; <ubyte>:1841 [#uses=1]
	store ubyte %1841, ubyte* %1167
	seteq ubyte %1840, 1		; <bool>:804 [#uses=1]
	br bool %804, label %805, label %804

; <label>:805		; preds = %803, %804
	add uint %546, 4294967126		; <uint>:561 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %561		; <ubyte*>:1168 [#uses=1]
	load ubyte* %1168		; <ubyte>:1842 [#uses=1]
	seteq ubyte %1842, 0		; <bool>:805 [#uses=1]
	br bool %805, label %807, label %806

; <label>:806		; preds = %805, %806
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %561		; <ubyte*>:1169 [#uses=2]
	load ubyte* %1169		; <ubyte>:1843 [#uses=2]
	add ubyte %1843, 255		; <ubyte>:1844 [#uses=1]
	store ubyte %1844, ubyte* %1169
	seteq ubyte %1843, 1		; <bool>:806 [#uses=1]
	br bool %806, label %807, label %806

; <label>:807		; preds = %805, %806
	add uint %546, 4294967243		; <uint>:562 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %562		; <ubyte*>:1170 [#uses=1]
	load ubyte* %1170		; <ubyte>:1845 [#uses=1]
	seteq ubyte %1845, 0		; <bool>:807 [#uses=1]
	br bool %807, label %809, label %808

; <label>:808		; preds = %807, %808
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %561		; <ubyte*>:1171 [#uses=2]
	load ubyte* %1171		; <ubyte>:1846 [#uses=1]
	add ubyte %1846, 1		; <ubyte>:1847 [#uses=1]
	store ubyte %1847, ubyte* %1171
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %562		; <ubyte*>:1172 [#uses=2]
	load ubyte* %1172		; <ubyte>:1848 [#uses=2]
	add ubyte %1848, 255		; <ubyte>:1849 [#uses=1]
	store ubyte %1849, ubyte* %1172
	seteq ubyte %1848, 1		; <bool>:808 [#uses=1]
	br bool %808, label %809, label %808

; <label>:809		; preds = %807, %808
	add uint %546, 4294967132		; <uint>:563 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %563		; <ubyte*>:1173 [#uses=1]
	load ubyte* %1173		; <ubyte>:1850 [#uses=1]
	seteq ubyte %1850, 0		; <bool>:809 [#uses=1]
	br bool %809, label %811, label %810

; <label>:810		; preds = %809, %810
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %563		; <ubyte*>:1174 [#uses=2]
	load ubyte* %1174		; <ubyte>:1851 [#uses=2]
	add ubyte %1851, 255		; <ubyte>:1852 [#uses=1]
	store ubyte %1852, ubyte* %1174
	seteq ubyte %1851, 1		; <bool>:810 [#uses=1]
	br bool %810, label %811, label %810

; <label>:811		; preds = %809, %810
	add uint %546, 4294967249		; <uint>:564 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %564		; <ubyte*>:1175 [#uses=1]
	load ubyte* %1175		; <ubyte>:1853 [#uses=1]
	seteq ubyte %1853, 0		; <bool>:811 [#uses=1]
	br bool %811, label %813, label %812

; <label>:812		; preds = %811, %812
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %563		; <ubyte*>:1176 [#uses=2]
	load ubyte* %1176		; <ubyte>:1854 [#uses=1]
	add ubyte %1854, 1		; <ubyte>:1855 [#uses=1]
	store ubyte %1855, ubyte* %1176
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %564		; <ubyte*>:1177 [#uses=2]
	load ubyte* %1177		; <ubyte>:1856 [#uses=2]
	add ubyte %1856, 255		; <ubyte>:1857 [#uses=1]
	store ubyte %1857, ubyte* %1177
	seteq ubyte %1856, 1		; <bool>:812 [#uses=1]
	br bool %812, label %813, label %812

; <label>:813		; preds = %811, %812
	add uint %546, 4294967138		; <uint>:565 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %565		; <ubyte*>:1178 [#uses=1]
	load ubyte* %1178		; <ubyte>:1858 [#uses=1]
	seteq ubyte %1858, 0		; <bool>:813 [#uses=1]
	br bool %813, label %815, label %814

; <label>:814		; preds = %813, %814
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %565		; <ubyte*>:1179 [#uses=2]
	load ubyte* %1179		; <ubyte>:1859 [#uses=2]
	add ubyte %1859, 255		; <ubyte>:1860 [#uses=1]
	store ubyte %1860, ubyte* %1179
	seteq ubyte %1859, 1		; <bool>:814 [#uses=1]
	br bool %814, label %815, label %814

; <label>:815		; preds = %813, %814
	add uint %546, 4294967255		; <uint>:566 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %566		; <ubyte*>:1180 [#uses=1]
	load ubyte* %1180		; <ubyte>:1861 [#uses=1]
	seteq ubyte %1861, 0		; <bool>:815 [#uses=1]
	br bool %815, label %817, label %816

; <label>:816		; preds = %815, %816
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %565		; <ubyte*>:1181 [#uses=2]
	load ubyte* %1181		; <ubyte>:1862 [#uses=1]
	add ubyte %1862, 1		; <ubyte>:1863 [#uses=1]
	store ubyte %1863, ubyte* %1181
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %566		; <ubyte*>:1182 [#uses=2]
	load ubyte* %1182		; <ubyte>:1864 [#uses=2]
	add ubyte %1864, 255		; <ubyte>:1865 [#uses=1]
	store ubyte %1865, ubyte* %1182
	seteq ubyte %1864, 1		; <bool>:816 [#uses=1]
	br bool %816, label %817, label %816

; <label>:817		; preds = %815, %816
	add uint %546, 4294967144		; <uint>:567 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %567		; <ubyte*>:1183 [#uses=1]
	load ubyte* %1183		; <ubyte>:1866 [#uses=1]
	seteq ubyte %1866, 0		; <bool>:817 [#uses=1]
	br bool %817, label %819, label %818

; <label>:818		; preds = %817, %818
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %567		; <ubyte*>:1184 [#uses=2]
	load ubyte* %1184		; <ubyte>:1867 [#uses=2]
	add ubyte %1867, 255		; <ubyte>:1868 [#uses=1]
	store ubyte %1868, ubyte* %1184
	seteq ubyte %1867, 1		; <bool>:818 [#uses=1]
	br bool %818, label %819, label %818

; <label>:819		; preds = %817, %818
	add uint %546, 4294967261		; <uint>:568 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %568		; <ubyte*>:1185 [#uses=1]
	load ubyte* %1185		; <ubyte>:1869 [#uses=1]
	seteq ubyte %1869, 0		; <bool>:819 [#uses=1]
	br bool %819, label %821, label %820

; <label>:820		; preds = %819, %820
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %567		; <ubyte*>:1186 [#uses=2]
	load ubyte* %1186		; <ubyte>:1870 [#uses=1]
	add ubyte %1870, 1		; <ubyte>:1871 [#uses=1]
	store ubyte %1871, ubyte* %1186
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %568		; <ubyte*>:1187 [#uses=2]
	load ubyte* %1187		; <ubyte>:1872 [#uses=2]
	add ubyte %1872, 255		; <ubyte>:1873 [#uses=1]
	store ubyte %1873, ubyte* %1187
	seteq ubyte %1872, 1		; <bool>:820 [#uses=1]
	br bool %820, label %821, label %820

; <label>:821		; preds = %819, %820
	add uint %546, 4294967150		; <uint>:569 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %569		; <ubyte*>:1188 [#uses=1]
	load ubyte* %1188		; <ubyte>:1874 [#uses=1]
	seteq ubyte %1874, 0		; <bool>:821 [#uses=1]
	br bool %821, label %823, label %822

; <label>:822		; preds = %821, %822
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %569		; <ubyte*>:1189 [#uses=2]
	load ubyte* %1189		; <ubyte>:1875 [#uses=2]
	add ubyte %1875, 255		; <ubyte>:1876 [#uses=1]
	store ubyte %1876, ubyte* %1189
	seteq ubyte %1875, 1		; <bool>:822 [#uses=1]
	br bool %822, label %823, label %822

; <label>:823		; preds = %821, %822
	add uint %546, 4294967267		; <uint>:570 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %570		; <ubyte*>:1190 [#uses=1]
	load ubyte* %1190		; <ubyte>:1877 [#uses=1]
	seteq ubyte %1877, 0		; <bool>:823 [#uses=1]
	br bool %823, label %825, label %824

; <label>:824		; preds = %823, %824
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %569		; <ubyte*>:1191 [#uses=2]
	load ubyte* %1191		; <ubyte>:1878 [#uses=1]
	add ubyte %1878, 1		; <ubyte>:1879 [#uses=1]
	store ubyte %1879, ubyte* %1191
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %570		; <ubyte*>:1192 [#uses=2]
	load ubyte* %1192		; <ubyte>:1880 [#uses=2]
	add ubyte %1880, 255		; <ubyte>:1881 [#uses=1]
	store ubyte %1881, ubyte* %1192
	seteq ubyte %1880, 1		; <bool>:824 [#uses=1]
	br bool %824, label %825, label %824

; <label>:825		; preds = %823, %824
	add uint %546, 4294967156		; <uint>:571 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %571		; <ubyte*>:1193 [#uses=1]
	load ubyte* %1193		; <ubyte>:1882 [#uses=1]
	seteq ubyte %1882, 0		; <bool>:825 [#uses=1]
	br bool %825, label %827, label %826

; <label>:826		; preds = %825, %826
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %571		; <ubyte*>:1194 [#uses=2]
	load ubyte* %1194		; <ubyte>:1883 [#uses=2]
	add ubyte %1883, 255		; <ubyte>:1884 [#uses=1]
	store ubyte %1884, ubyte* %1194
	seteq ubyte %1883, 1		; <bool>:826 [#uses=1]
	br bool %826, label %827, label %826

; <label>:827		; preds = %825, %826
	add uint %546, 4294967273		; <uint>:572 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %572		; <ubyte*>:1195 [#uses=1]
	load ubyte* %1195		; <ubyte>:1885 [#uses=1]
	seteq ubyte %1885, 0		; <bool>:827 [#uses=1]
	br bool %827, label %829, label %828

; <label>:828		; preds = %827, %828
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %571		; <ubyte*>:1196 [#uses=2]
	load ubyte* %1196		; <ubyte>:1886 [#uses=1]
	add ubyte %1886, 1		; <ubyte>:1887 [#uses=1]
	store ubyte %1887, ubyte* %1196
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %572		; <ubyte*>:1197 [#uses=2]
	load ubyte* %1197		; <ubyte>:1888 [#uses=2]
	add ubyte %1888, 255		; <ubyte>:1889 [#uses=1]
	store ubyte %1889, ubyte* %1197
	seteq ubyte %1888, 1		; <bool>:828 [#uses=1]
	br bool %828, label %829, label %828

; <label>:829		; preds = %827, %828
	add uint %546, 4294967162		; <uint>:573 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %573		; <ubyte*>:1198 [#uses=1]
	load ubyte* %1198		; <ubyte>:1890 [#uses=1]
	seteq ubyte %1890, 0		; <bool>:829 [#uses=1]
	br bool %829, label %831, label %830

; <label>:830		; preds = %829, %830
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %573		; <ubyte*>:1199 [#uses=2]
	load ubyte* %1199		; <ubyte>:1891 [#uses=2]
	add ubyte %1891, 255		; <ubyte>:1892 [#uses=1]
	store ubyte %1892, ubyte* %1199
	seteq ubyte %1891, 1		; <bool>:830 [#uses=1]
	br bool %830, label %831, label %830

; <label>:831		; preds = %829, %830
	add uint %546, 4294967279		; <uint>:574 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %574		; <ubyte*>:1200 [#uses=1]
	load ubyte* %1200		; <ubyte>:1893 [#uses=1]
	seteq ubyte %1893, 0		; <bool>:831 [#uses=1]
	br bool %831, label %833, label %832

; <label>:832		; preds = %831, %832
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %573		; <ubyte*>:1201 [#uses=2]
	load ubyte* %1201		; <ubyte>:1894 [#uses=1]
	add ubyte %1894, 1		; <ubyte>:1895 [#uses=1]
	store ubyte %1895, ubyte* %1201
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %574		; <ubyte*>:1202 [#uses=2]
	load ubyte* %1202		; <ubyte>:1896 [#uses=2]
	add ubyte %1896, 255		; <ubyte>:1897 [#uses=1]
	store ubyte %1897, ubyte* %1202
	seteq ubyte %1896, 1		; <bool>:832 [#uses=1]
	br bool %832, label %833, label %832

; <label>:833		; preds = %831, %832
	add uint %546, 4294967168		; <uint>:575 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %575		; <ubyte*>:1203 [#uses=1]
	load ubyte* %1203		; <ubyte>:1898 [#uses=1]
	seteq ubyte %1898, 0		; <bool>:833 [#uses=1]
	br bool %833, label %835, label %834

; <label>:834		; preds = %833, %834
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %575		; <ubyte*>:1204 [#uses=2]
	load ubyte* %1204		; <ubyte>:1899 [#uses=2]
	add ubyte %1899, 255		; <ubyte>:1900 [#uses=1]
	store ubyte %1900, ubyte* %1204
	seteq ubyte %1899, 1		; <bool>:834 [#uses=1]
	br bool %834, label %835, label %834

; <label>:835		; preds = %833, %834
	add uint %546, 4294967285		; <uint>:576 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %576		; <ubyte*>:1205 [#uses=1]
	load ubyte* %1205		; <ubyte>:1901 [#uses=1]
	seteq ubyte %1901, 0		; <bool>:835 [#uses=1]
	br bool %835, label %837, label %836

; <label>:836		; preds = %835, %836
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %575		; <ubyte*>:1206 [#uses=2]
	load ubyte* %1206		; <ubyte>:1902 [#uses=1]
	add ubyte %1902, 1		; <ubyte>:1903 [#uses=1]
	store ubyte %1903, ubyte* %1206
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %576		; <ubyte*>:1207 [#uses=2]
	load ubyte* %1207		; <ubyte>:1904 [#uses=2]
	add ubyte %1904, 255		; <ubyte>:1905 [#uses=1]
	store ubyte %1905, ubyte* %1207
	seteq ubyte %1904, 1		; <bool>:836 [#uses=1]
	br bool %836, label %837, label %836

; <label>:837		; preds = %835, %836
	add uint %546, 4294967174		; <uint>:577 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %577		; <ubyte*>:1208 [#uses=1]
	load ubyte* %1208		; <ubyte>:1906 [#uses=1]
	seteq ubyte %1906, 0		; <bool>:837 [#uses=1]
	br bool %837, label %839, label %838

; <label>:838		; preds = %837, %838
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %577		; <ubyte*>:1209 [#uses=2]
	load ubyte* %1209		; <ubyte>:1907 [#uses=2]
	add ubyte %1907, 255		; <ubyte>:1908 [#uses=1]
	store ubyte %1908, ubyte* %1209
	seteq ubyte %1907, 1		; <bool>:838 [#uses=1]
	br bool %838, label %839, label %838

; <label>:839		; preds = %837, %838
	add uint %546, 4294967291		; <uint>:578 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %578		; <ubyte*>:1210 [#uses=1]
	load ubyte* %1210		; <ubyte>:1909 [#uses=1]
	seteq ubyte %1909, 0		; <bool>:839 [#uses=1]
	br bool %839, label %841, label %840

; <label>:840		; preds = %839, %840
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %577		; <ubyte*>:1211 [#uses=2]
	load ubyte* %1211		; <ubyte>:1910 [#uses=1]
	add ubyte %1910, 1		; <ubyte>:1911 [#uses=1]
	store ubyte %1911, ubyte* %1211
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %578		; <ubyte*>:1212 [#uses=2]
	load ubyte* %1212		; <ubyte>:1912 [#uses=2]
	add ubyte %1912, 255		; <ubyte>:1913 [#uses=1]
	store ubyte %1913, ubyte* %1212
	seteq ubyte %1912, 1		; <bool>:840 [#uses=1]
	br bool %840, label %841, label %840

; <label>:841		; preds = %839, %840
	add uint %546, 4294967197		; <uint>:579 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %579		; <ubyte*>:1213 [#uses=1]
	load ubyte* %1213		; <ubyte>:1914 [#uses=1]
	seteq ubyte %1914, 0		; <bool>:841 [#uses=1]
	br bool %841, label %843, label %842

; <label>:842		; preds = %841, %842
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %579		; <ubyte*>:1214 [#uses=2]
	load ubyte* %1214		; <ubyte>:1915 [#uses=2]
	add ubyte %1915, 255		; <ubyte>:1916 [#uses=1]
	store ubyte %1916, ubyte* %1214
	seteq ubyte %1915, 1		; <bool>:842 [#uses=1]
	br bool %842, label %843, label %842

; <label>:843		; preds = %841, %842
	add uint %546, 4294967195		; <uint>:580 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %580		; <ubyte*>:1215 [#uses=1]
	load ubyte* %1215		; <ubyte>:1917 [#uses=1]
	seteq ubyte %1917, 0		; <bool>:843 [#uses=1]
	br bool %843, label %845, label %844

; <label>:844		; preds = %843, %844
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %580		; <ubyte*>:1216 [#uses=2]
	load ubyte* %1216		; <ubyte>:1918 [#uses=2]
	add ubyte %1918, 255		; <ubyte>:1919 [#uses=1]
	store ubyte %1919, ubyte* %1216
	seteq ubyte %1918, 1		; <bool>:844 [#uses=1]
	br bool %844, label %845, label %844

; <label>:845		; preds = %843, %844
	add uint %546, 4294967184		; <uint>:581 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %581		; <ubyte*>:1217 [#uses=2]
	load ubyte* %1217		; <ubyte>:1920 [#uses=1]
	add ubyte %1920, 8		; <ubyte>:1921 [#uses=1]
	store ubyte %1921, ubyte* %1217
	add uint %546, 4294967186		; <uint>:582 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %582		; <ubyte*>:1218 [#uses=1]
	load ubyte* %1218		; <ubyte>:1922 [#uses=1]
	seteq ubyte %1922, 0		; <bool>:845 [#uses=1]
	br bool %845, label %781, label %780

; <label>:846		; preds = %589, %949
	phi uint [ %411, %589 ], [ %658, %949 ]		; <uint>:583 [#uses=74]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %583		; <ubyte*>:1219 [#uses=2]
	load ubyte* %1219		; <ubyte>:1923 [#uses=1]
	add ubyte %1923, 255		; <ubyte>:1924 [#uses=1]
	store ubyte %1924, ubyte* %1219
	add uint %583, 10		; <uint>:584 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %584		; <ubyte*>:1220 [#uses=1]
	load ubyte* %1220		; <ubyte>:1925 [#uses=1]
	seteq ubyte %1925, 0		; <bool>:846 [#uses=1]
	br bool %846, label %849, label %848

; <label>:847		; preds = %589, %949
	phi uint [ %411, %589 ], [ %658, %949 ]		; <uint>:585 [#uses=1]
	add uint %585, 4294967295		; <uint>:586 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %586		; <ubyte*>:1221 [#uses=1]
	load ubyte* %1221		; <ubyte>:1926 [#uses=1]
	seteq ubyte %1926, 0		; <bool>:847 [#uses=1]
	br bool %847, label %587, label %586

; <label>:848		; preds = %846, %848
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %584		; <ubyte*>:1222 [#uses=2]
	load ubyte* %1222		; <ubyte>:1927 [#uses=2]
	add ubyte %1927, 255		; <ubyte>:1928 [#uses=1]
	store ubyte %1928, ubyte* %1222
	seteq ubyte %1927, 1		; <bool>:848 [#uses=1]
	br bool %848, label %849, label %848

; <label>:849		; preds = %846, %848
	add uint %583, 4		; <uint>:587 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %587		; <ubyte*>:1223 [#uses=1]
	load ubyte* %1223		; <ubyte>:1929 [#uses=1]
	seteq ubyte %1929, 0		; <bool>:849 [#uses=1]
	br bool %849, label %851, label %850

; <label>:850		; preds = %849, %850
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %587		; <ubyte*>:1224 [#uses=2]
	load ubyte* %1224		; <ubyte>:1930 [#uses=1]
	add ubyte %1930, 255		; <ubyte>:1931 [#uses=1]
	store ubyte %1931, ubyte* %1224
	add uint %583, 5		; <uint>:588 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %588		; <ubyte*>:1225 [#uses=2]
	load ubyte* %1225		; <ubyte>:1932 [#uses=1]
	add ubyte %1932, 1		; <ubyte>:1933 [#uses=1]
	store ubyte %1933, ubyte* %1225
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %584		; <ubyte*>:1226 [#uses=2]
	load ubyte* %1226		; <ubyte>:1934 [#uses=1]
	add ubyte %1934, 1		; <ubyte>:1935 [#uses=1]
	store ubyte %1935, ubyte* %1226
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %587		; <ubyte*>:1227 [#uses=1]
	load ubyte* %1227		; <ubyte>:1936 [#uses=1]
	seteq ubyte %1936, 0		; <bool>:850 [#uses=1]
	br bool %850, label %851, label %850

; <label>:851		; preds = %849, %850
	add uint %583, 5		; <uint>:589 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %589		; <ubyte*>:1228 [#uses=1]
	load ubyte* %1228		; <ubyte>:1937 [#uses=1]
	seteq ubyte %1937, 0		; <bool>:851 [#uses=1]
	br bool %851, label %853, label %852

; <label>:852		; preds = %851, %852
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %587		; <ubyte*>:1229 [#uses=2]
	load ubyte* %1229		; <ubyte>:1938 [#uses=1]
	add ubyte %1938, 1		; <ubyte>:1939 [#uses=1]
	store ubyte %1939, ubyte* %1229
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %589		; <ubyte*>:1230 [#uses=2]
	load ubyte* %1230		; <ubyte>:1940 [#uses=2]
	add ubyte %1940, 255		; <ubyte>:1941 [#uses=1]
	store ubyte %1941, ubyte* %1230
	seteq ubyte %1940, 1		; <bool>:852 [#uses=1]
	br bool %852, label %853, label %852

; <label>:853		; preds = %851, %852
	add uint %583, 12		; <uint>:590 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %590		; <ubyte*>:1231 [#uses=1]
	load ubyte* %1231		; <ubyte>:1942 [#uses=1]
	seteq ubyte %1942, 0		; <bool>:853 [#uses=1]
	br bool %853, label %855, label %854

; <label>:854		; preds = %853, %854
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %590		; <ubyte*>:1232 [#uses=2]
	load ubyte* %1232		; <ubyte>:1943 [#uses=2]
	add ubyte %1943, 255		; <ubyte>:1944 [#uses=1]
	store ubyte %1944, ubyte* %1232
	seteq ubyte %1943, 1		; <bool>:854 [#uses=1]
	br bool %854, label %855, label %854

; <label>:855		; preds = %853, %854
	add uint %583, 4294967191		; <uint>:591 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %591		; <ubyte*>:1233 [#uses=1]
	load ubyte* %1233		; <ubyte>:1945 [#uses=1]
	seteq ubyte %1945, 0		; <bool>:855 [#uses=1]
	br bool %855, label %857, label %856

; <label>:856		; preds = %855, %856
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %591		; <ubyte*>:1234 [#uses=2]
	load ubyte* %1234		; <ubyte>:1946 [#uses=1]
	add ubyte %1946, 255		; <ubyte>:1947 [#uses=1]
	store ubyte %1947, ubyte* %1234
	add uint %583, 4294967192		; <uint>:592 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %592		; <ubyte*>:1235 [#uses=2]
	load ubyte* %1235		; <ubyte>:1948 [#uses=1]
	add ubyte %1948, 1		; <ubyte>:1949 [#uses=1]
	store ubyte %1949, ubyte* %1235
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %590		; <ubyte*>:1236 [#uses=2]
	load ubyte* %1236		; <ubyte>:1950 [#uses=1]
	add ubyte %1950, 1		; <ubyte>:1951 [#uses=1]
	store ubyte %1951, ubyte* %1236
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %591		; <ubyte*>:1237 [#uses=1]
	load ubyte* %1237		; <ubyte>:1952 [#uses=1]
	seteq ubyte %1952, 0		; <bool>:856 [#uses=1]
	br bool %856, label %857, label %856

; <label>:857		; preds = %855, %856
	add uint %583, 4294967192		; <uint>:593 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %593		; <ubyte*>:1238 [#uses=1]
	load ubyte* %1238		; <ubyte>:1953 [#uses=1]
	seteq ubyte %1953, 0		; <bool>:857 [#uses=1]
	br bool %857, label %859, label %858

; <label>:858		; preds = %857, %858
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %591		; <ubyte*>:1239 [#uses=2]
	load ubyte* %1239		; <ubyte>:1954 [#uses=1]
	add ubyte %1954, 1		; <ubyte>:1955 [#uses=1]
	store ubyte %1955, ubyte* %1239
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %593		; <ubyte*>:1240 [#uses=2]
	load ubyte* %1240		; <ubyte>:1956 [#uses=2]
	add ubyte %1956, 255		; <ubyte>:1957 [#uses=1]
	store ubyte %1957, ubyte* %1240
	seteq ubyte %1956, 1		; <bool>:858 [#uses=1]
	br bool %858, label %859, label %858

; <label>:859		; preds = %857, %858
	add uint %583, 22		; <uint>:594 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %594		; <ubyte*>:1241 [#uses=1]
	load ubyte* %1241		; <ubyte>:1958 [#uses=1]
	seteq ubyte %1958, 0		; <bool>:859 [#uses=1]
	br bool %859, label %861, label %860

; <label>:860		; preds = %859, %860
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %594		; <ubyte*>:1242 [#uses=2]
	load ubyte* %1242		; <ubyte>:1959 [#uses=2]
	add ubyte %1959, 255		; <ubyte>:1960 [#uses=1]
	store ubyte %1960, ubyte* %1242
	seteq ubyte %1959, 1		; <bool>:860 [#uses=1]
	br bool %860, label %861, label %860

; <label>:861		; preds = %859, %860
	add uint %583, 4294967201		; <uint>:595 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %595		; <ubyte*>:1243 [#uses=1]
	load ubyte* %1243		; <ubyte>:1961 [#uses=1]
	seteq ubyte %1961, 0		; <bool>:861 [#uses=1]
	br bool %861, label %863, label %862

; <label>:862		; preds = %861, %862
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %595		; <ubyte*>:1244 [#uses=2]
	load ubyte* %1244		; <ubyte>:1962 [#uses=1]
	add ubyte %1962, 255		; <ubyte>:1963 [#uses=1]
	store ubyte %1963, ubyte* %1244
	add uint %583, 4294967202		; <uint>:596 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %596		; <ubyte*>:1245 [#uses=2]
	load ubyte* %1245		; <ubyte>:1964 [#uses=1]
	add ubyte %1964, 1		; <ubyte>:1965 [#uses=1]
	store ubyte %1965, ubyte* %1245
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %594		; <ubyte*>:1246 [#uses=2]
	load ubyte* %1246		; <ubyte>:1966 [#uses=1]
	add ubyte %1966, 1		; <ubyte>:1967 [#uses=1]
	store ubyte %1967, ubyte* %1246
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %595		; <ubyte*>:1247 [#uses=1]
	load ubyte* %1247		; <ubyte>:1968 [#uses=1]
	seteq ubyte %1968, 0		; <bool>:862 [#uses=1]
	br bool %862, label %863, label %862

; <label>:863		; preds = %861, %862
	add uint %583, 4294967202		; <uint>:597 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %597		; <ubyte*>:1248 [#uses=1]
	load ubyte* %1248		; <ubyte>:1969 [#uses=1]
	seteq ubyte %1969, 0		; <bool>:863 [#uses=1]
	br bool %863, label %865, label %864

; <label>:864		; preds = %863, %864
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %595		; <ubyte*>:1249 [#uses=2]
	load ubyte* %1249		; <ubyte>:1970 [#uses=1]
	add ubyte %1970, 1		; <ubyte>:1971 [#uses=1]
	store ubyte %1971, ubyte* %1249
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %597		; <ubyte*>:1250 [#uses=2]
	load ubyte* %1250		; <ubyte>:1972 [#uses=2]
	add ubyte %1972, 255		; <ubyte>:1973 [#uses=1]
	store ubyte %1973, ubyte* %1250
	seteq ubyte %1972, 1		; <bool>:864 [#uses=1]
	br bool %864, label %865, label %864

; <label>:865		; preds = %863, %864
	add uint %583, 28		; <uint>:598 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %598		; <ubyte*>:1251 [#uses=1]
	load ubyte* %1251		; <ubyte>:1974 [#uses=1]
	seteq ubyte %1974, 0		; <bool>:865 [#uses=1]
	br bool %865, label %867, label %866

; <label>:866		; preds = %865, %866
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %598		; <ubyte*>:1252 [#uses=2]
	load ubyte* %1252		; <ubyte>:1975 [#uses=2]
	add ubyte %1975, 255		; <ubyte>:1976 [#uses=1]
	store ubyte %1976, ubyte* %1252
	seteq ubyte %1975, 1		; <bool>:866 [#uses=1]
	br bool %866, label %867, label %866

; <label>:867		; preds = %865, %866
	add uint %583, 4294967207		; <uint>:599 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %599		; <ubyte*>:1253 [#uses=1]
	load ubyte* %1253		; <ubyte>:1977 [#uses=1]
	seteq ubyte %1977, 0		; <bool>:867 [#uses=1]
	br bool %867, label %869, label %868

; <label>:868		; preds = %867, %868
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %599		; <ubyte*>:1254 [#uses=2]
	load ubyte* %1254		; <ubyte>:1978 [#uses=1]
	add ubyte %1978, 255		; <ubyte>:1979 [#uses=1]
	store ubyte %1979, ubyte* %1254
	add uint %583, 4294967208		; <uint>:600 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %600		; <ubyte*>:1255 [#uses=2]
	load ubyte* %1255		; <ubyte>:1980 [#uses=1]
	add ubyte %1980, 1		; <ubyte>:1981 [#uses=1]
	store ubyte %1981, ubyte* %1255
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %598		; <ubyte*>:1256 [#uses=2]
	load ubyte* %1256		; <ubyte>:1982 [#uses=1]
	add ubyte %1982, 1		; <ubyte>:1983 [#uses=1]
	store ubyte %1983, ubyte* %1256
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %599		; <ubyte*>:1257 [#uses=1]
	load ubyte* %1257		; <ubyte>:1984 [#uses=1]
	seteq ubyte %1984, 0		; <bool>:868 [#uses=1]
	br bool %868, label %869, label %868

; <label>:869		; preds = %867, %868
	add uint %583, 4294967208		; <uint>:601 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %601		; <ubyte*>:1258 [#uses=1]
	load ubyte* %1258		; <ubyte>:1985 [#uses=1]
	seteq ubyte %1985, 0		; <bool>:869 [#uses=1]
	br bool %869, label %871, label %870

; <label>:870		; preds = %869, %870
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %599		; <ubyte*>:1259 [#uses=2]
	load ubyte* %1259		; <ubyte>:1986 [#uses=1]
	add ubyte %1986, 1		; <ubyte>:1987 [#uses=1]
	store ubyte %1987, ubyte* %1259
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %601		; <ubyte*>:1260 [#uses=2]
	load ubyte* %1260		; <ubyte>:1988 [#uses=2]
	add ubyte %1988, 255		; <ubyte>:1989 [#uses=1]
	store ubyte %1989, ubyte* %1260
	seteq ubyte %1988, 1		; <bool>:870 [#uses=1]
	br bool %870, label %871, label %870

; <label>:871		; preds = %869, %870
	add uint %583, 34		; <uint>:602 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %602		; <ubyte*>:1261 [#uses=1]
	load ubyte* %1261		; <ubyte>:1990 [#uses=1]
	seteq ubyte %1990, 0		; <bool>:871 [#uses=1]
	br bool %871, label %873, label %872

; <label>:872		; preds = %871, %872
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %602		; <ubyte*>:1262 [#uses=2]
	load ubyte* %1262		; <ubyte>:1991 [#uses=2]
	add ubyte %1991, 255		; <ubyte>:1992 [#uses=1]
	store ubyte %1992, ubyte* %1262
	seteq ubyte %1991, 1		; <bool>:872 [#uses=1]
	br bool %872, label %873, label %872

; <label>:873		; preds = %871, %872
	add uint %583, 4294967213		; <uint>:603 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %603		; <ubyte*>:1263 [#uses=1]
	load ubyte* %1263		; <ubyte>:1993 [#uses=1]
	seteq ubyte %1993, 0		; <bool>:873 [#uses=1]
	br bool %873, label %875, label %874

; <label>:874		; preds = %873, %874
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %603		; <ubyte*>:1264 [#uses=2]
	load ubyte* %1264		; <ubyte>:1994 [#uses=1]
	add ubyte %1994, 255		; <ubyte>:1995 [#uses=1]
	store ubyte %1995, ubyte* %1264
	add uint %583, 4294967214		; <uint>:604 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %604		; <ubyte*>:1265 [#uses=2]
	load ubyte* %1265		; <ubyte>:1996 [#uses=1]
	add ubyte %1996, 1		; <ubyte>:1997 [#uses=1]
	store ubyte %1997, ubyte* %1265
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %602		; <ubyte*>:1266 [#uses=2]
	load ubyte* %1266		; <ubyte>:1998 [#uses=1]
	add ubyte %1998, 1		; <ubyte>:1999 [#uses=1]
	store ubyte %1999, ubyte* %1266
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %603		; <ubyte*>:1267 [#uses=1]
	load ubyte* %1267		; <ubyte>:2000 [#uses=1]
	seteq ubyte %2000, 0		; <bool>:874 [#uses=1]
	br bool %874, label %875, label %874

; <label>:875		; preds = %873, %874
	add uint %583, 4294967214		; <uint>:605 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %605		; <ubyte*>:1268 [#uses=1]
	load ubyte* %1268		; <ubyte>:2001 [#uses=1]
	seteq ubyte %2001, 0		; <bool>:875 [#uses=1]
	br bool %875, label %877, label %876

; <label>:876		; preds = %875, %876
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %603		; <ubyte*>:1269 [#uses=2]
	load ubyte* %1269		; <ubyte>:2002 [#uses=1]
	add ubyte %2002, 1		; <ubyte>:2003 [#uses=1]
	store ubyte %2003, ubyte* %1269
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %605		; <ubyte*>:1270 [#uses=2]
	load ubyte* %1270		; <ubyte>:2004 [#uses=2]
	add ubyte %2004, 255		; <ubyte>:2005 [#uses=1]
	store ubyte %2005, ubyte* %1270
	seteq ubyte %2004, 1		; <bool>:876 [#uses=1]
	br bool %876, label %877, label %876

; <label>:877		; preds = %875, %876
	add uint %583, 40		; <uint>:606 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %606		; <ubyte*>:1271 [#uses=1]
	load ubyte* %1271		; <ubyte>:2006 [#uses=1]
	seteq ubyte %2006, 0		; <bool>:877 [#uses=1]
	br bool %877, label %879, label %878

; <label>:878		; preds = %877, %878
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %606		; <ubyte*>:1272 [#uses=2]
	load ubyte* %1272		; <ubyte>:2007 [#uses=2]
	add ubyte %2007, 255		; <ubyte>:2008 [#uses=1]
	store ubyte %2008, ubyte* %1272
	seteq ubyte %2007, 1		; <bool>:878 [#uses=1]
	br bool %878, label %879, label %878

; <label>:879		; preds = %877, %878
	add uint %583, 4294967219		; <uint>:607 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %607		; <ubyte*>:1273 [#uses=1]
	load ubyte* %1273		; <ubyte>:2009 [#uses=1]
	seteq ubyte %2009, 0		; <bool>:879 [#uses=1]
	br bool %879, label %881, label %880

; <label>:880		; preds = %879, %880
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %607		; <ubyte*>:1274 [#uses=2]
	load ubyte* %1274		; <ubyte>:2010 [#uses=1]
	add ubyte %2010, 255		; <ubyte>:2011 [#uses=1]
	store ubyte %2011, ubyte* %1274
	add uint %583, 4294967220		; <uint>:608 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %608		; <ubyte*>:1275 [#uses=2]
	load ubyte* %1275		; <ubyte>:2012 [#uses=1]
	add ubyte %2012, 1		; <ubyte>:2013 [#uses=1]
	store ubyte %2013, ubyte* %1275
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %606		; <ubyte*>:1276 [#uses=2]
	load ubyte* %1276		; <ubyte>:2014 [#uses=1]
	add ubyte %2014, 1		; <ubyte>:2015 [#uses=1]
	store ubyte %2015, ubyte* %1276
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %607		; <ubyte*>:1277 [#uses=1]
	load ubyte* %1277		; <ubyte>:2016 [#uses=1]
	seteq ubyte %2016, 0		; <bool>:880 [#uses=1]
	br bool %880, label %881, label %880

; <label>:881		; preds = %879, %880
	add uint %583, 4294967220		; <uint>:609 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %609		; <ubyte*>:1278 [#uses=1]
	load ubyte* %1278		; <ubyte>:2017 [#uses=1]
	seteq ubyte %2017, 0		; <bool>:881 [#uses=1]
	br bool %881, label %883, label %882

; <label>:882		; preds = %881, %882
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %607		; <ubyte*>:1279 [#uses=2]
	load ubyte* %1279		; <ubyte>:2018 [#uses=1]
	add ubyte %2018, 1		; <ubyte>:2019 [#uses=1]
	store ubyte %2019, ubyte* %1279
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %609		; <ubyte*>:1280 [#uses=2]
	load ubyte* %1280		; <ubyte>:2020 [#uses=2]
	add ubyte %2020, 255		; <ubyte>:2021 [#uses=1]
	store ubyte %2021, ubyte* %1280
	seteq ubyte %2020, 1		; <bool>:882 [#uses=1]
	br bool %882, label %883, label %882

; <label>:883		; preds = %881, %882
	add uint %583, 46		; <uint>:610 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %610		; <ubyte*>:1281 [#uses=1]
	load ubyte* %1281		; <ubyte>:2022 [#uses=1]
	seteq ubyte %2022, 0		; <bool>:883 [#uses=1]
	br bool %883, label %885, label %884

; <label>:884		; preds = %883, %884
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %610		; <ubyte*>:1282 [#uses=2]
	load ubyte* %1282		; <ubyte>:2023 [#uses=2]
	add ubyte %2023, 255		; <ubyte>:2024 [#uses=1]
	store ubyte %2024, ubyte* %1282
	seteq ubyte %2023, 1		; <bool>:884 [#uses=1]
	br bool %884, label %885, label %884

; <label>:885		; preds = %883, %884
	add uint %583, 4294967225		; <uint>:611 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %611		; <ubyte*>:1283 [#uses=1]
	load ubyte* %1283		; <ubyte>:2025 [#uses=1]
	seteq ubyte %2025, 0		; <bool>:885 [#uses=1]
	br bool %885, label %887, label %886

; <label>:886		; preds = %885, %886
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %611		; <ubyte*>:1284 [#uses=2]
	load ubyte* %1284		; <ubyte>:2026 [#uses=1]
	add ubyte %2026, 255		; <ubyte>:2027 [#uses=1]
	store ubyte %2027, ubyte* %1284
	add uint %583, 4294967226		; <uint>:612 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %612		; <ubyte*>:1285 [#uses=2]
	load ubyte* %1285		; <ubyte>:2028 [#uses=1]
	add ubyte %2028, 1		; <ubyte>:2029 [#uses=1]
	store ubyte %2029, ubyte* %1285
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %610		; <ubyte*>:1286 [#uses=2]
	load ubyte* %1286		; <ubyte>:2030 [#uses=1]
	add ubyte %2030, 1		; <ubyte>:2031 [#uses=1]
	store ubyte %2031, ubyte* %1286
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %611		; <ubyte*>:1287 [#uses=1]
	load ubyte* %1287		; <ubyte>:2032 [#uses=1]
	seteq ubyte %2032, 0		; <bool>:886 [#uses=1]
	br bool %886, label %887, label %886

; <label>:887		; preds = %885, %886
	add uint %583, 4294967226		; <uint>:613 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %613		; <ubyte*>:1288 [#uses=1]
	load ubyte* %1288		; <ubyte>:2033 [#uses=1]
	seteq ubyte %2033, 0		; <bool>:887 [#uses=1]
	br bool %887, label %889, label %888

; <label>:888		; preds = %887, %888
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %611		; <ubyte*>:1289 [#uses=2]
	load ubyte* %1289		; <ubyte>:2034 [#uses=1]
	add ubyte %2034, 1		; <ubyte>:2035 [#uses=1]
	store ubyte %2035, ubyte* %1289
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %613		; <ubyte*>:1290 [#uses=2]
	load ubyte* %1290		; <ubyte>:2036 [#uses=2]
	add ubyte %2036, 255		; <ubyte>:2037 [#uses=1]
	store ubyte %2037, ubyte* %1290
	seteq ubyte %2036, 1		; <bool>:888 [#uses=1]
	br bool %888, label %889, label %888

; <label>:889		; preds = %887, %888
	add uint %583, 52		; <uint>:614 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %614		; <ubyte*>:1291 [#uses=1]
	load ubyte* %1291		; <ubyte>:2038 [#uses=1]
	seteq ubyte %2038, 0		; <bool>:889 [#uses=1]
	br bool %889, label %891, label %890

; <label>:890		; preds = %889, %890
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %614		; <ubyte*>:1292 [#uses=2]
	load ubyte* %1292		; <ubyte>:2039 [#uses=2]
	add ubyte %2039, 255		; <ubyte>:2040 [#uses=1]
	store ubyte %2040, ubyte* %1292
	seteq ubyte %2039, 1		; <bool>:890 [#uses=1]
	br bool %890, label %891, label %890

; <label>:891		; preds = %889, %890
	add uint %583, 4294967231		; <uint>:615 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %615		; <ubyte*>:1293 [#uses=1]
	load ubyte* %1293		; <ubyte>:2041 [#uses=1]
	seteq ubyte %2041, 0		; <bool>:891 [#uses=1]
	br bool %891, label %893, label %892

; <label>:892		; preds = %891, %892
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %615		; <ubyte*>:1294 [#uses=2]
	load ubyte* %1294		; <ubyte>:2042 [#uses=1]
	add ubyte %2042, 255		; <ubyte>:2043 [#uses=1]
	store ubyte %2043, ubyte* %1294
	add uint %583, 4294967232		; <uint>:616 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %616		; <ubyte*>:1295 [#uses=2]
	load ubyte* %1295		; <ubyte>:2044 [#uses=1]
	add ubyte %2044, 1		; <ubyte>:2045 [#uses=1]
	store ubyte %2045, ubyte* %1295
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %614		; <ubyte*>:1296 [#uses=2]
	load ubyte* %1296		; <ubyte>:2046 [#uses=1]
	add ubyte %2046, 1		; <ubyte>:2047 [#uses=1]
	store ubyte %2047, ubyte* %1296
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %615		; <ubyte*>:1297 [#uses=1]
	load ubyte* %1297		; <ubyte>:2048 [#uses=1]
	seteq ubyte %2048, 0		; <bool>:892 [#uses=1]
	br bool %892, label %893, label %892

; <label>:893		; preds = %891, %892
	add uint %583, 4294967232		; <uint>:617 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %617		; <ubyte*>:1298 [#uses=1]
	load ubyte* %1298		; <ubyte>:2049 [#uses=1]
	seteq ubyte %2049, 0		; <bool>:893 [#uses=1]
	br bool %893, label %895, label %894

; <label>:894		; preds = %893, %894
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %615		; <ubyte*>:1299 [#uses=2]
	load ubyte* %1299		; <ubyte>:2050 [#uses=1]
	add ubyte %2050, 1		; <ubyte>:2051 [#uses=1]
	store ubyte %2051, ubyte* %1299
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %617		; <ubyte*>:1300 [#uses=2]
	load ubyte* %1300		; <ubyte>:2052 [#uses=2]
	add ubyte %2052, 255		; <ubyte>:2053 [#uses=1]
	store ubyte %2053, ubyte* %1300
	seteq ubyte %2052, 1		; <bool>:894 [#uses=1]
	br bool %894, label %895, label %894

; <label>:895		; preds = %893, %894
	add uint %583, 58		; <uint>:618 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %618		; <ubyte*>:1301 [#uses=1]
	load ubyte* %1301		; <ubyte>:2054 [#uses=1]
	seteq ubyte %2054, 0		; <bool>:895 [#uses=1]
	br bool %895, label %897, label %896

; <label>:896		; preds = %895, %896
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %618		; <ubyte*>:1302 [#uses=2]
	load ubyte* %1302		; <ubyte>:2055 [#uses=2]
	add ubyte %2055, 255		; <ubyte>:2056 [#uses=1]
	store ubyte %2056, ubyte* %1302
	seteq ubyte %2055, 1		; <bool>:896 [#uses=1]
	br bool %896, label %897, label %896

; <label>:897		; preds = %895, %896
	add uint %583, 4294967237		; <uint>:619 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %619		; <ubyte*>:1303 [#uses=1]
	load ubyte* %1303		; <ubyte>:2057 [#uses=1]
	seteq ubyte %2057, 0		; <bool>:897 [#uses=1]
	br bool %897, label %899, label %898

; <label>:898		; preds = %897, %898
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %619		; <ubyte*>:1304 [#uses=2]
	load ubyte* %1304		; <ubyte>:2058 [#uses=1]
	add ubyte %2058, 255		; <ubyte>:2059 [#uses=1]
	store ubyte %2059, ubyte* %1304
	add uint %583, 4294967238		; <uint>:620 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %620		; <ubyte*>:1305 [#uses=2]
	load ubyte* %1305		; <ubyte>:2060 [#uses=1]
	add ubyte %2060, 1		; <ubyte>:2061 [#uses=1]
	store ubyte %2061, ubyte* %1305
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %618		; <ubyte*>:1306 [#uses=2]
	load ubyte* %1306		; <ubyte>:2062 [#uses=1]
	add ubyte %2062, 1		; <ubyte>:2063 [#uses=1]
	store ubyte %2063, ubyte* %1306
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %619		; <ubyte*>:1307 [#uses=1]
	load ubyte* %1307		; <ubyte>:2064 [#uses=1]
	seteq ubyte %2064, 0		; <bool>:898 [#uses=1]
	br bool %898, label %899, label %898

; <label>:899		; preds = %897, %898
	add uint %583, 4294967238		; <uint>:621 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %621		; <ubyte*>:1308 [#uses=1]
	load ubyte* %1308		; <ubyte>:2065 [#uses=1]
	seteq ubyte %2065, 0		; <bool>:899 [#uses=1]
	br bool %899, label %901, label %900

; <label>:900		; preds = %899, %900
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %619		; <ubyte*>:1309 [#uses=2]
	load ubyte* %1309		; <ubyte>:2066 [#uses=1]
	add ubyte %2066, 1		; <ubyte>:2067 [#uses=1]
	store ubyte %2067, ubyte* %1309
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %621		; <ubyte*>:1310 [#uses=2]
	load ubyte* %1310		; <ubyte>:2068 [#uses=2]
	add ubyte %2068, 255		; <ubyte>:2069 [#uses=1]
	store ubyte %2069, ubyte* %1310
	seteq ubyte %2068, 1		; <bool>:900 [#uses=1]
	br bool %900, label %901, label %900

; <label>:901		; preds = %899, %900
	add uint %583, 64		; <uint>:622 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %622		; <ubyte*>:1311 [#uses=1]
	load ubyte* %1311		; <ubyte>:2070 [#uses=1]
	seteq ubyte %2070, 0		; <bool>:901 [#uses=1]
	br bool %901, label %903, label %902

; <label>:902		; preds = %901, %902
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %622		; <ubyte*>:1312 [#uses=2]
	load ubyte* %1312		; <ubyte>:2071 [#uses=2]
	add ubyte %2071, 255		; <ubyte>:2072 [#uses=1]
	store ubyte %2072, ubyte* %1312
	seteq ubyte %2071, 1		; <bool>:902 [#uses=1]
	br bool %902, label %903, label %902

; <label>:903		; preds = %901, %902
	add uint %583, 4294967243		; <uint>:623 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %623		; <ubyte*>:1313 [#uses=1]
	load ubyte* %1313		; <ubyte>:2073 [#uses=1]
	seteq ubyte %2073, 0		; <bool>:903 [#uses=1]
	br bool %903, label %905, label %904

; <label>:904		; preds = %903, %904
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %623		; <ubyte*>:1314 [#uses=2]
	load ubyte* %1314		; <ubyte>:2074 [#uses=1]
	add ubyte %2074, 255		; <ubyte>:2075 [#uses=1]
	store ubyte %2075, ubyte* %1314
	add uint %583, 4294967244		; <uint>:624 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %624		; <ubyte*>:1315 [#uses=2]
	load ubyte* %1315		; <ubyte>:2076 [#uses=1]
	add ubyte %2076, 1		; <ubyte>:2077 [#uses=1]
	store ubyte %2077, ubyte* %1315
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %622		; <ubyte*>:1316 [#uses=2]
	load ubyte* %1316		; <ubyte>:2078 [#uses=1]
	add ubyte %2078, 1		; <ubyte>:2079 [#uses=1]
	store ubyte %2079, ubyte* %1316
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %623		; <ubyte*>:1317 [#uses=1]
	load ubyte* %1317		; <ubyte>:2080 [#uses=1]
	seteq ubyte %2080, 0		; <bool>:904 [#uses=1]
	br bool %904, label %905, label %904

; <label>:905		; preds = %903, %904
	add uint %583, 4294967244		; <uint>:625 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %625		; <ubyte*>:1318 [#uses=1]
	load ubyte* %1318		; <ubyte>:2081 [#uses=1]
	seteq ubyte %2081, 0		; <bool>:905 [#uses=1]
	br bool %905, label %907, label %906

; <label>:906		; preds = %905, %906
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %623		; <ubyte*>:1319 [#uses=2]
	load ubyte* %1319		; <ubyte>:2082 [#uses=1]
	add ubyte %2082, 1		; <ubyte>:2083 [#uses=1]
	store ubyte %2083, ubyte* %1319
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %625		; <ubyte*>:1320 [#uses=2]
	load ubyte* %1320		; <ubyte>:2084 [#uses=2]
	add ubyte %2084, 255		; <ubyte>:2085 [#uses=1]
	store ubyte %2085, ubyte* %1320
	seteq ubyte %2084, 1		; <bool>:906 [#uses=1]
	br bool %906, label %907, label %906

; <label>:907		; preds = %905, %906
	add uint %583, 70		; <uint>:626 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %626		; <ubyte*>:1321 [#uses=1]
	load ubyte* %1321		; <ubyte>:2086 [#uses=1]
	seteq ubyte %2086, 0		; <bool>:907 [#uses=1]
	br bool %907, label %909, label %908

; <label>:908		; preds = %907, %908
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %626		; <ubyte*>:1322 [#uses=2]
	load ubyte* %1322		; <ubyte>:2087 [#uses=2]
	add ubyte %2087, 255		; <ubyte>:2088 [#uses=1]
	store ubyte %2088, ubyte* %1322
	seteq ubyte %2087, 1		; <bool>:908 [#uses=1]
	br bool %908, label %909, label %908

; <label>:909		; preds = %907, %908
	add uint %583, 4294967249		; <uint>:627 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %627		; <ubyte*>:1323 [#uses=1]
	load ubyte* %1323		; <ubyte>:2089 [#uses=1]
	seteq ubyte %2089, 0		; <bool>:909 [#uses=1]
	br bool %909, label %911, label %910

; <label>:910		; preds = %909, %910
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %627		; <ubyte*>:1324 [#uses=2]
	load ubyte* %1324		; <ubyte>:2090 [#uses=1]
	add ubyte %2090, 255		; <ubyte>:2091 [#uses=1]
	store ubyte %2091, ubyte* %1324
	add uint %583, 4294967250		; <uint>:628 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %628		; <ubyte*>:1325 [#uses=2]
	load ubyte* %1325		; <ubyte>:2092 [#uses=1]
	add ubyte %2092, 1		; <ubyte>:2093 [#uses=1]
	store ubyte %2093, ubyte* %1325
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %626		; <ubyte*>:1326 [#uses=2]
	load ubyte* %1326		; <ubyte>:2094 [#uses=1]
	add ubyte %2094, 1		; <ubyte>:2095 [#uses=1]
	store ubyte %2095, ubyte* %1326
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %627		; <ubyte*>:1327 [#uses=1]
	load ubyte* %1327		; <ubyte>:2096 [#uses=1]
	seteq ubyte %2096, 0		; <bool>:910 [#uses=1]
	br bool %910, label %911, label %910

; <label>:911		; preds = %909, %910
	add uint %583, 4294967250		; <uint>:629 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %629		; <ubyte*>:1328 [#uses=1]
	load ubyte* %1328		; <ubyte>:2097 [#uses=1]
	seteq ubyte %2097, 0		; <bool>:911 [#uses=1]
	br bool %911, label %913, label %912

; <label>:912		; preds = %911, %912
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %627		; <ubyte*>:1329 [#uses=2]
	load ubyte* %1329		; <ubyte>:2098 [#uses=1]
	add ubyte %2098, 1		; <ubyte>:2099 [#uses=1]
	store ubyte %2099, ubyte* %1329
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %629		; <ubyte*>:1330 [#uses=2]
	load ubyte* %1330		; <ubyte>:2100 [#uses=2]
	add ubyte %2100, 255		; <ubyte>:2101 [#uses=1]
	store ubyte %2101, ubyte* %1330
	seteq ubyte %2100, 1		; <bool>:912 [#uses=1]
	br bool %912, label %913, label %912

; <label>:913		; preds = %911, %912
	add uint %583, 76		; <uint>:630 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %630		; <ubyte*>:1331 [#uses=1]
	load ubyte* %1331		; <ubyte>:2102 [#uses=1]
	seteq ubyte %2102, 0		; <bool>:913 [#uses=1]
	br bool %913, label %915, label %914

; <label>:914		; preds = %913, %914
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %630		; <ubyte*>:1332 [#uses=2]
	load ubyte* %1332		; <ubyte>:2103 [#uses=2]
	add ubyte %2103, 255		; <ubyte>:2104 [#uses=1]
	store ubyte %2104, ubyte* %1332
	seteq ubyte %2103, 1		; <bool>:914 [#uses=1]
	br bool %914, label %915, label %914

; <label>:915		; preds = %913, %914
	add uint %583, 4294967255		; <uint>:631 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %631		; <ubyte*>:1333 [#uses=1]
	load ubyte* %1333		; <ubyte>:2105 [#uses=1]
	seteq ubyte %2105, 0		; <bool>:915 [#uses=1]
	br bool %915, label %917, label %916

; <label>:916		; preds = %915, %916
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %631		; <ubyte*>:1334 [#uses=2]
	load ubyte* %1334		; <ubyte>:2106 [#uses=1]
	add ubyte %2106, 255		; <ubyte>:2107 [#uses=1]
	store ubyte %2107, ubyte* %1334
	add uint %583, 4294967256		; <uint>:632 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %632		; <ubyte*>:1335 [#uses=2]
	load ubyte* %1335		; <ubyte>:2108 [#uses=1]
	add ubyte %2108, 1		; <ubyte>:2109 [#uses=1]
	store ubyte %2109, ubyte* %1335
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %630		; <ubyte*>:1336 [#uses=2]
	load ubyte* %1336		; <ubyte>:2110 [#uses=1]
	add ubyte %2110, 1		; <ubyte>:2111 [#uses=1]
	store ubyte %2111, ubyte* %1336
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %631		; <ubyte*>:1337 [#uses=1]
	load ubyte* %1337		; <ubyte>:2112 [#uses=1]
	seteq ubyte %2112, 0		; <bool>:916 [#uses=1]
	br bool %916, label %917, label %916

; <label>:917		; preds = %915, %916
	add uint %583, 4294967256		; <uint>:633 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %633		; <ubyte*>:1338 [#uses=1]
	load ubyte* %1338		; <ubyte>:2113 [#uses=1]
	seteq ubyte %2113, 0		; <bool>:917 [#uses=1]
	br bool %917, label %919, label %918

; <label>:918		; preds = %917, %918
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %631		; <ubyte*>:1339 [#uses=2]
	load ubyte* %1339		; <ubyte>:2114 [#uses=1]
	add ubyte %2114, 1		; <ubyte>:2115 [#uses=1]
	store ubyte %2115, ubyte* %1339
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %633		; <ubyte*>:1340 [#uses=2]
	load ubyte* %1340		; <ubyte>:2116 [#uses=2]
	add ubyte %2116, 255		; <ubyte>:2117 [#uses=1]
	store ubyte %2117, ubyte* %1340
	seteq ubyte %2116, 1		; <bool>:918 [#uses=1]
	br bool %918, label %919, label %918

; <label>:919		; preds = %917, %918
	add uint %583, 82		; <uint>:634 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %634		; <ubyte*>:1341 [#uses=1]
	load ubyte* %1341		; <ubyte>:2118 [#uses=1]
	seteq ubyte %2118, 0		; <bool>:919 [#uses=1]
	br bool %919, label %921, label %920

; <label>:920		; preds = %919, %920
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %634		; <ubyte*>:1342 [#uses=2]
	load ubyte* %1342		; <ubyte>:2119 [#uses=2]
	add ubyte %2119, 255		; <ubyte>:2120 [#uses=1]
	store ubyte %2120, ubyte* %1342
	seteq ubyte %2119, 1		; <bool>:920 [#uses=1]
	br bool %920, label %921, label %920

; <label>:921		; preds = %919, %920
	add uint %583, 4294967261		; <uint>:635 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %635		; <ubyte*>:1343 [#uses=1]
	load ubyte* %1343		; <ubyte>:2121 [#uses=1]
	seteq ubyte %2121, 0		; <bool>:921 [#uses=1]
	br bool %921, label %923, label %922

; <label>:922		; preds = %921, %922
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %635		; <ubyte*>:1344 [#uses=2]
	load ubyte* %1344		; <ubyte>:2122 [#uses=1]
	add ubyte %2122, 255		; <ubyte>:2123 [#uses=1]
	store ubyte %2123, ubyte* %1344
	add uint %583, 4294967262		; <uint>:636 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %636		; <ubyte*>:1345 [#uses=2]
	load ubyte* %1345		; <ubyte>:2124 [#uses=1]
	add ubyte %2124, 1		; <ubyte>:2125 [#uses=1]
	store ubyte %2125, ubyte* %1345
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %634		; <ubyte*>:1346 [#uses=2]
	load ubyte* %1346		; <ubyte>:2126 [#uses=1]
	add ubyte %2126, 1		; <ubyte>:2127 [#uses=1]
	store ubyte %2127, ubyte* %1346
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %635		; <ubyte*>:1347 [#uses=1]
	load ubyte* %1347		; <ubyte>:2128 [#uses=1]
	seteq ubyte %2128, 0		; <bool>:922 [#uses=1]
	br bool %922, label %923, label %922

; <label>:923		; preds = %921, %922
	add uint %583, 4294967262		; <uint>:637 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %637		; <ubyte*>:1348 [#uses=1]
	load ubyte* %1348		; <ubyte>:2129 [#uses=1]
	seteq ubyte %2129, 0		; <bool>:923 [#uses=1]
	br bool %923, label %925, label %924

; <label>:924		; preds = %923, %924
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %635		; <ubyte*>:1349 [#uses=2]
	load ubyte* %1349		; <ubyte>:2130 [#uses=1]
	add ubyte %2130, 1		; <ubyte>:2131 [#uses=1]
	store ubyte %2131, ubyte* %1349
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %637		; <ubyte*>:1350 [#uses=2]
	load ubyte* %1350		; <ubyte>:2132 [#uses=2]
	add ubyte %2132, 255		; <ubyte>:2133 [#uses=1]
	store ubyte %2133, ubyte* %1350
	seteq ubyte %2132, 1		; <bool>:924 [#uses=1]
	br bool %924, label %925, label %924

; <label>:925		; preds = %923, %924
	add uint %583, 88		; <uint>:638 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %638		; <ubyte*>:1351 [#uses=1]
	load ubyte* %1351		; <ubyte>:2134 [#uses=1]
	seteq ubyte %2134, 0		; <bool>:925 [#uses=1]
	br bool %925, label %927, label %926

; <label>:926		; preds = %925, %926
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %638		; <ubyte*>:1352 [#uses=2]
	load ubyte* %1352		; <ubyte>:2135 [#uses=2]
	add ubyte %2135, 255		; <ubyte>:2136 [#uses=1]
	store ubyte %2136, ubyte* %1352
	seteq ubyte %2135, 1		; <bool>:926 [#uses=1]
	br bool %926, label %927, label %926

; <label>:927		; preds = %925, %926
	add uint %583, 4294967267		; <uint>:639 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %639		; <ubyte*>:1353 [#uses=1]
	load ubyte* %1353		; <ubyte>:2137 [#uses=1]
	seteq ubyte %2137, 0		; <bool>:927 [#uses=1]
	br bool %927, label %929, label %928

; <label>:928		; preds = %927, %928
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %639		; <ubyte*>:1354 [#uses=2]
	load ubyte* %1354		; <ubyte>:2138 [#uses=1]
	add ubyte %2138, 255		; <ubyte>:2139 [#uses=1]
	store ubyte %2139, ubyte* %1354
	add uint %583, 4294967268		; <uint>:640 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %640		; <ubyte*>:1355 [#uses=2]
	load ubyte* %1355		; <ubyte>:2140 [#uses=1]
	add ubyte %2140, 1		; <ubyte>:2141 [#uses=1]
	store ubyte %2141, ubyte* %1355
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %638		; <ubyte*>:1356 [#uses=2]
	load ubyte* %1356		; <ubyte>:2142 [#uses=1]
	add ubyte %2142, 1		; <ubyte>:2143 [#uses=1]
	store ubyte %2143, ubyte* %1356
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %639		; <ubyte*>:1357 [#uses=1]
	load ubyte* %1357		; <ubyte>:2144 [#uses=1]
	seteq ubyte %2144, 0		; <bool>:928 [#uses=1]
	br bool %928, label %929, label %928

; <label>:929		; preds = %927, %928
	add uint %583, 4294967268		; <uint>:641 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %641		; <ubyte*>:1358 [#uses=1]
	load ubyte* %1358		; <ubyte>:2145 [#uses=1]
	seteq ubyte %2145, 0		; <bool>:929 [#uses=1]
	br bool %929, label %931, label %930

; <label>:930		; preds = %929, %930
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %639		; <ubyte*>:1359 [#uses=2]
	load ubyte* %1359		; <ubyte>:2146 [#uses=1]
	add ubyte %2146, 1		; <ubyte>:2147 [#uses=1]
	store ubyte %2147, ubyte* %1359
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %641		; <ubyte*>:1360 [#uses=2]
	load ubyte* %1360		; <ubyte>:2148 [#uses=2]
	add ubyte %2148, 255		; <ubyte>:2149 [#uses=1]
	store ubyte %2149, ubyte* %1360
	seteq ubyte %2148, 1		; <bool>:930 [#uses=1]
	br bool %930, label %931, label %930

; <label>:931		; preds = %929, %930
	add uint %583, 94		; <uint>:642 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %642		; <ubyte*>:1361 [#uses=1]
	load ubyte* %1361		; <ubyte>:2150 [#uses=1]
	seteq ubyte %2150, 0		; <bool>:931 [#uses=1]
	br bool %931, label %933, label %932

; <label>:932		; preds = %931, %932
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %642		; <ubyte*>:1362 [#uses=2]
	load ubyte* %1362		; <ubyte>:2151 [#uses=2]
	add ubyte %2151, 255		; <ubyte>:2152 [#uses=1]
	store ubyte %2152, ubyte* %1362
	seteq ubyte %2151, 1		; <bool>:932 [#uses=1]
	br bool %932, label %933, label %932

; <label>:933		; preds = %931, %932
	add uint %583, 4294967273		; <uint>:643 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %643		; <ubyte*>:1363 [#uses=1]
	load ubyte* %1363		; <ubyte>:2153 [#uses=1]
	seteq ubyte %2153, 0		; <bool>:933 [#uses=1]
	br bool %933, label %935, label %934

; <label>:934		; preds = %933, %934
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %643		; <ubyte*>:1364 [#uses=2]
	load ubyte* %1364		; <ubyte>:2154 [#uses=1]
	add ubyte %2154, 255		; <ubyte>:2155 [#uses=1]
	store ubyte %2155, ubyte* %1364
	add uint %583, 4294967274		; <uint>:644 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %644		; <ubyte*>:1365 [#uses=2]
	load ubyte* %1365		; <ubyte>:2156 [#uses=1]
	add ubyte %2156, 1		; <ubyte>:2157 [#uses=1]
	store ubyte %2157, ubyte* %1365
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %642		; <ubyte*>:1366 [#uses=2]
	load ubyte* %1366		; <ubyte>:2158 [#uses=1]
	add ubyte %2158, 1		; <ubyte>:2159 [#uses=1]
	store ubyte %2159, ubyte* %1366
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %643		; <ubyte*>:1367 [#uses=1]
	load ubyte* %1367		; <ubyte>:2160 [#uses=1]
	seteq ubyte %2160, 0		; <bool>:934 [#uses=1]
	br bool %934, label %935, label %934

; <label>:935		; preds = %933, %934
	add uint %583, 4294967274		; <uint>:645 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %645		; <ubyte*>:1368 [#uses=1]
	load ubyte* %1368		; <ubyte>:2161 [#uses=1]
	seteq ubyte %2161, 0		; <bool>:935 [#uses=1]
	br bool %935, label %937, label %936

; <label>:936		; preds = %935, %936
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %643		; <ubyte*>:1369 [#uses=2]
	load ubyte* %1369		; <ubyte>:2162 [#uses=1]
	add ubyte %2162, 1		; <ubyte>:2163 [#uses=1]
	store ubyte %2163, ubyte* %1369
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %645		; <ubyte*>:1370 [#uses=2]
	load ubyte* %1370		; <ubyte>:2164 [#uses=2]
	add ubyte %2164, 255		; <ubyte>:2165 [#uses=1]
	store ubyte %2165, ubyte* %1370
	seteq ubyte %2164, 1		; <bool>:936 [#uses=1]
	br bool %936, label %937, label %936

; <label>:937		; preds = %935, %936
	add uint %583, 100		; <uint>:646 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %646		; <ubyte*>:1371 [#uses=1]
	load ubyte* %1371		; <ubyte>:2166 [#uses=1]
	seteq ubyte %2166, 0		; <bool>:937 [#uses=1]
	br bool %937, label %939, label %938

; <label>:938		; preds = %937, %938
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %646		; <ubyte*>:1372 [#uses=2]
	load ubyte* %1372		; <ubyte>:2167 [#uses=2]
	add ubyte %2167, 255		; <ubyte>:2168 [#uses=1]
	store ubyte %2168, ubyte* %1372
	seteq ubyte %2167, 1		; <bool>:938 [#uses=1]
	br bool %938, label %939, label %938

; <label>:939		; preds = %937, %938
	add uint %583, 4294967279		; <uint>:647 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %647		; <ubyte*>:1373 [#uses=1]
	load ubyte* %1373		; <ubyte>:2169 [#uses=1]
	seteq ubyte %2169, 0		; <bool>:939 [#uses=1]
	br bool %939, label %941, label %940

; <label>:940		; preds = %939, %940
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %647		; <ubyte*>:1374 [#uses=2]
	load ubyte* %1374		; <ubyte>:2170 [#uses=1]
	add ubyte %2170, 255		; <ubyte>:2171 [#uses=1]
	store ubyte %2171, ubyte* %1374
	add uint %583, 4294967280		; <uint>:648 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %648		; <ubyte*>:1375 [#uses=2]
	load ubyte* %1375		; <ubyte>:2172 [#uses=1]
	add ubyte %2172, 1		; <ubyte>:2173 [#uses=1]
	store ubyte %2173, ubyte* %1375
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %646		; <ubyte*>:1376 [#uses=2]
	load ubyte* %1376		; <ubyte>:2174 [#uses=1]
	add ubyte %2174, 1		; <ubyte>:2175 [#uses=1]
	store ubyte %2175, ubyte* %1376
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %647		; <ubyte*>:1377 [#uses=1]
	load ubyte* %1377		; <ubyte>:2176 [#uses=1]
	seteq ubyte %2176, 0		; <bool>:940 [#uses=1]
	br bool %940, label %941, label %940

; <label>:941		; preds = %939, %940
	add uint %583, 4294967280		; <uint>:649 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %649		; <ubyte*>:1378 [#uses=1]
	load ubyte* %1378		; <ubyte>:2177 [#uses=1]
	seteq ubyte %2177, 0		; <bool>:941 [#uses=1]
	br bool %941, label %943, label %942

; <label>:942		; preds = %941, %942
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %647		; <ubyte*>:1379 [#uses=2]
	load ubyte* %1379		; <ubyte>:2178 [#uses=1]
	add ubyte %2178, 1		; <ubyte>:2179 [#uses=1]
	store ubyte %2179, ubyte* %1379
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %649		; <ubyte*>:1380 [#uses=2]
	load ubyte* %1380		; <ubyte>:2180 [#uses=2]
	add ubyte %2180, 255		; <ubyte>:2181 [#uses=1]
	store ubyte %2181, ubyte* %1380
	seteq ubyte %2180, 1		; <bool>:942 [#uses=1]
	br bool %942, label %943, label %942

; <label>:943		; preds = %941, %942
	add uint %583, 106		; <uint>:650 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %650		; <ubyte*>:1381 [#uses=1]
	load ubyte* %1381		; <ubyte>:2182 [#uses=1]
	seteq ubyte %2182, 0		; <bool>:943 [#uses=1]
	br bool %943, label %945, label %944

; <label>:944		; preds = %943, %944
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %650		; <ubyte*>:1382 [#uses=2]
	load ubyte* %1382		; <ubyte>:2183 [#uses=2]
	add ubyte %2183, 255		; <ubyte>:2184 [#uses=1]
	store ubyte %2184, ubyte* %1382
	seteq ubyte %2183, 1		; <bool>:944 [#uses=1]
	br bool %944, label %945, label %944

; <label>:945		; preds = %943, %944
	add uint %583, 4294967285		; <uint>:651 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %651		; <ubyte*>:1383 [#uses=1]
	load ubyte* %1383		; <ubyte>:2185 [#uses=1]
	seteq ubyte %2185, 0		; <bool>:945 [#uses=1]
	br bool %945, label %947, label %946

; <label>:946		; preds = %945, %946
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %651		; <ubyte*>:1384 [#uses=2]
	load ubyte* %1384		; <ubyte>:2186 [#uses=1]
	add ubyte %2186, 255		; <ubyte>:2187 [#uses=1]
	store ubyte %2187, ubyte* %1384
	add uint %583, 4294967286		; <uint>:652 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %652		; <ubyte*>:1385 [#uses=2]
	load ubyte* %1385		; <ubyte>:2188 [#uses=1]
	add ubyte %2188, 1		; <ubyte>:2189 [#uses=1]
	store ubyte %2189, ubyte* %1385
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %650		; <ubyte*>:1386 [#uses=2]
	load ubyte* %1386		; <ubyte>:2190 [#uses=1]
	add ubyte %2190, 1		; <ubyte>:2191 [#uses=1]
	store ubyte %2191, ubyte* %1386
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %651		; <ubyte*>:1387 [#uses=1]
	load ubyte* %1387		; <ubyte>:2192 [#uses=1]
	seteq ubyte %2192, 0		; <bool>:946 [#uses=1]
	br bool %946, label %947, label %946

; <label>:947		; preds = %945, %946
	add uint %583, 4294967286		; <uint>:653 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %653		; <ubyte*>:1388 [#uses=1]
	load ubyte* %1388		; <ubyte>:2193 [#uses=1]
	seteq ubyte %2193, 0		; <bool>:947 [#uses=1]
	br bool %947, label %949, label %948

; <label>:948		; preds = %947, %948
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %651		; <ubyte*>:1389 [#uses=2]
	load ubyte* %1389		; <ubyte>:2194 [#uses=1]
	add ubyte %2194, 1		; <ubyte>:2195 [#uses=1]
	store ubyte %2195, ubyte* %1389
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %653		; <ubyte*>:1390 [#uses=2]
	load ubyte* %1390		; <ubyte>:2196 [#uses=2]
	add ubyte %2196, 255		; <ubyte>:2197 [#uses=1]
	store ubyte %2197, ubyte* %1390
	seteq ubyte %2196, 1		; <bool>:948 [#uses=1]
	br bool %948, label %949, label %948

; <label>:949		; preds = %947, %948
	add uint %583, 110		; <uint>:654 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %654		; <ubyte*>:1391 [#uses=2]
	load ubyte* %1391		; <ubyte>:2198 [#uses=1]
	add ubyte %2198, 10		; <ubyte>:2199 [#uses=1]
	store ubyte %2199, ubyte* %1391
	add uint %583, 113		; <uint>:655 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %655		; <ubyte*>:1392 [#uses=2]
	load ubyte* %1392		; <ubyte>:2200 [#uses=1]
	add ubyte %2200, 1		; <ubyte>:2201 [#uses=1]
	store ubyte %2201, ubyte* %1392
	add uint %583, 116		; <uint>:656 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %656		; <ubyte*>:1393 [#uses=2]
	load ubyte* %1393		; <ubyte>:2202 [#uses=1]
	add ubyte %2202, 1		; <ubyte>:2203 [#uses=1]
	store ubyte %2203, ubyte* %1393
	add uint %583, 119		; <uint>:657 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %657		; <ubyte*>:1394 [#uses=2]
	load ubyte* %1394		; <ubyte>:2204 [#uses=1]
	add ubyte %2204, 1		; <ubyte>:2205 [#uses=1]
	store ubyte %2205, ubyte* %1394
	add uint %583, 124		; <uint>:658 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %658		; <ubyte*>:1395 [#uses=1]
	load ubyte* %1395		; <ubyte>:2206 [#uses=1]
	seteq ubyte %2206, 0		; <bool>:949 [#uses=1]
	br bool %949, label %847, label %846

; <label>:950		; preds = %587, %1005
	phi uint [ %408, %587 ], [ %683, %1005 ]		; <uint>:659 [#uses=23]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %659		; <ubyte*>:1396 [#uses=2]
	load ubyte* %1396		; <ubyte>:2207 [#uses=1]
	add ubyte %2207, 255		; <ubyte>:2208 [#uses=1]
	store ubyte %2208, ubyte* %1396
	add uint %659, 10		; <uint>:660 [#uses=18]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %660		; <ubyte*>:1397 [#uses=1]
	load ubyte* %1397		; <ubyte>:2209 [#uses=1]
	seteq ubyte %2209, 0		; <bool>:950 [#uses=1]
	br bool %950, label %953, label %952

; <label>:951		; preds = %587, %1005
	phi uint [ %408, %587 ], [ %683, %1005 ]		; <uint>:661 [#uses=1]
	add uint %661, 4294967295		; <uint>:662 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %662		; <ubyte*>:1398 [#uses=1]
	load ubyte* %1398		; <ubyte>:2210 [#uses=1]
	seteq ubyte %2210, 0		; <bool>:951 [#uses=1]
	br bool %951, label %585, label %584

; <label>:952		; preds = %950, %952
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %660		; <ubyte*>:1399 [#uses=2]
	load ubyte* %1399		; <ubyte>:2211 [#uses=2]
	add ubyte %2211, 255		; <ubyte>:2212 [#uses=1]
	store ubyte %2212, ubyte* %1399
	seteq ubyte %2211, 1		; <bool>:952 [#uses=1]
	br bool %952, label %953, label %952

; <label>:953		; preds = %950, %952
	add uint %659, 4294967189		; <uint>:663 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %663		; <ubyte*>:1400 [#uses=1]
	load ubyte* %1400		; <ubyte>:2213 [#uses=1]
	seteq ubyte %2213, 0		; <bool>:953 [#uses=1]
	br bool %953, label %955, label %954

; <label>:954		; preds = %953, %954
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %663		; <ubyte*>:1401 [#uses=2]
	load ubyte* %1401		; <ubyte>:2214 [#uses=1]
	add ubyte %2214, 255		; <ubyte>:2215 [#uses=1]
	store ubyte %2215, ubyte* %1401
	add uint %659, 4294967190		; <uint>:664 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %664		; <ubyte*>:1402 [#uses=2]
	load ubyte* %1402		; <ubyte>:2216 [#uses=1]
	add ubyte %2216, 1		; <ubyte>:2217 [#uses=1]
	store ubyte %2217, ubyte* %1402
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %660		; <ubyte*>:1403 [#uses=2]
	load ubyte* %1403		; <ubyte>:2218 [#uses=1]
	add ubyte %2218, 1		; <ubyte>:2219 [#uses=1]
	store ubyte %2219, ubyte* %1403
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %663		; <ubyte*>:1404 [#uses=1]
	load ubyte* %1404		; <ubyte>:2220 [#uses=1]
	seteq ubyte %2220, 0		; <bool>:954 [#uses=1]
	br bool %954, label %955, label %954

; <label>:955		; preds = %953, %954
	add uint %659, 4294967190		; <uint>:665 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %665		; <ubyte*>:1405 [#uses=1]
	load ubyte* %1405		; <ubyte>:2221 [#uses=1]
	seteq ubyte %2221, 0		; <bool>:955 [#uses=1]
	br bool %955, label %957, label %956

; <label>:956		; preds = %955, %956
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %663		; <ubyte*>:1406 [#uses=2]
	load ubyte* %1406		; <ubyte>:2222 [#uses=1]
	add ubyte %2222, 1		; <ubyte>:2223 [#uses=1]
	store ubyte %2223, ubyte* %1406
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %665		; <ubyte*>:1407 [#uses=2]
	load ubyte* %1407		; <ubyte>:2224 [#uses=2]
	add ubyte %2224, 255		; <ubyte>:2225 [#uses=1]
	store ubyte %2225, ubyte* %1407
	seteq ubyte %2224, 1		; <bool>:956 [#uses=1]
	br bool %956, label %957, label %956

; <label>:957		; preds = %955, %956
	add uint %659, 12		; <uint>:666 [#uses=15]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %666		; <ubyte*>:1408 [#uses=1]
	load ubyte* %1408		; <ubyte>:2226 [#uses=1]
	seteq ubyte %2226, 0		; <bool>:957 [#uses=1]
	br bool %957, label %959, label %958

; <label>:958		; preds = %957, %958
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %666		; <ubyte*>:1409 [#uses=2]
	load ubyte* %1409		; <ubyte>:2227 [#uses=2]
	add ubyte %2227, 255		; <ubyte>:2228 [#uses=1]
	store ubyte %2228, ubyte* %1409
	seteq ubyte %2227, 1		; <bool>:958 [#uses=1]
	br bool %958, label %959, label %958

; <label>:959		; preds = %957, %958
	add uint %659, 6		; <uint>:667 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %667		; <ubyte*>:1410 [#uses=1]
	load ubyte* %1410		; <ubyte>:2229 [#uses=1]
	seteq ubyte %2229, 0		; <bool>:959 [#uses=1]
	br bool %959, label %961, label %960

; <label>:960		; preds = %959, %960
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %667		; <ubyte*>:1411 [#uses=2]
	load ubyte* %1411		; <ubyte>:2230 [#uses=1]
	add ubyte %2230, 255		; <ubyte>:2231 [#uses=1]
	store ubyte %2231, ubyte* %1411
	add uint %659, 7		; <uint>:668 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %668		; <ubyte*>:1412 [#uses=2]
	load ubyte* %1412		; <ubyte>:2232 [#uses=1]
	add ubyte %2232, 1		; <ubyte>:2233 [#uses=1]
	store ubyte %2233, ubyte* %1412
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %666		; <ubyte*>:1413 [#uses=2]
	load ubyte* %1413		; <ubyte>:2234 [#uses=1]
	add ubyte %2234, 1		; <ubyte>:2235 [#uses=1]
	store ubyte %2235, ubyte* %1413
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %667		; <ubyte*>:1414 [#uses=1]
	load ubyte* %1414		; <ubyte>:2236 [#uses=1]
	seteq ubyte %2236, 0		; <bool>:960 [#uses=1]
	br bool %960, label %961, label %960

; <label>:961		; preds = %959, %960
	add uint %659, 7		; <uint>:669 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %669		; <ubyte*>:1415 [#uses=1]
	load ubyte* %1415		; <ubyte>:2237 [#uses=1]
	seteq ubyte %2237, 0		; <bool>:961 [#uses=1]
	br bool %961, label %963, label %962

; <label>:962		; preds = %961, %962
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %667		; <ubyte*>:1416 [#uses=2]
	load ubyte* %1416		; <ubyte>:2238 [#uses=1]
	add ubyte %2238, 1		; <ubyte>:2239 [#uses=1]
	store ubyte %2239, ubyte* %1416
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %669		; <ubyte*>:1417 [#uses=2]
	load ubyte* %1417		; <ubyte>:2240 [#uses=2]
	add ubyte %2240, 255		; <ubyte>:2241 [#uses=1]
	store ubyte %2241, ubyte* %1417
	seteq ubyte %2240, 1		; <bool>:962 [#uses=1]
	br bool %962, label %963, label %962

; <label>:963		; preds = %961, %962
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %666		; <ubyte*>:1418 [#uses=1]
	load ubyte* %1418		; <ubyte>:2242 [#uses=1]
	seteq ubyte %2242, 0		; <bool>:963 [#uses=1]
	br bool %963, label %965, label %964

; <label>:964		; preds = %963, %964
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %660		; <ubyte*>:1419 [#uses=2]
	load ubyte* %1419		; <ubyte>:2243 [#uses=1]
	add ubyte %2243, 255		; <ubyte>:2244 [#uses=1]
	store ubyte %2244, ubyte* %1419
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %666		; <ubyte*>:1420 [#uses=2]
	load ubyte* %1420		; <ubyte>:2245 [#uses=2]
	add ubyte %2245, 255		; <ubyte>:2246 [#uses=1]
	store ubyte %2246, ubyte* %1420
	seteq ubyte %2245, 1		; <bool>:964 [#uses=1]
	br bool %964, label %965, label %964

; <label>:965		; preds = %963, %964
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %666		; <ubyte*>:1421 [#uses=2]
	load ubyte* %1421		; <ubyte>:2247 [#uses=1]
	add ubyte %2247, 1		; <ubyte>:2248 [#uses=1]
	store ubyte %2248, ubyte* %1421
	add uint %659, 14		; <uint>:670 [#uses=9]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %670		; <ubyte*>:1422 [#uses=2]
	load ubyte* %1422		; <ubyte>:2249 [#uses=2]
	add ubyte %2249, 1		; <ubyte>:2250 [#uses=1]
	store ubyte %2250, ubyte* %1422
	seteq ubyte %2249, 255		; <bool>:965 [#uses=1]
	br bool %965, label %967, label %966

; <label>:966		; preds = %965, %989
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %670		; <ubyte*>:1423 [#uses=2]
	load ubyte* %1423		; <ubyte>:2251 [#uses=1]
	add ubyte %2251, 1		; <ubyte>:2252 [#uses=1]
	store ubyte %2252, ubyte* %1423
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %660		; <ubyte*>:1424 [#uses=1]
	load ubyte* %1424		; <ubyte>:2253 [#uses=1]
	seteq ubyte %2253, 0		; <bool>:966 [#uses=1]
	br bool %966, label %969, label %968

; <label>:967		; preds = %965, %989
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %660		; <ubyte*>:1425 [#uses=1]
	load ubyte* %1425		; <ubyte>:2254 [#uses=1]
	seteq ubyte %2254, 0		; <bool>:967 [#uses=1]
	br bool %967, label %991, label %990

; <label>:968		; preds = %966, %968
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %660		; <ubyte*>:1426 [#uses=2]
	load ubyte* %1426		; <ubyte>:2255 [#uses=1]
	add ubyte %2255, 255		; <ubyte>:2256 [#uses=1]
	store ubyte %2256, ubyte* %1426
	add uint %659, 11		; <uint>:671 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %671		; <ubyte*>:1427 [#uses=2]
	load ubyte* %1427		; <ubyte>:2257 [#uses=1]
	add ubyte %2257, 1		; <ubyte>:2258 [#uses=1]
	store ubyte %2258, ubyte* %1427
	add uint %659, 15		; <uint>:672 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %672		; <ubyte*>:1428 [#uses=2]
	load ubyte* %1428		; <ubyte>:2259 [#uses=1]
	add ubyte %2259, 1		; <ubyte>:2260 [#uses=1]
	store ubyte %2260, ubyte* %1428
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %660		; <ubyte*>:1429 [#uses=1]
	load ubyte* %1429		; <ubyte>:2261 [#uses=1]
	seteq ubyte %2261, 0		; <bool>:968 [#uses=1]
	br bool %968, label %969, label %968

; <label>:969		; preds = %966, %968
	add uint %659, 11		; <uint>:673 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %673		; <ubyte*>:1430 [#uses=1]
	load ubyte* %1430		; <ubyte>:2262 [#uses=1]
	seteq ubyte %2262, 0		; <bool>:969 [#uses=1]
	br bool %969, label %971, label %970

; <label>:970		; preds = %969, %970
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %660		; <ubyte*>:1431 [#uses=2]
	load ubyte* %1431		; <ubyte>:2263 [#uses=1]
	add ubyte %2263, 1		; <ubyte>:2264 [#uses=1]
	store ubyte %2264, ubyte* %1431
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %673		; <ubyte*>:1432 [#uses=2]
	load ubyte* %1432		; <ubyte>:2265 [#uses=2]
	add ubyte %2265, 255		; <ubyte>:2266 [#uses=1]
	store ubyte %2266, ubyte* %1432
	seteq ubyte %2265, 1		; <bool>:970 [#uses=1]
	br bool %970, label %971, label %970

; <label>:971		; preds = %969, %970
	add uint %659, 15		; <uint>:674 [#uses=13]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %674		; <ubyte*>:1433 [#uses=1]
	load ubyte* %1433		; <ubyte>:2267 [#uses=1]
	seteq ubyte %2267, 0		; <bool>:971 [#uses=1]
	br bool %971, label %973, label %972

; <label>:972		; preds = %971, %975
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %674		; <ubyte*>:1434 [#uses=1]
	load ubyte* %1434		; <ubyte>:2268 [#uses=1]
	seteq ubyte %2268, 0		; <bool>:972 [#uses=1]
	br bool %972, label %975, label %974

; <label>:973		; preds = %971, %975
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %666		; <ubyte*>:1435 [#uses=1]
	load ubyte* %1435		; <ubyte>:2269 [#uses=1]
	seteq ubyte %2269, 0		; <bool>:973 [#uses=1]
	br bool %973, label %977, label %976

; <label>:974		; preds = %972, %974
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %674		; <ubyte*>:1436 [#uses=2]
	load ubyte* %1436		; <ubyte>:2270 [#uses=2]
	add ubyte %2270, 255		; <ubyte>:2271 [#uses=1]
	store ubyte %2271, ubyte* %1436
	seteq ubyte %2270, 1		; <bool>:974 [#uses=1]
	br bool %974, label %975, label %974

; <label>:975		; preds = %972, %974
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %670		; <ubyte*>:1437 [#uses=2]
	load ubyte* %1437		; <ubyte>:2272 [#uses=1]
	add ubyte %2272, 255		; <ubyte>:2273 [#uses=1]
	store ubyte %2273, ubyte* %1437
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %674		; <ubyte*>:1438 [#uses=1]
	load ubyte* %1438		; <ubyte>:2274 [#uses=1]
	seteq ubyte %2274, 0		; <bool>:975 [#uses=1]
	br bool %975, label %973, label %972

; <label>:976		; preds = %973, %976
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %666		; <ubyte*>:1439 [#uses=2]
	load ubyte* %1439		; <ubyte>:2275 [#uses=1]
	add ubyte %2275, 255		; <ubyte>:2276 [#uses=1]
	store ubyte %2276, ubyte* %1439
	add uint %659, 13		; <uint>:675 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %675		; <ubyte*>:1440 [#uses=2]
	load ubyte* %1440		; <ubyte>:2277 [#uses=1]
	add ubyte %2277, 1		; <ubyte>:2278 [#uses=1]
	store ubyte %2278, ubyte* %1440
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %674		; <ubyte*>:1441 [#uses=2]
	load ubyte* %1441		; <ubyte>:2279 [#uses=1]
	add ubyte %2279, 1		; <ubyte>:2280 [#uses=1]
	store ubyte %2280, ubyte* %1441
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %666		; <ubyte*>:1442 [#uses=1]
	load ubyte* %1442		; <ubyte>:2281 [#uses=1]
	seteq ubyte %2281, 0		; <bool>:976 [#uses=1]
	br bool %976, label %977, label %976

; <label>:977		; preds = %973, %976
	add uint %659, 13		; <uint>:676 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %676		; <ubyte*>:1443 [#uses=1]
	load ubyte* %1443		; <ubyte>:2282 [#uses=1]
	seteq ubyte %2282, 0		; <bool>:977 [#uses=1]
	br bool %977, label %979, label %978

; <label>:978		; preds = %977, %978
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %666		; <ubyte*>:1444 [#uses=2]
	load ubyte* %1444		; <ubyte>:2283 [#uses=1]
	add ubyte %2283, 1		; <ubyte>:2284 [#uses=1]
	store ubyte %2284, ubyte* %1444
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %676		; <ubyte*>:1445 [#uses=2]
	load ubyte* %1445		; <ubyte>:2285 [#uses=2]
	add ubyte %2285, 255		; <ubyte>:2286 [#uses=1]
	store ubyte %2286, ubyte* %1445
	seteq ubyte %2285, 1		; <bool>:978 [#uses=1]
	br bool %978, label %979, label %978

; <label>:979		; preds = %977, %978
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %674		; <ubyte*>:1446 [#uses=1]
	load ubyte* %1446		; <ubyte>:2287 [#uses=1]
	seteq ubyte %2287, 0		; <bool>:979 [#uses=1]
	br bool %979, label %981, label %980

; <label>:980		; preds = %979, %983
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %674		; <ubyte*>:1447 [#uses=1]
	load ubyte* %1447		; <ubyte>:2288 [#uses=1]
	seteq ubyte %2288, 0		; <bool>:980 [#uses=1]
	br bool %980, label %983, label %982

; <label>:981		; preds = %979, %983
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %674		; <ubyte*>:1448 [#uses=2]
	load ubyte* %1448		; <ubyte>:2289 [#uses=1]
	add ubyte %2289, 1		; <ubyte>:2290 [#uses=1]
	store ubyte %2290, ubyte* %1448
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %670		; <ubyte*>:1449 [#uses=1]
	load ubyte* %1449		; <ubyte>:2291 [#uses=1]
	seteq ubyte %2291, 0		; <bool>:981 [#uses=1]
	br bool %981, label %985, label %984

; <label>:982		; preds = %980, %982
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %674		; <ubyte*>:1450 [#uses=2]
	load ubyte* %1450		; <ubyte>:2292 [#uses=2]
	add ubyte %2292, 255		; <ubyte>:2293 [#uses=1]
	store ubyte %2293, ubyte* %1450
	seteq ubyte %2292, 1		; <bool>:982 [#uses=1]
	br bool %982, label %983, label %982

; <label>:983		; preds = %980, %982
	add uint %659, 14		; <uint>:677 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %677		; <ubyte*>:1451 [#uses=2]
	load ubyte* %1451		; <ubyte>:2294 [#uses=1]
	add ubyte %2294, 255		; <ubyte>:2295 [#uses=1]
	store ubyte %2295, ubyte* %1451
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %674		; <ubyte*>:1452 [#uses=1]
	load ubyte* %1452		; <ubyte>:2296 [#uses=1]
	seteq ubyte %2296, 0		; <bool>:983 [#uses=1]
	br bool %983, label %981, label %980

; <label>:984		; preds = %981, %987
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %670		; <ubyte*>:1453 [#uses=1]
	load ubyte* %1453		; <ubyte>:2297 [#uses=1]
	seteq ubyte %2297, 0		; <bool>:984 [#uses=1]
	br bool %984, label %987, label %986

; <label>:985		; preds = %981, %987
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %674		; <ubyte*>:1454 [#uses=1]
	load ubyte* %1454		; <ubyte>:2298 [#uses=1]
	seteq ubyte %2298, 0		; <bool>:985 [#uses=1]
	br bool %985, label %989, label %988

; <label>:986		; preds = %984, %986
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %670		; <ubyte*>:1455 [#uses=2]
	load ubyte* %1455		; <ubyte>:2299 [#uses=2]
	add ubyte %2299, 255		; <ubyte>:2300 [#uses=1]
	store ubyte %2300, ubyte* %1455
	seteq ubyte %2299, 1		; <bool>:986 [#uses=1]
	br bool %986, label %987, label %986

; <label>:987		; preds = %984, %986
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %674		; <ubyte*>:1456 [#uses=2]
	load ubyte* %1456		; <ubyte>:2301 [#uses=1]
	add ubyte %2301, 255		; <ubyte>:2302 [#uses=1]
	store ubyte %2302, ubyte* %1456
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %670		; <ubyte*>:1457 [#uses=1]
	load ubyte* %1457		; <ubyte>:2303 [#uses=1]
	seteq ubyte %2303, 0		; <bool>:987 [#uses=1]
	br bool %987, label %985, label %984

; <label>:988		; preds = %985, %988
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %660		; <ubyte*>:1458 [#uses=2]
	load ubyte* %1458		; <ubyte>:2304 [#uses=1]
	add ubyte %2304, 255		; <ubyte>:2305 [#uses=1]
	store ubyte %2305, ubyte* %1458
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %666		; <ubyte*>:1459 [#uses=2]
	load ubyte* %1459		; <ubyte>:2306 [#uses=1]
	add ubyte %2306, 255		; <ubyte>:2307 [#uses=1]
	store ubyte %2307, ubyte* %1459
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %670		; <ubyte*>:1460 [#uses=2]
	load ubyte* %1460		; <ubyte>:2308 [#uses=1]
	add ubyte %2308, 1		; <ubyte>:2309 [#uses=1]
	store ubyte %2309, ubyte* %1460
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %674		; <ubyte*>:1461 [#uses=2]
	load ubyte* %1461		; <ubyte>:2310 [#uses=2]
	add ubyte %2310, 255		; <ubyte>:2311 [#uses=1]
	store ubyte %2311, ubyte* %1461
	seteq ubyte %2310, 1		; <bool>:988 [#uses=1]
	br bool %988, label %989, label %988

; <label>:989		; preds = %985, %988
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %670		; <ubyte*>:1462 [#uses=1]
	load ubyte* %1462		; <ubyte>:2312 [#uses=1]
	seteq ubyte %2312, 0		; <bool>:989 [#uses=1]
	br bool %989, label %967, label %966

; <label>:990		; preds = %967, %993
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %660		; <ubyte*>:1463 [#uses=1]
	load ubyte* %1463		; <ubyte>:2313 [#uses=1]
	seteq ubyte %2313, 0		; <bool>:990 [#uses=1]
	br bool %990, label %993, label %992

; <label>:991		; preds = %967, %993
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %666		; <ubyte*>:1464 [#uses=1]
	load ubyte* %1464		; <ubyte>:2314 [#uses=1]
	seteq ubyte %2314, 0		; <bool>:991 [#uses=1]
	br bool %991, label %995, label %994

; <label>:992		; preds = %990, %992
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %660		; <ubyte*>:1465 [#uses=2]
	load ubyte* %1465		; <ubyte>:2315 [#uses=2]
	add ubyte %2315, 255		; <ubyte>:2316 [#uses=1]
	store ubyte %2316, ubyte* %1465
	seteq ubyte %2315, 1		; <bool>:992 [#uses=1]
	br bool %992, label %993, label %992

; <label>:993		; preds = %990, %992
	add uint %659, 11		; <uint>:678 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %678		; <ubyte*>:1466 [#uses=2]
	load ubyte* %1466		; <ubyte>:2317 [#uses=1]
	add ubyte %2317, 1		; <ubyte>:2318 [#uses=1]
	store ubyte %2318, ubyte* %1466
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %660		; <ubyte*>:1467 [#uses=1]
	load ubyte* %1467		; <ubyte>:2319 [#uses=1]
	seteq ubyte %2319, 0		; <bool>:993 [#uses=1]
	br bool %993, label %991, label %990

; <label>:994		; preds = %991, %997
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %666		; <ubyte*>:1468 [#uses=1]
	load ubyte* %1468		; <ubyte>:2320 [#uses=1]
	seteq ubyte %2320, 0		; <bool>:994 [#uses=1]
	br bool %994, label %997, label %996

; <label>:995		; preds = %991, %997
	add uint %659, 13		; <uint>:679 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %679		; <ubyte*>:1469 [#uses=1]
	load ubyte* %1469		; <ubyte>:2321 [#uses=1]
	seteq ubyte %2321, 0		; <bool>:995 [#uses=1]
	br bool %995, label %999, label %998

; <label>:996		; preds = %994, %996
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %666		; <ubyte*>:1470 [#uses=2]
	load ubyte* %1470		; <ubyte>:2322 [#uses=2]
	add ubyte %2322, 255		; <ubyte>:2323 [#uses=1]
	store ubyte %2323, ubyte* %1470
	seteq ubyte %2322, 1		; <bool>:996 [#uses=1]
	br bool %996, label %997, label %996

; <label>:997		; preds = %994, %996
	add uint %659, 13		; <uint>:680 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %680		; <ubyte*>:1471 [#uses=2]
	load ubyte* %1471		; <ubyte>:2324 [#uses=1]
	add ubyte %2324, 1		; <ubyte>:2325 [#uses=1]
	store ubyte %2325, ubyte* %1471
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %666		; <ubyte*>:1472 [#uses=1]
	load ubyte* %1472		; <ubyte>:2326 [#uses=1]
	seteq ubyte %2326, 0		; <bool>:997 [#uses=1]
	br bool %997, label %995, label %994

; <label>:998		; preds = %995, %998
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %679		; <ubyte*>:1473 [#uses=2]
	load ubyte* %1473		; <ubyte>:2327 [#uses=2]
	add ubyte %2327, 255		; <ubyte>:2328 [#uses=1]
	store ubyte %2328, ubyte* %1473
	seteq ubyte %2327, 1		; <bool>:998 [#uses=1]
	br bool %998, label %999, label %998

; <label>:999		; preds = %995, %998
	add uint %659, 11		; <uint>:681 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %681		; <ubyte*>:1474 [#uses=1]
	load ubyte* %1474		; <ubyte>:2329 [#uses=1]
	seteq ubyte %2329, 0		; <bool>:999 [#uses=1]
	br bool %999, label %1001, label %1000

; <label>:1000		; preds = %999, %1003
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %681		; <ubyte*>:1475 [#uses=1]
	load ubyte* %1475		; <ubyte>:2330 [#uses=1]
	seteq ubyte %2330, 0		; <bool>:1000 [#uses=1]
	br bool %1000, label %1003, label %1002

; <label>:1001		; preds = %999, %1003
	add uint %659, 4294967295		; <uint>:682 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %682		; <ubyte*>:1476 [#uses=2]
	load ubyte* %1476		; <ubyte>:2331 [#uses=1]
	add ubyte %2331, 11		; <ubyte>:2332 [#uses=1]
	store ubyte %2332, ubyte* %1476
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %660		; <ubyte*>:1477 [#uses=1]
	load ubyte* %1477		; <ubyte>:2333 [#uses=1]
	seteq ubyte %2333, 0		; <bool>:1001 [#uses=1]
	br bool %1001, label %1005, label %1004

; <label>:1002		; preds = %1000, %1002
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %681		; <ubyte*>:1478 [#uses=2]
	load ubyte* %1478		; <ubyte>:2334 [#uses=2]
	add ubyte %2334, 255		; <ubyte>:2335 [#uses=1]
	store ubyte %2335, ubyte* %1478
	seteq ubyte %2334, 1		; <bool>:1002 [#uses=1]
	br bool %1002, label %1003, label %1002

; <label>:1003		; preds = %1000, %1002
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %660		; <ubyte*>:1479 [#uses=2]
	load ubyte* %1479		; <ubyte>:2336 [#uses=1]
	add ubyte %2336, 1		; <ubyte>:2337 [#uses=1]
	store ubyte %2337, ubyte* %1479
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %681		; <ubyte*>:1480 [#uses=1]
	load ubyte* %1480		; <ubyte>:2338 [#uses=1]
	seteq ubyte %2338, 0		; <bool>:1003 [#uses=1]
	br bool %1003, label %1001, label %1000

; <label>:1004		; preds = %1001, %1007
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %660		; <ubyte*>:1481 [#uses=1]
	load ubyte* %1481		; <ubyte>:2339 [#uses=1]
	seteq ubyte %2339, 0		; <bool>:1004 [#uses=1]
	br bool %1004, label %1007, label %1006

; <label>:1005		; preds = %1001, %1007
	add uint %659, 1		; <uint>:683 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %683		; <ubyte*>:1482 [#uses=1]
	load ubyte* %1482		; <ubyte>:2340 [#uses=1]
	seteq ubyte %2340, 0		; <bool>:1005 [#uses=1]
	br bool %1005, label %951, label %950

; <label>:1006		; preds = %1004, %1006
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %660		; <ubyte*>:1483 [#uses=2]
	load ubyte* %1483		; <ubyte>:2341 [#uses=2]
	add ubyte %2341, 255		; <ubyte>:2342 [#uses=1]
	store ubyte %2342, ubyte* %1483
	seteq ubyte %2341, 1		; <bool>:1006 [#uses=1]
	br bool %1006, label %1007, label %1006

; <label>:1007		; preds = %1004, %1006
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %682		; <ubyte*>:1484 [#uses=2]
	load ubyte* %1484		; <ubyte>:2343 [#uses=1]
	add ubyte %2343, 1		; <ubyte>:2344 [#uses=1]
	store ubyte %2344, ubyte* %1484
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %660		; <ubyte*>:1485 [#uses=1]
	load ubyte* %1485		; <ubyte>:2345 [#uses=1]
	seteq ubyte %2345, 0		; <bool>:1007 [#uses=1]
	br bool %1007, label %1005, label %1004

; <label>:1008		; preds = %585, %1379
	phi uint [ %405, %585 ], [ %905, %1379 ]		; <uint>:684 [#uses=71]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %684		; <ubyte*>:1486 [#uses=2]
	load ubyte* %1486		; <ubyte>:2346 [#uses=1]
	add ubyte %2346, 255		; <ubyte>:2347 [#uses=1]
	store ubyte %2347, ubyte* %1486
	add uint %684, 10		; <uint>:685 [#uses=6]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %685		; <ubyte*>:1487 [#uses=1]
	load ubyte* %1487		; <ubyte>:2348 [#uses=1]
	seteq ubyte %2348, 0		; <bool>:1008 [#uses=1]
	br bool %1008, label %1011, label %1010

; <label>:1009		; preds = %585, %1379
	phi uint [ %405, %585 ], [ %905, %1379 ]		; <uint>:686 [#uses=1]
	add uint %686, 4294967295		; <uint>:687 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %687		; <ubyte*>:1488 [#uses=1]
	load ubyte* %1488		; <ubyte>:2349 [#uses=1]
	seteq ubyte %2349, 0		; <bool>:1009 [#uses=1]
	br bool %1009, label %583, label %582

; <label>:1010		; preds = %1008, %1010
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %685		; <ubyte*>:1489 [#uses=2]
	load ubyte* %1489		; <ubyte>:2350 [#uses=2]
	add ubyte %2350, 255		; <ubyte>:2351 [#uses=1]
	store ubyte %2351, ubyte* %1489
	seteq ubyte %2350, 1		; <bool>:1010 [#uses=1]
	br bool %1010, label %1011, label %1010

; <label>:1011		; preds = %1008, %1010
	add uint %684, 6		; <uint>:688 [#uses=7]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %688		; <ubyte*>:1490 [#uses=1]
	load ubyte* %1490		; <ubyte>:2352 [#uses=1]
	seteq ubyte %2352, 0		; <bool>:1011 [#uses=1]
	br bool %1011, label %1013, label %1012

; <label>:1012		; preds = %1011, %1012
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %688		; <ubyte*>:1491 [#uses=2]
	load ubyte* %1491		; <ubyte>:2353 [#uses=1]
	add ubyte %2353, 255		; <ubyte>:2354 [#uses=1]
	store ubyte %2354, ubyte* %1491
	add uint %684, 7		; <uint>:689 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %689		; <ubyte*>:1492 [#uses=2]
	load ubyte* %1492		; <ubyte>:2355 [#uses=1]
	add ubyte %2355, 1		; <ubyte>:2356 [#uses=1]
	store ubyte %2356, ubyte* %1492
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %685		; <ubyte*>:1493 [#uses=2]
	load ubyte* %1493		; <ubyte>:2357 [#uses=1]
	add ubyte %2357, 1		; <ubyte>:2358 [#uses=1]
	store ubyte %2358, ubyte* %1493
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %688		; <ubyte*>:1494 [#uses=1]
	load ubyte* %1494		; <ubyte>:2359 [#uses=1]
	seteq ubyte %2359, 0		; <bool>:1012 [#uses=1]
	br bool %1012, label %1013, label %1012

; <label>:1013		; preds = %1011, %1012
	add uint %684, 7		; <uint>:690 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %690		; <ubyte*>:1495 [#uses=1]
	load ubyte* %1495		; <ubyte>:2360 [#uses=1]
	seteq ubyte %2360, 0		; <bool>:1013 [#uses=1]
	br bool %1013, label %1015, label %1014

; <label>:1014		; preds = %1013, %1014
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %688		; <ubyte*>:1496 [#uses=2]
	load ubyte* %1496		; <ubyte>:2361 [#uses=1]
	add ubyte %2361, 1		; <ubyte>:2362 [#uses=1]
	store ubyte %2362, ubyte* %1496
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %690		; <ubyte*>:1497 [#uses=2]
	load ubyte* %1497		; <ubyte>:2363 [#uses=2]
	add ubyte %2363, 255		; <ubyte>:2364 [#uses=1]
	store ubyte %2364, ubyte* %1497
	seteq ubyte %2363, 1		; <bool>:1014 [#uses=1]
	br bool %1014, label %1015, label %1014

; <label>:1015		; preds = %1013, %1014
	add uint %684, 12		; <uint>:691 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %691		; <ubyte*>:1498 [#uses=2]
	load ubyte* %1498		; <ubyte>:2365 [#uses=2]
	add ubyte %2365, 1		; <ubyte>:2366 [#uses=1]
	store ubyte %2366, ubyte* %1498
	seteq ubyte %2365, 255		; <bool>:1015 [#uses=1]
	br bool %1015, label %1017, label %1016

; <label>:1016		; preds = %1015, %1016
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %685		; <ubyte*>:1499 [#uses=2]
	load ubyte* %1499		; <ubyte>:2367 [#uses=1]
	add ubyte %2367, 255		; <ubyte>:2368 [#uses=1]
	store ubyte %2368, ubyte* %1499
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %691		; <ubyte*>:1500 [#uses=2]
	load ubyte* %1500		; <ubyte>:2369 [#uses=2]
	add ubyte %2369, 255		; <ubyte>:2370 [#uses=1]
	store ubyte %2370, ubyte* %1500
	seteq ubyte %2369, 1		; <bool>:1016 [#uses=1]
	br bool %1016, label %1017, label %1016

; <label>:1017		; preds = %1015, %1016
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %688		; <ubyte*>:1501 [#uses=1]
	load ubyte* %1501		; <ubyte>:2371 [#uses=1]
	seteq ubyte %2371, 0		; <bool>:1017 [#uses=1]
	br bool %1017, label %1019, label %1018

; <label>:1018		; preds = %1017, %1018
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %688		; <ubyte*>:1502 [#uses=2]
	load ubyte* %1502		; <ubyte>:2372 [#uses=2]
	add ubyte %2372, 255		; <ubyte>:2373 [#uses=1]
	store ubyte %2373, ubyte* %1502
	seteq ubyte %2372, 1		; <bool>:1018 [#uses=1]
	br bool %1018, label %1019, label %1018

; <label>:1019		; preds = %1017, %1018
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %685		; <ubyte*>:1503 [#uses=1]
	load ubyte* %1503		; <ubyte>:2374 [#uses=1]
	seteq ubyte %2374, 0		; <bool>:1019 [#uses=1]
	br bool %1019, label %1021, label %1020

; <label>:1020		; preds = %1019, %1020
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %688		; <ubyte*>:1504 [#uses=2]
	load ubyte* %1504		; <ubyte>:2375 [#uses=1]
	add ubyte %2375, 1		; <ubyte>:2376 [#uses=1]
	store ubyte %2376, ubyte* %1504
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %685		; <ubyte*>:1505 [#uses=2]
	load ubyte* %1505		; <ubyte>:2377 [#uses=2]
	add ubyte %2377, 255		; <ubyte>:2378 [#uses=1]
	store ubyte %2378, ubyte* %1505
	seteq ubyte %2377, 1		; <bool>:1020 [#uses=1]
	br bool %1020, label %1021, label %1020

; <label>:1021		; preds = %1019, %1020
	add uint %684, 18		; <uint>:692 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %692		; <ubyte*>:1506 [#uses=1]
	load ubyte* %1506		; <ubyte>:2379 [#uses=1]
	seteq ubyte %2379, 0		; <bool>:1021 [#uses=1]
	br bool %1021, label %1023, label %1022

; <label>:1022		; preds = %1021, %1022
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %692		; <ubyte*>:1507 [#uses=2]
	load ubyte* %1507		; <ubyte>:2380 [#uses=2]
	add ubyte %2380, 255		; <ubyte>:2381 [#uses=1]
	store ubyte %2381, ubyte* %1507
	seteq ubyte %2380, 1		; <bool>:1022 [#uses=1]
	br bool %1022, label %1023, label %1022

; <label>:1023		; preds = %1021, %1022
	add uint %684, 4294967201		; <uint>:693 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %693		; <ubyte*>:1508 [#uses=1]
	load ubyte* %1508		; <ubyte>:2382 [#uses=1]
	seteq ubyte %2382, 0		; <bool>:1023 [#uses=1]
	br bool %1023, label %1025, label %1024

; <label>:1024		; preds = %1023, %1024
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %693		; <ubyte*>:1509 [#uses=2]
	load ubyte* %1509		; <ubyte>:2383 [#uses=1]
	add ubyte %2383, 255		; <ubyte>:2384 [#uses=1]
	store ubyte %2384, ubyte* %1509
	add uint %684, 4294967202		; <uint>:694 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %694		; <ubyte*>:1510 [#uses=2]
	load ubyte* %1510		; <ubyte>:2385 [#uses=1]
	add ubyte %2385, 1		; <ubyte>:2386 [#uses=1]
	store ubyte %2386, ubyte* %1510
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %692		; <ubyte*>:1511 [#uses=2]
	load ubyte* %1511		; <ubyte>:2387 [#uses=1]
	add ubyte %2387, 1		; <ubyte>:2388 [#uses=1]
	store ubyte %2388, ubyte* %1511
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %693		; <ubyte*>:1512 [#uses=1]
	load ubyte* %1512		; <ubyte>:2389 [#uses=1]
	seteq ubyte %2389, 0		; <bool>:1024 [#uses=1]
	br bool %1024, label %1025, label %1024

; <label>:1025		; preds = %1023, %1024
	add uint %684, 4294967202		; <uint>:695 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %695		; <ubyte*>:1513 [#uses=1]
	load ubyte* %1513		; <ubyte>:2390 [#uses=1]
	seteq ubyte %2390, 0		; <bool>:1025 [#uses=1]
	br bool %1025, label %1027, label %1026

; <label>:1026		; preds = %1025, %1026
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %693		; <ubyte*>:1514 [#uses=2]
	load ubyte* %1514		; <ubyte>:2391 [#uses=1]
	add ubyte %2391, 1		; <ubyte>:2392 [#uses=1]
	store ubyte %2392, ubyte* %1514
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %695		; <ubyte*>:1515 [#uses=2]
	load ubyte* %1515		; <ubyte>:2393 [#uses=2]
	add ubyte %2393, 255		; <ubyte>:2394 [#uses=1]
	store ubyte %2394, ubyte* %1515
	seteq ubyte %2393, 1		; <bool>:1026 [#uses=1]
	br bool %1026, label %1027, label %1026

; <label>:1027		; preds = %1025, %1026
	add uint %684, 24		; <uint>:696 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %696		; <ubyte*>:1516 [#uses=1]
	load ubyte* %1516		; <ubyte>:2395 [#uses=1]
	seteq ubyte %2395, 0		; <bool>:1027 [#uses=1]
	br bool %1027, label %1029, label %1028

; <label>:1028		; preds = %1027, %1028
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %696		; <ubyte*>:1517 [#uses=2]
	load ubyte* %1517		; <ubyte>:2396 [#uses=2]
	add ubyte %2396, 255		; <ubyte>:2397 [#uses=1]
	store ubyte %2397, ubyte* %1517
	seteq ubyte %2396, 1		; <bool>:1028 [#uses=1]
	br bool %1028, label %1029, label %1028

; <label>:1029		; preds = %1027, %1028
	add uint %684, 4294967207		; <uint>:697 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %697		; <ubyte*>:1518 [#uses=1]
	load ubyte* %1518		; <ubyte>:2398 [#uses=1]
	seteq ubyte %2398, 0		; <bool>:1029 [#uses=1]
	br bool %1029, label %1031, label %1030

; <label>:1030		; preds = %1029, %1030
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %697		; <ubyte*>:1519 [#uses=2]
	load ubyte* %1519		; <ubyte>:2399 [#uses=1]
	add ubyte %2399, 255		; <ubyte>:2400 [#uses=1]
	store ubyte %2400, ubyte* %1519
	add uint %684, 4294967208		; <uint>:698 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %698		; <ubyte*>:1520 [#uses=2]
	load ubyte* %1520		; <ubyte>:2401 [#uses=1]
	add ubyte %2401, 1		; <ubyte>:2402 [#uses=1]
	store ubyte %2402, ubyte* %1520
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %696		; <ubyte*>:1521 [#uses=2]
	load ubyte* %1521		; <ubyte>:2403 [#uses=1]
	add ubyte %2403, 1		; <ubyte>:2404 [#uses=1]
	store ubyte %2404, ubyte* %1521
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %697		; <ubyte*>:1522 [#uses=1]
	load ubyte* %1522		; <ubyte>:2405 [#uses=1]
	seteq ubyte %2405, 0		; <bool>:1030 [#uses=1]
	br bool %1030, label %1031, label %1030

; <label>:1031		; preds = %1029, %1030
	add uint %684, 4294967208		; <uint>:699 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %699		; <ubyte*>:1523 [#uses=1]
	load ubyte* %1523		; <ubyte>:2406 [#uses=1]
	seteq ubyte %2406, 0		; <bool>:1031 [#uses=1]
	br bool %1031, label %1033, label %1032

; <label>:1032		; preds = %1031, %1032
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %697		; <ubyte*>:1524 [#uses=2]
	load ubyte* %1524		; <ubyte>:2407 [#uses=1]
	add ubyte %2407, 1		; <ubyte>:2408 [#uses=1]
	store ubyte %2408, ubyte* %1524
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %699		; <ubyte*>:1525 [#uses=2]
	load ubyte* %1525		; <ubyte>:2409 [#uses=2]
	add ubyte %2409, 255		; <ubyte>:2410 [#uses=1]
	store ubyte %2410, ubyte* %1525
	seteq ubyte %2409, 1		; <bool>:1032 [#uses=1]
	br bool %1032, label %1033, label %1032

; <label>:1033		; preds = %1031, %1032
	add uint %684, 30		; <uint>:700 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %700		; <ubyte*>:1526 [#uses=1]
	load ubyte* %1526		; <ubyte>:2411 [#uses=1]
	seteq ubyte %2411, 0		; <bool>:1033 [#uses=1]
	br bool %1033, label %1035, label %1034

; <label>:1034		; preds = %1033, %1034
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %700		; <ubyte*>:1527 [#uses=2]
	load ubyte* %1527		; <ubyte>:2412 [#uses=2]
	add ubyte %2412, 255		; <ubyte>:2413 [#uses=1]
	store ubyte %2413, ubyte* %1527
	seteq ubyte %2412, 1		; <bool>:1034 [#uses=1]
	br bool %1034, label %1035, label %1034

; <label>:1035		; preds = %1033, %1034
	add uint %684, 4294967213		; <uint>:701 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %701		; <ubyte*>:1528 [#uses=1]
	load ubyte* %1528		; <ubyte>:2414 [#uses=1]
	seteq ubyte %2414, 0		; <bool>:1035 [#uses=1]
	br bool %1035, label %1037, label %1036

; <label>:1036		; preds = %1035, %1036
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %701		; <ubyte*>:1529 [#uses=2]
	load ubyte* %1529		; <ubyte>:2415 [#uses=1]
	add ubyte %2415, 255		; <ubyte>:2416 [#uses=1]
	store ubyte %2416, ubyte* %1529
	add uint %684, 4294967214		; <uint>:702 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %702		; <ubyte*>:1530 [#uses=2]
	load ubyte* %1530		; <ubyte>:2417 [#uses=1]
	add ubyte %2417, 1		; <ubyte>:2418 [#uses=1]
	store ubyte %2418, ubyte* %1530
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %700		; <ubyte*>:1531 [#uses=2]
	load ubyte* %1531		; <ubyte>:2419 [#uses=1]
	add ubyte %2419, 1		; <ubyte>:2420 [#uses=1]
	store ubyte %2420, ubyte* %1531
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %701		; <ubyte*>:1532 [#uses=1]
	load ubyte* %1532		; <ubyte>:2421 [#uses=1]
	seteq ubyte %2421, 0		; <bool>:1036 [#uses=1]
	br bool %1036, label %1037, label %1036

; <label>:1037		; preds = %1035, %1036
	add uint %684, 4294967214		; <uint>:703 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %703		; <ubyte*>:1533 [#uses=1]
	load ubyte* %1533		; <ubyte>:2422 [#uses=1]
	seteq ubyte %2422, 0		; <bool>:1037 [#uses=1]
	br bool %1037, label %1039, label %1038

; <label>:1038		; preds = %1037, %1038
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %701		; <ubyte*>:1534 [#uses=2]
	load ubyte* %1534		; <ubyte>:2423 [#uses=1]
	add ubyte %2423, 1		; <ubyte>:2424 [#uses=1]
	store ubyte %2424, ubyte* %1534
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %703		; <ubyte*>:1535 [#uses=2]
	load ubyte* %1535		; <ubyte>:2425 [#uses=2]
	add ubyte %2425, 255		; <ubyte>:2426 [#uses=1]
	store ubyte %2426, ubyte* %1535
	seteq ubyte %2425, 1		; <bool>:1038 [#uses=1]
	br bool %1038, label %1039, label %1038

; <label>:1039		; preds = %1037, %1038
	add uint %684, 36		; <uint>:704 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %704		; <ubyte*>:1536 [#uses=1]
	load ubyte* %1536		; <ubyte>:2427 [#uses=1]
	seteq ubyte %2427, 0		; <bool>:1039 [#uses=1]
	br bool %1039, label %1041, label %1040

; <label>:1040		; preds = %1039, %1040
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %704		; <ubyte*>:1537 [#uses=2]
	load ubyte* %1537		; <ubyte>:2428 [#uses=2]
	add ubyte %2428, 255		; <ubyte>:2429 [#uses=1]
	store ubyte %2429, ubyte* %1537
	seteq ubyte %2428, 1		; <bool>:1040 [#uses=1]
	br bool %1040, label %1041, label %1040

; <label>:1041		; preds = %1039, %1040
	add uint %684, 4294967219		; <uint>:705 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %705		; <ubyte*>:1538 [#uses=1]
	load ubyte* %1538		; <ubyte>:2430 [#uses=1]
	seteq ubyte %2430, 0		; <bool>:1041 [#uses=1]
	br bool %1041, label %1043, label %1042

; <label>:1042		; preds = %1041, %1042
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %705		; <ubyte*>:1539 [#uses=2]
	load ubyte* %1539		; <ubyte>:2431 [#uses=1]
	add ubyte %2431, 255		; <ubyte>:2432 [#uses=1]
	store ubyte %2432, ubyte* %1539
	add uint %684, 4294967220		; <uint>:706 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %706		; <ubyte*>:1540 [#uses=2]
	load ubyte* %1540		; <ubyte>:2433 [#uses=1]
	add ubyte %2433, 1		; <ubyte>:2434 [#uses=1]
	store ubyte %2434, ubyte* %1540
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %704		; <ubyte*>:1541 [#uses=2]
	load ubyte* %1541		; <ubyte>:2435 [#uses=1]
	add ubyte %2435, 1		; <ubyte>:2436 [#uses=1]
	store ubyte %2436, ubyte* %1541
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %705		; <ubyte*>:1542 [#uses=1]
	load ubyte* %1542		; <ubyte>:2437 [#uses=1]
	seteq ubyte %2437, 0		; <bool>:1042 [#uses=1]
	br bool %1042, label %1043, label %1042

; <label>:1043		; preds = %1041, %1042
	add uint %684, 4294967220		; <uint>:707 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %707		; <ubyte*>:1543 [#uses=1]
	load ubyte* %1543		; <ubyte>:2438 [#uses=1]
	seteq ubyte %2438, 0		; <bool>:1043 [#uses=1]
	br bool %1043, label %1045, label %1044

; <label>:1044		; preds = %1043, %1044
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %705		; <ubyte*>:1544 [#uses=2]
	load ubyte* %1544		; <ubyte>:2439 [#uses=1]
	add ubyte %2439, 1		; <ubyte>:2440 [#uses=1]
	store ubyte %2440, ubyte* %1544
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %707		; <ubyte*>:1545 [#uses=2]
	load ubyte* %1545		; <ubyte>:2441 [#uses=2]
	add ubyte %2441, 255		; <ubyte>:2442 [#uses=1]
	store ubyte %2442, ubyte* %1545
	seteq ubyte %2441, 1		; <bool>:1044 [#uses=1]
	br bool %1044, label %1045, label %1044

; <label>:1045		; preds = %1043, %1044
	add uint %684, 42		; <uint>:708 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %708		; <ubyte*>:1546 [#uses=1]
	load ubyte* %1546		; <ubyte>:2443 [#uses=1]
	seteq ubyte %2443, 0		; <bool>:1045 [#uses=1]
	br bool %1045, label %1047, label %1046

; <label>:1046		; preds = %1045, %1046
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %708		; <ubyte*>:1547 [#uses=2]
	load ubyte* %1547		; <ubyte>:2444 [#uses=2]
	add ubyte %2444, 255		; <ubyte>:2445 [#uses=1]
	store ubyte %2445, ubyte* %1547
	seteq ubyte %2444, 1		; <bool>:1046 [#uses=1]
	br bool %1046, label %1047, label %1046

; <label>:1047		; preds = %1045, %1046
	add uint %684, 4294967225		; <uint>:709 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %709		; <ubyte*>:1548 [#uses=1]
	load ubyte* %1548		; <ubyte>:2446 [#uses=1]
	seteq ubyte %2446, 0		; <bool>:1047 [#uses=1]
	br bool %1047, label %1049, label %1048

; <label>:1048		; preds = %1047, %1048
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %709		; <ubyte*>:1549 [#uses=2]
	load ubyte* %1549		; <ubyte>:2447 [#uses=1]
	add ubyte %2447, 255		; <ubyte>:2448 [#uses=1]
	store ubyte %2448, ubyte* %1549
	add uint %684, 4294967226		; <uint>:710 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %710		; <ubyte*>:1550 [#uses=2]
	load ubyte* %1550		; <ubyte>:2449 [#uses=1]
	add ubyte %2449, 1		; <ubyte>:2450 [#uses=1]
	store ubyte %2450, ubyte* %1550
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %708		; <ubyte*>:1551 [#uses=2]
	load ubyte* %1551		; <ubyte>:2451 [#uses=1]
	add ubyte %2451, 1		; <ubyte>:2452 [#uses=1]
	store ubyte %2452, ubyte* %1551
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %709		; <ubyte*>:1552 [#uses=1]
	load ubyte* %1552		; <ubyte>:2453 [#uses=1]
	seteq ubyte %2453, 0		; <bool>:1048 [#uses=1]
	br bool %1048, label %1049, label %1048

; <label>:1049		; preds = %1047, %1048
	add uint %684, 4294967226		; <uint>:711 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %711		; <ubyte*>:1553 [#uses=1]
	load ubyte* %1553		; <ubyte>:2454 [#uses=1]
	seteq ubyte %2454, 0		; <bool>:1049 [#uses=1]
	br bool %1049, label %1051, label %1050

; <label>:1050		; preds = %1049, %1050
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %709		; <ubyte*>:1554 [#uses=2]
	load ubyte* %1554		; <ubyte>:2455 [#uses=1]
	add ubyte %2455, 1		; <ubyte>:2456 [#uses=1]
	store ubyte %2456, ubyte* %1554
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %711		; <ubyte*>:1555 [#uses=2]
	load ubyte* %1555		; <ubyte>:2457 [#uses=2]
	add ubyte %2457, 255		; <ubyte>:2458 [#uses=1]
	store ubyte %2458, ubyte* %1555
	seteq ubyte %2457, 1		; <bool>:1050 [#uses=1]
	br bool %1050, label %1051, label %1050

; <label>:1051		; preds = %1049, %1050
	add uint %684, 48		; <uint>:712 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %712		; <ubyte*>:1556 [#uses=1]
	load ubyte* %1556		; <ubyte>:2459 [#uses=1]
	seteq ubyte %2459, 0		; <bool>:1051 [#uses=1]
	br bool %1051, label %1053, label %1052

; <label>:1052		; preds = %1051, %1052
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %712		; <ubyte*>:1557 [#uses=2]
	load ubyte* %1557		; <ubyte>:2460 [#uses=2]
	add ubyte %2460, 255		; <ubyte>:2461 [#uses=1]
	store ubyte %2461, ubyte* %1557
	seteq ubyte %2460, 1		; <bool>:1052 [#uses=1]
	br bool %1052, label %1053, label %1052

; <label>:1053		; preds = %1051, %1052
	add uint %684, 4294967231		; <uint>:713 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %713		; <ubyte*>:1558 [#uses=1]
	load ubyte* %1558		; <ubyte>:2462 [#uses=1]
	seteq ubyte %2462, 0		; <bool>:1053 [#uses=1]
	br bool %1053, label %1055, label %1054

; <label>:1054		; preds = %1053, %1054
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %713		; <ubyte*>:1559 [#uses=2]
	load ubyte* %1559		; <ubyte>:2463 [#uses=1]
	add ubyte %2463, 255		; <ubyte>:2464 [#uses=1]
	store ubyte %2464, ubyte* %1559
	add uint %684, 4294967232		; <uint>:714 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %714		; <ubyte*>:1560 [#uses=2]
	load ubyte* %1560		; <ubyte>:2465 [#uses=1]
	add ubyte %2465, 1		; <ubyte>:2466 [#uses=1]
	store ubyte %2466, ubyte* %1560
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %712		; <ubyte*>:1561 [#uses=2]
	load ubyte* %1561		; <ubyte>:2467 [#uses=1]
	add ubyte %2467, 1		; <ubyte>:2468 [#uses=1]
	store ubyte %2468, ubyte* %1561
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %713		; <ubyte*>:1562 [#uses=1]
	load ubyte* %1562		; <ubyte>:2469 [#uses=1]
	seteq ubyte %2469, 0		; <bool>:1054 [#uses=1]
	br bool %1054, label %1055, label %1054

; <label>:1055		; preds = %1053, %1054
	add uint %684, 4294967232		; <uint>:715 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %715		; <ubyte*>:1563 [#uses=1]
	load ubyte* %1563		; <ubyte>:2470 [#uses=1]
	seteq ubyte %2470, 0		; <bool>:1055 [#uses=1]
	br bool %1055, label %1057, label %1056

; <label>:1056		; preds = %1055, %1056
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %713		; <ubyte*>:1564 [#uses=2]
	load ubyte* %1564		; <ubyte>:2471 [#uses=1]
	add ubyte %2471, 1		; <ubyte>:2472 [#uses=1]
	store ubyte %2472, ubyte* %1564
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %715		; <ubyte*>:1565 [#uses=2]
	load ubyte* %1565		; <ubyte>:2473 [#uses=2]
	add ubyte %2473, 255		; <ubyte>:2474 [#uses=1]
	store ubyte %2474, ubyte* %1565
	seteq ubyte %2473, 1		; <bool>:1056 [#uses=1]
	br bool %1056, label %1057, label %1056

; <label>:1057		; preds = %1055, %1056
	add uint %684, 54		; <uint>:716 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %716		; <ubyte*>:1566 [#uses=1]
	load ubyte* %1566		; <ubyte>:2475 [#uses=1]
	seteq ubyte %2475, 0		; <bool>:1057 [#uses=1]
	br bool %1057, label %1059, label %1058

; <label>:1058		; preds = %1057, %1058
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %716		; <ubyte*>:1567 [#uses=2]
	load ubyte* %1567		; <ubyte>:2476 [#uses=2]
	add ubyte %2476, 255		; <ubyte>:2477 [#uses=1]
	store ubyte %2477, ubyte* %1567
	seteq ubyte %2476, 1		; <bool>:1058 [#uses=1]
	br bool %1058, label %1059, label %1058

; <label>:1059		; preds = %1057, %1058
	add uint %684, 4294967237		; <uint>:717 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %717		; <ubyte*>:1568 [#uses=1]
	load ubyte* %1568		; <ubyte>:2478 [#uses=1]
	seteq ubyte %2478, 0		; <bool>:1059 [#uses=1]
	br bool %1059, label %1061, label %1060

; <label>:1060		; preds = %1059, %1060
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %717		; <ubyte*>:1569 [#uses=2]
	load ubyte* %1569		; <ubyte>:2479 [#uses=1]
	add ubyte %2479, 255		; <ubyte>:2480 [#uses=1]
	store ubyte %2480, ubyte* %1569
	add uint %684, 4294967238		; <uint>:718 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %718		; <ubyte*>:1570 [#uses=2]
	load ubyte* %1570		; <ubyte>:2481 [#uses=1]
	add ubyte %2481, 1		; <ubyte>:2482 [#uses=1]
	store ubyte %2482, ubyte* %1570
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %716		; <ubyte*>:1571 [#uses=2]
	load ubyte* %1571		; <ubyte>:2483 [#uses=1]
	add ubyte %2483, 1		; <ubyte>:2484 [#uses=1]
	store ubyte %2484, ubyte* %1571
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %717		; <ubyte*>:1572 [#uses=1]
	load ubyte* %1572		; <ubyte>:2485 [#uses=1]
	seteq ubyte %2485, 0		; <bool>:1060 [#uses=1]
	br bool %1060, label %1061, label %1060

; <label>:1061		; preds = %1059, %1060
	add uint %684, 4294967238		; <uint>:719 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %719		; <ubyte*>:1573 [#uses=1]
	load ubyte* %1573		; <ubyte>:2486 [#uses=1]
	seteq ubyte %2486, 0		; <bool>:1061 [#uses=1]
	br bool %1061, label %1063, label %1062

; <label>:1062		; preds = %1061, %1062
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %717		; <ubyte*>:1574 [#uses=2]
	load ubyte* %1574		; <ubyte>:2487 [#uses=1]
	add ubyte %2487, 1		; <ubyte>:2488 [#uses=1]
	store ubyte %2488, ubyte* %1574
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %719		; <ubyte*>:1575 [#uses=2]
	load ubyte* %1575		; <ubyte>:2489 [#uses=2]
	add ubyte %2489, 255		; <ubyte>:2490 [#uses=1]
	store ubyte %2490, ubyte* %1575
	seteq ubyte %2489, 1		; <bool>:1062 [#uses=1]
	br bool %1062, label %1063, label %1062

; <label>:1063		; preds = %1061, %1062
	add uint %684, 60		; <uint>:720 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %720		; <ubyte*>:1576 [#uses=1]
	load ubyte* %1576		; <ubyte>:2491 [#uses=1]
	seteq ubyte %2491, 0		; <bool>:1063 [#uses=1]
	br bool %1063, label %1065, label %1064

; <label>:1064		; preds = %1063, %1064
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %720		; <ubyte*>:1577 [#uses=2]
	load ubyte* %1577		; <ubyte>:2492 [#uses=2]
	add ubyte %2492, 255		; <ubyte>:2493 [#uses=1]
	store ubyte %2493, ubyte* %1577
	seteq ubyte %2492, 1		; <bool>:1064 [#uses=1]
	br bool %1064, label %1065, label %1064

; <label>:1065		; preds = %1063, %1064
	add uint %684, 4294967243		; <uint>:721 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %721		; <ubyte*>:1578 [#uses=1]
	load ubyte* %1578		; <ubyte>:2494 [#uses=1]
	seteq ubyte %2494, 0		; <bool>:1065 [#uses=1]
	br bool %1065, label %1067, label %1066

; <label>:1066		; preds = %1065, %1066
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %721		; <ubyte*>:1579 [#uses=2]
	load ubyte* %1579		; <ubyte>:2495 [#uses=1]
	add ubyte %2495, 255		; <ubyte>:2496 [#uses=1]
	store ubyte %2496, ubyte* %1579
	add uint %684, 4294967244		; <uint>:722 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %722		; <ubyte*>:1580 [#uses=2]
	load ubyte* %1580		; <ubyte>:2497 [#uses=1]
	add ubyte %2497, 1		; <ubyte>:2498 [#uses=1]
	store ubyte %2498, ubyte* %1580
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %720		; <ubyte*>:1581 [#uses=2]
	load ubyte* %1581		; <ubyte>:2499 [#uses=1]
	add ubyte %2499, 1		; <ubyte>:2500 [#uses=1]
	store ubyte %2500, ubyte* %1581
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %721		; <ubyte*>:1582 [#uses=1]
	load ubyte* %1582		; <ubyte>:2501 [#uses=1]
	seteq ubyte %2501, 0		; <bool>:1066 [#uses=1]
	br bool %1066, label %1067, label %1066

; <label>:1067		; preds = %1065, %1066
	add uint %684, 4294967244		; <uint>:723 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %723		; <ubyte*>:1583 [#uses=1]
	load ubyte* %1583		; <ubyte>:2502 [#uses=1]
	seteq ubyte %2502, 0		; <bool>:1067 [#uses=1]
	br bool %1067, label %1069, label %1068

; <label>:1068		; preds = %1067, %1068
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %721		; <ubyte*>:1584 [#uses=2]
	load ubyte* %1584		; <ubyte>:2503 [#uses=1]
	add ubyte %2503, 1		; <ubyte>:2504 [#uses=1]
	store ubyte %2504, ubyte* %1584
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %723		; <ubyte*>:1585 [#uses=2]
	load ubyte* %1585		; <ubyte>:2505 [#uses=2]
	add ubyte %2505, 255		; <ubyte>:2506 [#uses=1]
	store ubyte %2506, ubyte* %1585
	seteq ubyte %2505, 1		; <bool>:1068 [#uses=1]
	br bool %1068, label %1069, label %1068

; <label>:1069		; preds = %1067, %1068
	add uint %684, 66		; <uint>:724 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %724		; <ubyte*>:1586 [#uses=1]
	load ubyte* %1586		; <ubyte>:2507 [#uses=1]
	seteq ubyte %2507, 0		; <bool>:1069 [#uses=1]
	br bool %1069, label %1071, label %1070

; <label>:1070		; preds = %1069, %1070
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %724		; <ubyte*>:1587 [#uses=2]
	load ubyte* %1587		; <ubyte>:2508 [#uses=2]
	add ubyte %2508, 255		; <ubyte>:2509 [#uses=1]
	store ubyte %2509, ubyte* %1587
	seteq ubyte %2508, 1		; <bool>:1070 [#uses=1]
	br bool %1070, label %1071, label %1070

; <label>:1071		; preds = %1069, %1070
	add uint %684, 4294967249		; <uint>:725 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %725		; <ubyte*>:1588 [#uses=1]
	load ubyte* %1588		; <ubyte>:2510 [#uses=1]
	seteq ubyte %2510, 0		; <bool>:1071 [#uses=1]
	br bool %1071, label %1073, label %1072

; <label>:1072		; preds = %1071, %1072
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %725		; <ubyte*>:1589 [#uses=2]
	load ubyte* %1589		; <ubyte>:2511 [#uses=1]
	add ubyte %2511, 255		; <ubyte>:2512 [#uses=1]
	store ubyte %2512, ubyte* %1589
	add uint %684, 4294967250		; <uint>:726 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %726		; <ubyte*>:1590 [#uses=2]
	load ubyte* %1590		; <ubyte>:2513 [#uses=1]
	add ubyte %2513, 1		; <ubyte>:2514 [#uses=1]
	store ubyte %2514, ubyte* %1590
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %724		; <ubyte*>:1591 [#uses=2]
	load ubyte* %1591		; <ubyte>:2515 [#uses=1]
	add ubyte %2515, 1		; <ubyte>:2516 [#uses=1]
	store ubyte %2516, ubyte* %1591
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %725		; <ubyte*>:1592 [#uses=1]
	load ubyte* %1592		; <ubyte>:2517 [#uses=1]
	seteq ubyte %2517, 0		; <bool>:1072 [#uses=1]
	br bool %1072, label %1073, label %1072

; <label>:1073		; preds = %1071, %1072
	add uint %684, 4294967250		; <uint>:727 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %727		; <ubyte*>:1593 [#uses=1]
	load ubyte* %1593		; <ubyte>:2518 [#uses=1]
	seteq ubyte %2518, 0		; <bool>:1073 [#uses=1]
	br bool %1073, label %1075, label %1074

; <label>:1074		; preds = %1073, %1074
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %725		; <ubyte*>:1594 [#uses=2]
	load ubyte* %1594		; <ubyte>:2519 [#uses=1]
	add ubyte %2519, 1		; <ubyte>:2520 [#uses=1]
	store ubyte %2520, ubyte* %1594
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %727		; <ubyte*>:1595 [#uses=2]
	load ubyte* %1595		; <ubyte>:2521 [#uses=2]
	add ubyte %2521, 255		; <ubyte>:2522 [#uses=1]
	store ubyte %2522, ubyte* %1595
	seteq ubyte %2521, 1		; <bool>:1074 [#uses=1]
	br bool %1074, label %1075, label %1074

; <label>:1075		; preds = %1073, %1074
	add uint %684, 72		; <uint>:728 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %728		; <ubyte*>:1596 [#uses=1]
	load ubyte* %1596		; <ubyte>:2523 [#uses=1]
	seteq ubyte %2523, 0		; <bool>:1075 [#uses=1]
	br bool %1075, label %1077, label %1076

; <label>:1076		; preds = %1075, %1076
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %728		; <ubyte*>:1597 [#uses=2]
	load ubyte* %1597		; <ubyte>:2524 [#uses=2]
	add ubyte %2524, 255		; <ubyte>:2525 [#uses=1]
	store ubyte %2525, ubyte* %1597
	seteq ubyte %2524, 1		; <bool>:1076 [#uses=1]
	br bool %1076, label %1077, label %1076

; <label>:1077		; preds = %1075, %1076
	add uint %684, 4294967255		; <uint>:729 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %729		; <ubyte*>:1598 [#uses=1]
	load ubyte* %1598		; <ubyte>:2526 [#uses=1]
	seteq ubyte %2526, 0		; <bool>:1077 [#uses=1]
	br bool %1077, label %1079, label %1078

; <label>:1078		; preds = %1077, %1078
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %729		; <ubyte*>:1599 [#uses=2]
	load ubyte* %1599		; <ubyte>:2527 [#uses=1]
	add ubyte %2527, 255		; <ubyte>:2528 [#uses=1]
	store ubyte %2528, ubyte* %1599
	add uint %684, 4294967256		; <uint>:730 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %730		; <ubyte*>:1600 [#uses=2]
	load ubyte* %1600		; <ubyte>:2529 [#uses=1]
	add ubyte %2529, 1		; <ubyte>:2530 [#uses=1]
	store ubyte %2530, ubyte* %1600
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %728		; <ubyte*>:1601 [#uses=2]
	load ubyte* %1601		; <ubyte>:2531 [#uses=1]
	add ubyte %2531, 1		; <ubyte>:2532 [#uses=1]
	store ubyte %2532, ubyte* %1601
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %729		; <ubyte*>:1602 [#uses=1]
	load ubyte* %1602		; <ubyte>:2533 [#uses=1]
	seteq ubyte %2533, 0		; <bool>:1078 [#uses=1]
	br bool %1078, label %1079, label %1078

; <label>:1079		; preds = %1077, %1078
	add uint %684, 4294967256		; <uint>:731 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %731		; <ubyte*>:1603 [#uses=1]
	load ubyte* %1603		; <ubyte>:2534 [#uses=1]
	seteq ubyte %2534, 0		; <bool>:1079 [#uses=1]
	br bool %1079, label %1081, label %1080

; <label>:1080		; preds = %1079, %1080
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %729		; <ubyte*>:1604 [#uses=2]
	load ubyte* %1604		; <ubyte>:2535 [#uses=1]
	add ubyte %2535, 1		; <ubyte>:2536 [#uses=1]
	store ubyte %2536, ubyte* %1604
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %731		; <ubyte*>:1605 [#uses=2]
	load ubyte* %1605		; <ubyte>:2537 [#uses=2]
	add ubyte %2537, 255		; <ubyte>:2538 [#uses=1]
	store ubyte %2538, ubyte* %1605
	seteq ubyte %2537, 1		; <bool>:1080 [#uses=1]
	br bool %1080, label %1081, label %1080

; <label>:1081		; preds = %1079, %1080
	add uint %684, 78		; <uint>:732 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %732		; <ubyte*>:1606 [#uses=1]
	load ubyte* %1606		; <ubyte>:2539 [#uses=1]
	seteq ubyte %2539, 0		; <bool>:1081 [#uses=1]
	br bool %1081, label %1083, label %1082

; <label>:1082		; preds = %1081, %1082
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %732		; <ubyte*>:1607 [#uses=2]
	load ubyte* %1607		; <ubyte>:2540 [#uses=2]
	add ubyte %2540, 255		; <ubyte>:2541 [#uses=1]
	store ubyte %2541, ubyte* %1607
	seteq ubyte %2540, 1		; <bool>:1082 [#uses=1]
	br bool %1082, label %1083, label %1082

; <label>:1083		; preds = %1081, %1082
	add uint %684, 4294967261		; <uint>:733 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %733		; <ubyte*>:1608 [#uses=1]
	load ubyte* %1608		; <ubyte>:2542 [#uses=1]
	seteq ubyte %2542, 0		; <bool>:1083 [#uses=1]
	br bool %1083, label %1085, label %1084

; <label>:1084		; preds = %1083, %1084
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %733		; <ubyte*>:1609 [#uses=2]
	load ubyte* %1609		; <ubyte>:2543 [#uses=1]
	add ubyte %2543, 255		; <ubyte>:2544 [#uses=1]
	store ubyte %2544, ubyte* %1609
	add uint %684, 4294967262		; <uint>:734 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %734		; <ubyte*>:1610 [#uses=2]
	load ubyte* %1610		; <ubyte>:2545 [#uses=1]
	add ubyte %2545, 1		; <ubyte>:2546 [#uses=1]
	store ubyte %2546, ubyte* %1610
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %732		; <ubyte*>:1611 [#uses=2]
	load ubyte* %1611		; <ubyte>:2547 [#uses=1]
	add ubyte %2547, 1		; <ubyte>:2548 [#uses=1]
	store ubyte %2548, ubyte* %1611
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %733		; <ubyte*>:1612 [#uses=1]
	load ubyte* %1612		; <ubyte>:2549 [#uses=1]
	seteq ubyte %2549, 0		; <bool>:1084 [#uses=1]
	br bool %1084, label %1085, label %1084

; <label>:1085		; preds = %1083, %1084
	add uint %684, 4294967262		; <uint>:735 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %735		; <ubyte*>:1613 [#uses=1]
	load ubyte* %1613		; <ubyte>:2550 [#uses=1]
	seteq ubyte %2550, 0		; <bool>:1085 [#uses=1]
	br bool %1085, label %1087, label %1086

; <label>:1086		; preds = %1085, %1086
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %733		; <ubyte*>:1614 [#uses=2]
	load ubyte* %1614		; <ubyte>:2551 [#uses=1]
	add ubyte %2551, 1		; <ubyte>:2552 [#uses=1]
	store ubyte %2552, ubyte* %1614
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %735		; <ubyte*>:1615 [#uses=2]
	load ubyte* %1615		; <ubyte>:2553 [#uses=2]
	add ubyte %2553, 255		; <ubyte>:2554 [#uses=1]
	store ubyte %2554, ubyte* %1615
	seteq ubyte %2553, 1		; <bool>:1086 [#uses=1]
	br bool %1086, label %1087, label %1086

; <label>:1087		; preds = %1085, %1086
	add uint %684, 84		; <uint>:736 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %736		; <ubyte*>:1616 [#uses=1]
	load ubyte* %1616		; <ubyte>:2555 [#uses=1]
	seteq ubyte %2555, 0		; <bool>:1087 [#uses=1]
	br bool %1087, label %1089, label %1088

; <label>:1088		; preds = %1087, %1088
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %736		; <ubyte*>:1617 [#uses=2]
	load ubyte* %1617		; <ubyte>:2556 [#uses=2]
	add ubyte %2556, 255		; <ubyte>:2557 [#uses=1]
	store ubyte %2557, ubyte* %1617
	seteq ubyte %2556, 1		; <bool>:1088 [#uses=1]
	br bool %1088, label %1089, label %1088

; <label>:1089		; preds = %1087, %1088
	add uint %684, 4294967267		; <uint>:737 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %737		; <ubyte*>:1618 [#uses=1]
	load ubyte* %1618		; <ubyte>:2558 [#uses=1]
	seteq ubyte %2558, 0		; <bool>:1089 [#uses=1]
	br bool %1089, label %1091, label %1090

; <label>:1090		; preds = %1089, %1090
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %737		; <ubyte*>:1619 [#uses=2]
	load ubyte* %1619		; <ubyte>:2559 [#uses=1]
	add ubyte %2559, 255		; <ubyte>:2560 [#uses=1]
	store ubyte %2560, ubyte* %1619
	add uint %684, 4294967268		; <uint>:738 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %738		; <ubyte*>:1620 [#uses=2]
	load ubyte* %1620		; <ubyte>:2561 [#uses=1]
	add ubyte %2561, 1		; <ubyte>:2562 [#uses=1]
	store ubyte %2562, ubyte* %1620
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %736		; <ubyte*>:1621 [#uses=2]
	load ubyte* %1621		; <ubyte>:2563 [#uses=1]
	add ubyte %2563, 1		; <ubyte>:2564 [#uses=1]
	store ubyte %2564, ubyte* %1621
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %737		; <ubyte*>:1622 [#uses=1]
	load ubyte* %1622		; <ubyte>:2565 [#uses=1]
	seteq ubyte %2565, 0		; <bool>:1090 [#uses=1]
	br bool %1090, label %1091, label %1090

; <label>:1091		; preds = %1089, %1090
	add uint %684, 4294967268		; <uint>:739 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %739		; <ubyte*>:1623 [#uses=1]
	load ubyte* %1623		; <ubyte>:2566 [#uses=1]
	seteq ubyte %2566, 0		; <bool>:1091 [#uses=1]
	br bool %1091, label %1093, label %1092

; <label>:1092		; preds = %1091, %1092
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %737		; <ubyte*>:1624 [#uses=2]
	load ubyte* %1624		; <ubyte>:2567 [#uses=1]
	add ubyte %2567, 1		; <ubyte>:2568 [#uses=1]
	store ubyte %2568, ubyte* %1624
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %739		; <ubyte*>:1625 [#uses=2]
	load ubyte* %1625		; <ubyte>:2569 [#uses=2]
	add ubyte %2569, 255		; <ubyte>:2570 [#uses=1]
	store ubyte %2570, ubyte* %1625
	seteq ubyte %2569, 1		; <bool>:1092 [#uses=1]
	br bool %1092, label %1093, label %1092

; <label>:1093		; preds = %1091, %1092
	add uint %684, 90		; <uint>:740 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %740		; <ubyte*>:1626 [#uses=1]
	load ubyte* %1626		; <ubyte>:2571 [#uses=1]
	seteq ubyte %2571, 0		; <bool>:1093 [#uses=1]
	br bool %1093, label %1095, label %1094

; <label>:1094		; preds = %1093, %1094
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %740		; <ubyte*>:1627 [#uses=2]
	load ubyte* %1627		; <ubyte>:2572 [#uses=2]
	add ubyte %2572, 255		; <ubyte>:2573 [#uses=1]
	store ubyte %2573, ubyte* %1627
	seteq ubyte %2572, 1		; <bool>:1094 [#uses=1]
	br bool %1094, label %1095, label %1094

; <label>:1095		; preds = %1093, %1094
	add uint %684, 4294967273		; <uint>:741 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %741		; <ubyte*>:1628 [#uses=1]
	load ubyte* %1628		; <ubyte>:2574 [#uses=1]
	seteq ubyte %2574, 0		; <bool>:1095 [#uses=1]
	br bool %1095, label %1097, label %1096

; <label>:1096		; preds = %1095, %1096
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %741		; <ubyte*>:1629 [#uses=2]
	load ubyte* %1629		; <ubyte>:2575 [#uses=1]
	add ubyte %2575, 255		; <ubyte>:2576 [#uses=1]
	store ubyte %2576, ubyte* %1629
	add uint %684, 4294967274		; <uint>:742 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %742		; <ubyte*>:1630 [#uses=2]
	load ubyte* %1630		; <ubyte>:2577 [#uses=1]
	add ubyte %2577, 1		; <ubyte>:2578 [#uses=1]
	store ubyte %2578, ubyte* %1630
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %740		; <ubyte*>:1631 [#uses=2]
	load ubyte* %1631		; <ubyte>:2579 [#uses=1]
	add ubyte %2579, 1		; <ubyte>:2580 [#uses=1]
	store ubyte %2580, ubyte* %1631
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %741		; <ubyte*>:1632 [#uses=1]
	load ubyte* %1632		; <ubyte>:2581 [#uses=1]
	seteq ubyte %2581, 0		; <bool>:1096 [#uses=1]
	br bool %1096, label %1097, label %1096

; <label>:1097		; preds = %1095, %1096
	add uint %684, 4294967274		; <uint>:743 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %743		; <ubyte*>:1633 [#uses=1]
	load ubyte* %1633		; <ubyte>:2582 [#uses=1]
	seteq ubyte %2582, 0		; <bool>:1097 [#uses=1]
	br bool %1097, label %1099, label %1098

; <label>:1098		; preds = %1097, %1098
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %741		; <ubyte*>:1634 [#uses=2]
	load ubyte* %1634		; <ubyte>:2583 [#uses=1]
	add ubyte %2583, 1		; <ubyte>:2584 [#uses=1]
	store ubyte %2584, ubyte* %1634
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %743		; <ubyte*>:1635 [#uses=2]
	load ubyte* %1635		; <ubyte>:2585 [#uses=2]
	add ubyte %2585, 255		; <ubyte>:2586 [#uses=1]
	store ubyte %2586, ubyte* %1635
	seteq ubyte %2585, 1		; <bool>:1098 [#uses=1]
	br bool %1098, label %1099, label %1098

; <label>:1099		; preds = %1097, %1098
	add uint %684, 96		; <uint>:744 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %744		; <ubyte*>:1636 [#uses=1]
	load ubyte* %1636		; <ubyte>:2587 [#uses=1]
	seteq ubyte %2587, 0		; <bool>:1099 [#uses=1]
	br bool %1099, label %1101, label %1100

; <label>:1100		; preds = %1099, %1100
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %744		; <ubyte*>:1637 [#uses=2]
	load ubyte* %1637		; <ubyte>:2588 [#uses=2]
	add ubyte %2588, 255		; <ubyte>:2589 [#uses=1]
	store ubyte %2589, ubyte* %1637
	seteq ubyte %2588, 1		; <bool>:1100 [#uses=1]
	br bool %1100, label %1101, label %1100

; <label>:1101		; preds = %1099, %1100
	add uint %684, 4294967279		; <uint>:745 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %745		; <ubyte*>:1638 [#uses=1]
	load ubyte* %1638		; <ubyte>:2590 [#uses=1]
	seteq ubyte %2590, 0		; <bool>:1101 [#uses=1]
	br bool %1101, label %1103, label %1102

; <label>:1102		; preds = %1101, %1102
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %745		; <ubyte*>:1639 [#uses=2]
	load ubyte* %1639		; <ubyte>:2591 [#uses=1]
	add ubyte %2591, 255		; <ubyte>:2592 [#uses=1]
	store ubyte %2592, ubyte* %1639
	add uint %684, 4294967280		; <uint>:746 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %746		; <ubyte*>:1640 [#uses=2]
	load ubyte* %1640		; <ubyte>:2593 [#uses=1]
	add ubyte %2593, 1		; <ubyte>:2594 [#uses=1]
	store ubyte %2594, ubyte* %1640
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %744		; <ubyte*>:1641 [#uses=2]
	load ubyte* %1641		; <ubyte>:2595 [#uses=1]
	add ubyte %2595, 1		; <ubyte>:2596 [#uses=1]
	store ubyte %2596, ubyte* %1641
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %745		; <ubyte*>:1642 [#uses=1]
	load ubyte* %1642		; <ubyte>:2597 [#uses=1]
	seteq ubyte %2597, 0		; <bool>:1102 [#uses=1]
	br bool %1102, label %1103, label %1102

; <label>:1103		; preds = %1101, %1102
	add uint %684, 4294967280		; <uint>:747 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %747		; <ubyte*>:1643 [#uses=1]
	load ubyte* %1643		; <ubyte>:2598 [#uses=1]
	seteq ubyte %2598, 0		; <bool>:1103 [#uses=1]
	br bool %1103, label %1105, label %1104

; <label>:1104		; preds = %1103, %1104
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %745		; <ubyte*>:1644 [#uses=2]
	load ubyte* %1644		; <ubyte>:2599 [#uses=1]
	add ubyte %2599, 1		; <ubyte>:2600 [#uses=1]
	store ubyte %2600, ubyte* %1644
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %747		; <ubyte*>:1645 [#uses=2]
	load ubyte* %1645		; <ubyte>:2601 [#uses=2]
	add ubyte %2601, 255		; <ubyte>:2602 [#uses=1]
	store ubyte %2602, ubyte* %1645
	seteq ubyte %2601, 1		; <bool>:1104 [#uses=1]
	br bool %1104, label %1105, label %1104

; <label>:1105		; preds = %1103, %1104
	add uint %684, 102		; <uint>:748 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %748		; <ubyte*>:1646 [#uses=1]
	load ubyte* %1646		; <ubyte>:2603 [#uses=1]
	seteq ubyte %2603, 0		; <bool>:1105 [#uses=1]
	br bool %1105, label %1107, label %1106

; <label>:1106		; preds = %1105, %1106
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %748		; <ubyte*>:1647 [#uses=2]
	load ubyte* %1647		; <ubyte>:2604 [#uses=2]
	add ubyte %2604, 255		; <ubyte>:2605 [#uses=1]
	store ubyte %2605, ubyte* %1647
	seteq ubyte %2604, 1		; <bool>:1106 [#uses=1]
	br bool %1106, label %1107, label %1106

; <label>:1107		; preds = %1105, %1106
	add uint %684, 4294967285		; <uint>:749 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %749		; <ubyte*>:1648 [#uses=1]
	load ubyte* %1648		; <ubyte>:2606 [#uses=1]
	seteq ubyte %2606, 0		; <bool>:1107 [#uses=1]
	br bool %1107, label %1109, label %1108

; <label>:1108		; preds = %1107, %1108
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %749		; <ubyte*>:1649 [#uses=2]
	load ubyte* %1649		; <ubyte>:2607 [#uses=1]
	add ubyte %2607, 255		; <ubyte>:2608 [#uses=1]
	store ubyte %2608, ubyte* %1649
	add uint %684, 4294967286		; <uint>:750 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %750		; <ubyte*>:1650 [#uses=2]
	load ubyte* %1650		; <ubyte>:2609 [#uses=1]
	add ubyte %2609, 1		; <ubyte>:2610 [#uses=1]
	store ubyte %2610, ubyte* %1650
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %748		; <ubyte*>:1651 [#uses=2]
	load ubyte* %1651		; <ubyte>:2611 [#uses=1]
	add ubyte %2611, 1		; <ubyte>:2612 [#uses=1]
	store ubyte %2612, ubyte* %1651
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %749		; <ubyte*>:1652 [#uses=1]
	load ubyte* %1652		; <ubyte>:2613 [#uses=1]
	seteq ubyte %2613, 0		; <bool>:1108 [#uses=1]
	br bool %1108, label %1109, label %1108

; <label>:1109		; preds = %1107, %1108
	add uint %684, 4294967286		; <uint>:751 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %751		; <ubyte*>:1653 [#uses=1]
	load ubyte* %1653		; <ubyte>:2614 [#uses=1]
	seteq ubyte %2614, 0		; <bool>:1109 [#uses=1]
	br bool %1109, label %1111, label %1110

; <label>:1110		; preds = %1109, %1110
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %749		; <ubyte*>:1654 [#uses=2]
	load ubyte* %1654		; <ubyte>:2615 [#uses=1]
	add ubyte %2615, 1		; <ubyte>:2616 [#uses=1]
	store ubyte %2616, ubyte* %1654
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %751		; <ubyte*>:1655 [#uses=2]
	load ubyte* %1655		; <ubyte>:2617 [#uses=2]
	add ubyte %2617, 255		; <ubyte>:2618 [#uses=1]
	store ubyte %2618, ubyte* %1655
	seteq ubyte %2617, 1		; <bool>:1110 [#uses=1]
	br bool %1110, label %1111, label %1110

; <label>:1111		; preds = %1109, %1110
	add uint %684, 106		; <uint>:752 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %752		; <ubyte*>:1656 [#uses=1]
	load ubyte* %1656		; <ubyte>:2619 [#uses=1]
	seteq ubyte %2619, 0		; <bool>:1111 [#uses=1]
	br bool %1111, label %1113, label %1112

; <label>:1112		; preds = %1111, %1112
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %752		; <ubyte*>:1657 [#uses=2]
	load ubyte* %1657		; <ubyte>:2620 [#uses=2]
	add ubyte %2620, 255		; <ubyte>:2621 [#uses=1]
	store ubyte %2621, ubyte* %1657
	seteq ubyte %2620, 1		; <bool>:1112 [#uses=1]
	br bool %1112, label %1113, label %1112

; <label>:1113		; preds = %1111, %1112
	add uint %684, 4		; <uint>:753 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %753		; <ubyte*>:1658 [#uses=1]
	load ubyte* %1658		; <ubyte>:2622 [#uses=1]
	seteq ubyte %2622, 0		; <bool>:1113 [#uses=1]
	br bool %1113, label %1115, label %1114

; <label>:1114		; preds = %1113, %1114
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %753		; <ubyte*>:1659 [#uses=2]
	load ubyte* %1659		; <ubyte>:2623 [#uses=1]
	add ubyte %2623, 255		; <ubyte>:2624 [#uses=1]
	store ubyte %2624, ubyte* %1659
	add uint %684, 5		; <uint>:754 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %754		; <ubyte*>:1660 [#uses=2]
	load ubyte* %1660		; <ubyte>:2625 [#uses=1]
	add ubyte %2625, 1		; <ubyte>:2626 [#uses=1]
	store ubyte %2626, ubyte* %1660
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %752		; <ubyte*>:1661 [#uses=2]
	load ubyte* %1661		; <ubyte>:2627 [#uses=1]
	add ubyte %2627, 1		; <ubyte>:2628 [#uses=1]
	store ubyte %2628, ubyte* %1661
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %753		; <ubyte*>:1662 [#uses=1]
	load ubyte* %1662		; <ubyte>:2629 [#uses=1]
	seteq ubyte %2629, 0		; <bool>:1114 [#uses=1]
	br bool %1114, label %1115, label %1114

; <label>:1115		; preds = %1113, %1114
	add uint %684, 5		; <uint>:755 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %755		; <ubyte*>:1663 [#uses=1]
	load ubyte* %1663		; <ubyte>:2630 [#uses=1]
	seteq ubyte %2630, 0		; <bool>:1115 [#uses=1]
	br bool %1115, label %1117, label %1116

; <label>:1116		; preds = %1115, %1116
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %753		; <ubyte*>:1664 [#uses=2]
	load ubyte* %1664		; <ubyte>:2631 [#uses=1]
	add ubyte %2631, 1		; <ubyte>:2632 [#uses=1]
	store ubyte %2632, ubyte* %1664
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %755		; <ubyte*>:1665 [#uses=2]
	load ubyte* %1665		; <ubyte>:2633 [#uses=2]
	add ubyte %2633, 255		; <ubyte>:2634 [#uses=1]
	store ubyte %2634, ubyte* %1665
	seteq ubyte %2633, 1		; <bool>:1116 [#uses=1]
	br bool %1116, label %1117, label %1116

; <label>:1117		; preds = %1115, %1116
	add uint %684, 20		; <uint>:756 [#uses=6]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %756		; <ubyte*>:1666 [#uses=1]
	load ubyte* %1666		; <ubyte>:2635 [#uses=1]
	seteq ubyte %2635, 0		; <bool>:1117 [#uses=1]
	br bool %1117, label %1119, label %1118

; <label>:1118		; preds = %1117, %1118
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %756		; <ubyte*>:1667 [#uses=2]
	load ubyte* %1667		; <ubyte>:2636 [#uses=2]
	add ubyte %2636, 255		; <ubyte>:2637 [#uses=1]
	store ubyte %2637, ubyte* %1667
	seteq ubyte %2636, 1		; <bool>:1118 [#uses=1]
	br bool %1118, label %1119, label %1118

; <label>:1119		; preds = %1117, %1118
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %752		; <ubyte*>:1668 [#uses=1]
	load ubyte* %1668		; <ubyte>:2638 [#uses=1]
	seteq ubyte %2638, 0		; <bool>:1119 [#uses=1]
	br bool %1119, label %1121, label %1120

; <label>:1120		; preds = %1119, %1120
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %756		; <ubyte*>:1669 [#uses=2]
	load ubyte* %1669		; <ubyte>:2639 [#uses=1]
	add ubyte %2639, 1		; <ubyte>:2640 [#uses=1]
	store ubyte %2640, ubyte* %1669
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %752		; <ubyte*>:1670 [#uses=2]
	load ubyte* %1670		; <ubyte>:2641 [#uses=2]
	add ubyte %2641, 255		; <ubyte>:2642 [#uses=1]
	store ubyte %2642, ubyte* %1670
	seteq ubyte %2641, 1		; <bool>:1120 [#uses=1]
	br bool %1120, label %1121, label %1120

; <label>:1121		; preds = %1119, %1120
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %756		; <ubyte*>:1671 [#uses=1]
	load ubyte* %1671		; <ubyte>:2643 [#uses=1]
	seteq ubyte %2643, 0		; <bool>:1121 [#uses=1]
	br bool %1121, label %1123, label %1122

; <label>:1122		; preds = %1121, %1125
	phi uint [ %756, %1121 ], [ %761, %1125 ]		; <uint>:757 [#uses=6]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %757		; <ubyte*>:1672 [#uses=1]
	load ubyte* %1672		; <ubyte>:2644 [#uses=1]
	seteq ubyte %2644, 0		; <bool>:1122 [#uses=1]
	br bool %1122, label %1125, label %1124

; <label>:1123		; preds = %1121, %1125
	phi uint [ %756, %1121 ], [ %761, %1125 ]		; <uint>:758 [#uses=7]
	add uint %758, 4294967292		; <uint>:759 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %759		; <ubyte*>:1673 [#uses=1]
	load ubyte* %1673		; <ubyte>:2645 [#uses=1]
	seteq ubyte %2645, 0		; <bool>:1123 [#uses=1]
	br bool %1123, label %1127, label %1126

; <label>:1124		; preds = %1122, %1124
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %757		; <ubyte*>:1674 [#uses=2]
	load ubyte* %1674		; <ubyte>:2646 [#uses=1]
	add ubyte %2646, 255		; <ubyte>:2647 [#uses=1]
	store ubyte %2647, ubyte* %1674
	add uint %757, 6		; <uint>:760 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %760		; <ubyte*>:1675 [#uses=2]
	load ubyte* %1675		; <ubyte>:2648 [#uses=1]
	add ubyte %2648, 1		; <ubyte>:2649 [#uses=1]
	store ubyte %2649, ubyte* %1675
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %757		; <ubyte*>:1676 [#uses=1]
	load ubyte* %1676		; <ubyte>:2650 [#uses=1]
	seteq ubyte %2650, 0		; <bool>:1124 [#uses=1]
	br bool %1124, label %1125, label %1124

; <label>:1125		; preds = %1122, %1124
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %757		; <ubyte*>:1677 [#uses=2]
	load ubyte* %1677		; <ubyte>:2651 [#uses=1]
	add ubyte %2651, 1		; <ubyte>:2652 [#uses=1]
	store ubyte %2652, ubyte* %1677
	add uint %757, 6		; <uint>:761 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %761		; <ubyte*>:1678 [#uses=2]
	load ubyte* %1678		; <ubyte>:2653 [#uses=2]
	add ubyte %2653, 255		; <ubyte>:2654 [#uses=1]
	store ubyte %2654, ubyte* %1678
	seteq ubyte %2653, 1		; <bool>:1125 [#uses=1]
	br bool %1125, label %1123, label %1122

; <label>:1126		; preds = %1123, %1126
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %759		; <ubyte*>:1679 [#uses=2]
	load ubyte* %1679		; <ubyte>:2655 [#uses=2]
	add ubyte %2655, 255		; <ubyte>:2656 [#uses=1]
	store ubyte %2656, ubyte* %1679
	seteq ubyte %2655, 1		; <bool>:1126 [#uses=1]
	br bool %1126, label %1127, label %1126

; <label>:1127		; preds = %1123, %1126
	add uint %758, 4294967294		; <uint>:762 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %762		; <ubyte*>:1680 [#uses=1]
	load ubyte* %1680		; <ubyte>:2657 [#uses=1]
	seteq ubyte %2657, 0		; <bool>:1127 [#uses=1]
	br bool %1127, label %1129, label %1128

; <label>:1128		; preds = %1127, %1128
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %759		; <ubyte*>:1681 [#uses=2]
	load ubyte* %1681		; <ubyte>:2658 [#uses=1]
	add ubyte %2658, 1		; <ubyte>:2659 [#uses=1]
	store ubyte %2659, ubyte* %1681
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %762		; <ubyte*>:1682 [#uses=2]
	load ubyte* %1682		; <ubyte>:2660 [#uses=1]
	add ubyte %2660, 255		; <ubyte>:2661 [#uses=1]
	store ubyte %2661, ubyte* %1682
	add uint %758, 4294967295		; <uint>:763 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %763		; <ubyte*>:1683 [#uses=2]
	load ubyte* %1683		; <ubyte>:2662 [#uses=1]
	add ubyte %2662, 1		; <ubyte>:2663 [#uses=1]
	store ubyte %2663, ubyte* %1683
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %762		; <ubyte*>:1684 [#uses=1]
	load ubyte* %1684		; <ubyte>:2664 [#uses=1]
	seteq ubyte %2664, 0		; <bool>:1128 [#uses=1]
	br bool %1128, label %1129, label %1128

; <label>:1129		; preds = %1127, %1128
	add uint %758, 4294967295		; <uint>:764 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %764		; <ubyte*>:1685 [#uses=1]
	load ubyte* %1685		; <ubyte>:2665 [#uses=1]
	seteq ubyte %2665, 0		; <bool>:1129 [#uses=1]
	br bool %1129, label %1131, label %1130

; <label>:1130		; preds = %1129, %1130
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %762		; <ubyte*>:1686 [#uses=2]
	load ubyte* %1686		; <ubyte>:2666 [#uses=1]
	add ubyte %2666, 1		; <ubyte>:2667 [#uses=1]
	store ubyte %2667, ubyte* %1686
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %764		; <ubyte*>:1687 [#uses=2]
	load ubyte* %1687		; <ubyte>:2668 [#uses=2]
	add ubyte %2668, 255		; <ubyte>:2669 [#uses=1]
	store ubyte %2669, ubyte* %1687
	seteq ubyte %2668, 1		; <bool>:1130 [#uses=1]
	br bool %1130, label %1131, label %1130

; <label>:1131		; preds = %1129, %1130
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %758		; <ubyte*>:1688 [#uses=2]
	load ubyte* %1688		; <ubyte>:2670 [#uses=2]
	add ubyte %2670, 1		; <ubyte>:2671 [#uses=1]
	store ubyte %2671, ubyte* %1688
	seteq ubyte %2670, 255		; <bool>:1131 [#uses=1]
	br bool %1131, label %1133, label %1132

; <label>:1132		; preds = %1131, %1137
	phi uint [ %758, %1131 ], [ %770, %1137 ]		; <uint>:765 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %765		; <ubyte*>:1689 [#uses=2]
	load ubyte* %1689		; <ubyte>:2672 [#uses=1]
	add ubyte %2672, 255		; <ubyte>:2673 [#uses=1]
	store ubyte %2673, ubyte* %1689
	add uint %765, 4294967286		; <uint>:766 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %766		; <ubyte*>:1690 [#uses=1]
	load ubyte* %1690		; <ubyte>:2674 [#uses=1]
	seteq ubyte %2674, 0		; <bool>:1132 [#uses=1]
	br bool %1132, label %1135, label %1134

; <label>:1133		; preds = %1131, %1137
	phi uint [ %758, %1131 ], [ %770, %1137 ]		; <uint>:767 [#uses=67]
	add uint %767, 4		; <uint>:768 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %768		; <ubyte*>:1691 [#uses=1]
	load ubyte* %1691		; <ubyte>:2675 [#uses=1]
	seteq ubyte %2675, 0		; <bool>:1133 [#uses=1]
	br bool %1133, label %1139, label %1138

; <label>:1134		; preds = %1132, %1134
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %766		; <ubyte*>:1692 [#uses=2]
	load ubyte* %1692		; <ubyte>:2676 [#uses=2]
	add ubyte %2676, 255		; <ubyte>:2677 [#uses=1]
	store ubyte %2677, ubyte* %1692
	seteq ubyte %2676, 1		; <bool>:1134 [#uses=1]
	br bool %1134, label %1135, label %1134

; <label>:1135		; preds = %1132, %1134
	add uint %765, 4294967292		; <uint>:769 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %769		; <ubyte*>:1693 [#uses=1]
	load ubyte* %1693		; <ubyte>:2678 [#uses=1]
	seteq ubyte %2678, 0		; <bool>:1135 [#uses=1]
	br bool %1135, label %1137, label %1136

; <label>:1136		; preds = %1135, %1136
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %766		; <ubyte*>:1694 [#uses=2]
	load ubyte* %1694		; <ubyte>:2679 [#uses=1]
	add ubyte %2679, 1		; <ubyte>:2680 [#uses=1]
	store ubyte %2680, ubyte* %1694
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %769		; <ubyte*>:1695 [#uses=2]
	load ubyte* %1695		; <ubyte>:2681 [#uses=2]
	add ubyte %2681, 255		; <ubyte>:2682 [#uses=1]
	store ubyte %2682, ubyte* %1695
	seteq ubyte %2681, 1		; <bool>:1136 [#uses=1]
	br bool %1136, label %1137, label %1136

; <label>:1137		; preds = %1135, %1136
	add uint %765, 4294967290		; <uint>:770 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %770		; <ubyte*>:1696 [#uses=1]
	load ubyte* %1696		; <ubyte>:2683 [#uses=1]
	seteq ubyte %2683, 0		; <bool>:1137 [#uses=1]
	br bool %1137, label %1133, label %1132

; <label>:1138		; preds = %1133, %1138
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %768		; <ubyte*>:1697 [#uses=2]
	load ubyte* %1697		; <ubyte>:2684 [#uses=2]
	add ubyte %2684, 255		; <ubyte>:2685 [#uses=1]
	store ubyte %2685, ubyte* %1697
	seteq ubyte %2684, 1		; <bool>:1138 [#uses=1]
	br bool %1138, label %1139, label %1138

; <label>:1139		; preds = %1133, %1138
	add uint %767, 10		; <uint>:771 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %771		; <ubyte*>:1698 [#uses=1]
	load ubyte* %1698		; <ubyte>:2686 [#uses=1]
	seteq ubyte %2686, 0		; <bool>:1139 [#uses=1]
	br bool %1139, label %1141, label %1140

; <label>:1140		; preds = %1139, %1140
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %771		; <ubyte*>:1699 [#uses=2]
	load ubyte* %1699		; <ubyte>:2687 [#uses=2]
	add ubyte %2687, 255		; <ubyte>:2688 [#uses=1]
	store ubyte %2688, ubyte* %1699
	seteq ubyte %2687, 1		; <bool>:1140 [#uses=1]
	br bool %1140, label %1141, label %1140

; <label>:1141		; preds = %1139, %1140
	add uint %767, 16		; <uint>:772 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %772		; <ubyte*>:1700 [#uses=1]
	load ubyte* %1700		; <ubyte>:2689 [#uses=1]
	seteq ubyte %2689, 0		; <bool>:1141 [#uses=1]
	br bool %1141, label %1143, label %1142

; <label>:1142		; preds = %1141, %1142
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %772		; <ubyte*>:1701 [#uses=2]
	load ubyte* %1701		; <ubyte>:2690 [#uses=2]
	add ubyte %2690, 255		; <ubyte>:2691 [#uses=1]
	store ubyte %2691, ubyte* %1701
	seteq ubyte %2690, 1		; <bool>:1142 [#uses=1]
	br bool %1142, label %1143, label %1142

; <label>:1143		; preds = %1141, %1142
	add uint %767, 22		; <uint>:773 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %773		; <ubyte*>:1702 [#uses=1]
	load ubyte* %1702		; <ubyte>:2692 [#uses=1]
	seteq ubyte %2692, 0		; <bool>:1143 [#uses=1]
	br bool %1143, label %1145, label %1144

; <label>:1144		; preds = %1143, %1144
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %773		; <ubyte*>:1703 [#uses=2]
	load ubyte* %1703		; <ubyte>:2693 [#uses=2]
	add ubyte %2693, 255		; <ubyte>:2694 [#uses=1]
	store ubyte %2694, ubyte* %1703
	seteq ubyte %2693, 1		; <bool>:1144 [#uses=1]
	br bool %1144, label %1145, label %1144

; <label>:1145		; preds = %1143, %1144
	add uint %767, 28		; <uint>:774 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %774		; <ubyte*>:1704 [#uses=1]
	load ubyte* %1704		; <ubyte>:2695 [#uses=1]
	seteq ubyte %2695, 0		; <bool>:1145 [#uses=1]
	br bool %1145, label %1147, label %1146

; <label>:1146		; preds = %1145, %1146
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %774		; <ubyte*>:1705 [#uses=2]
	load ubyte* %1705		; <ubyte>:2696 [#uses=2]
	add ubyte %2696, 255		; <ubyte>:2697 [#uses=1]
	store ubyte %2697, ubyte* %1705
	seteq ubyte %2696, 1		; <bool>:1146 [#uses=1]
	br bool %1146, label %1147, label %1146

; <label>:1147		; preds = %1145, %1146
	add uint %767, 34		; <uint>:775 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %775		; <ubyte*>:1706 [#uses=1]
	load ubyte* %1706		; <ubyte>:2698 [#uses=1]
	seteq ubyte %2698, 0		; <bool>:1147 [#uses=1]
	br bool %1147, label %1149, label %1148

; <label>:1148		; preds = %1147, %1148
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %775		; <ubyte*>:1707 [#uses=2]
	load ubyte* %1707		; <ubyte>:2699 [#uses=2]
	add ubyte %2699, 255		; <ubyte>:2700 [#uses=1]
	store ubyte %2700, ubyte* %1707
	seteq ubyte %2699, 1		; <bool>:1148 [#uses=1]
	br bool %1148, label %1149, label %1148

; <label>:1149		; preds = %1147, %1148
	add uint %767, 40		; <uint>:776 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %776		; <ubyte*>:1708 [#uses=1]
	load ubyte* %1708		; <ubyte>:2701 [#uses=1]
	seteq ubyte %2701, 0		; <bool>:1149 [#uses=1]
	br bool %1149, label %1151, label %1150

; <label>:1150		; preds = %1149, %1150
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %776		; <ubyte*>:1709 [#uses=2]
	load ubyte* %1709		; <ubyte>:2702 [#uses=2]
	add ubyte %2702, 255		; <ubyte>:2703 [#uses=1]
	store ubyte %2703, ubyte* %1709
	seteq ubyte %2702, 1		; <bool>:1150 [#uses=1]
	br bool %1150, label %1151, label %1150

; <label>:1151		; preds = %1149, %1150
	add uint %767, 46		; <uint>:777 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %777		; <ubyte*>:1710 [#uses=1]
	load ubyte* %1710		; <ubyte>:2704 [#uses=1]
	seteq ubyte %2704, 0		; <bool>:1151 [#uses=1]
	br bool %1151, label %1153, label %1152

; <label>:1152		; preds = %1151, %1152
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %777		; <ubyte*>:1711 [#uses=2]
	load ubyte* %1711		; <ubyte>:2705 [#uses=2]
	add ubyte %2705, 255		; <ubyte>:2706 [#uses=1]
	store ubyte %2706, ubyte* %1711
	seteq ubyte %2705, 1		; <bool>:1152 [#uses=1]
	br bool %1152, label %1153, label %1152

; <label>:1153		; preds = %1151, %1152
	add uint %767, 52		; <uint>:778 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %778		; <ubyte*>:1712 [#uses=1]
	load ubyte* %1712		; <ubyte>:2707 [#uses=1]
	seteq ubyte %2707, 0		; <bool>:1153 [#uses=1]
	br bool %1153, label %1155, label %1154

; <label>:1154		; preds = %1153, %1154
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %778		; <ubyte*>:1713 [#uses=2]
	load ubyte* %1713		; <ubyte>:2708 [#uses=2]
	add ubyte %2708, 255		; <ubyte>:2709 [#uses=1]
	store ubyte %2709, ubyte* %1713
	seteq ubyte %2708, 1		; <bool>:1154 [#uses=1]
	br bool %1154, label %1155, label %1154

; <label>:1155		; preds = %1153, %1154
	add uint %767, 58		; <uint>:779 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %779		; <ubyte*>:1714 [#uses=1]
	load ubyte* %1714		; <ubyte>:2710 [#uses=1]
	seteq ubyte %2710, 0		; <bool>:1155 [#uses=1]
	br bool %1155, label %1157, label %1156

; <label>:1156		; preds = %1155, %1156
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %779		; <ubyte*>:1715 [#uses=2]
	load ubyte* %1715		; <ubyte>:2711 [#uses=2]
	add ubyte %2711, 255		; <ubyte>:2712 [#uses=1]
	store ubyte %2712, ubyte* %1715
	seteq ubyte %2711, 1		; <bool>:1156 [#uses=1]
	br bool %1156, label %1157, label %1156

; <label>:1157		; preds = %1155, %1156
	add uint %767, 64		; <uint>:780 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %780		; <ubyte*>:1716 [#uses=1]
	load ubyte* %1716		; <ubyte>:2713 [#uses=1]
	seteq ubyte %2713, 0		; <bool>:1157 [#uses=1]
	br bool %1157, label %1159, label %1158

; <label>:1158		; preds = %1157, %1158
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %780		; <ubyte*>:1717 [#uses=2]
	load ubyte* %1717		; <ubyte>:2714 [#uses=2]
	add ubyte %2714, 255		; <ubyte>:2715 [#uses=1]
	store ubyte %2715, ubyte* %1717
	seteq ubyte %2714, 1		; <bool>:1158 [#uses=1]
	br bool %1158, label %1159, label %1158

; <label>:1159		; preds = %1157, %1158
	add uint %767, 70		; <uint>:781 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %781		; <ubyte*>:1718 [#uses=1]
	load ubyte* %1718		; <ubyte>:2716 [#uses=1]
	seteq ubyte %2716, 0		; <bool>:1159 [#uses=1]
	br bool %1159, label %1161, label %1160

; <label>:1160		; preds = %1159, %1160
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %781		; <ubyte*>:1719 [#uses=2]
	load ubyte* %1719		; <ubyte>:2717 [#uses=2]
	add ubyte %2717, 255		; <ubyte>:2718 [#uses=1]
	store ubyte %2718, ubyte* %1719
	seteq ubyte %2717, 1		; <bool>:1160 [#uses=1]
	br bool %1160, label %1161, label %1160

; <label>:1161		; preds = %1159, %1160
	add uint %767, 76		; <uint>:782 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %782		; <ubyte*>:1720 [#uses=1]
	load ubyte* %1720		; <ubyte>:2719 [#uses=1]
	seteq ubyte %2719, 0		; <bool>:1161 [#uses=1]
	br bool %1161, label %1163, label %1162

; <label>:1162		; preds = %1161, %1162
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %782		; <ubyte*>:1721 [#uses=2]
	load ubyte* %1721		; <ubyte>:2720 [#uses=2]
	add ubyte %2720, 255		; <ubyte>:2721 [#uses=1]
	store ubyte %2721, ubyte* %1721
	seteq ubyte %2720, 1		; <bool>:1162 [#uses=1]
	br bool %1162, label %1163, label %1162

; <label>:1163		; preds = %1161, %1162
	add uint %767, 82		; <uint>:783 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %783		; <ubyte*>:1722 [#uses=1]
	load ubyte* %1722		; <ubyte>:2722 [#uses=1]
	seteq ubyte %2722, 0		; <bool>:1163 [#uses=1]
	br bool %1163, label %1165, label %1164

; <label>:1164		; preds = %1163, %1164
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %783		; <ubyte*>:1723 [#uses=2]
	load ubyte* %1723		; <ubyte>:2723 [#uses=2]
	add ubyte %2723, 255		; <ubyte>:2724 [#uses=1]
	store ubyte %2724, ubyte* %1723
	seteq ubyte %2723, 1		; <bool>:1164 [#uses=1]
	br bool %1164, label %1165, label %1164

; <label>:1165		; preds = %1163, %1164
	add uint %767, 88		; <uint>:784 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %784		; <ubyte*>:1724 [#uses=1]
	load ubyte* %1724		; <ubyte>:2725 [#uses=1]
	seteq ubyte %2725, 0		; <bool>:1165 [#uses=1]
	br bool %1165, label %1167, label %1166

; <label>:1166		; preds = %1165, %1166
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %784		; <ubyte*>:1725 [#uses=2]
	load ubyte* %1725		; <ubyte>:2726 [#uses=2]
	add ubyte %2726, 255		; <ubyte>:2727 [#uses=1]
	store ubyte %2727, ubyte* %1725
	seteq ubyte %2726, 1		; <bool>:1166 [#uses=1]
	br bool %1166, label %1167, label %1166

; <label>:1167		; preds = %1165, %1166
	add uint %767, 4294967290		; <uint>:785 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %785		; <ubyte*>:1726 [#uses=1]
	load ubyte* %1726		; <ubyte>:2728 [#uses=1]
	seteq ubyte %2728, 0		; <bool>:1167 [#uses=1]
	br bool %1167, label %1169, label %1168

; <label>:1168		; preds = %1167, %1168
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %785		; <ubyte*>:1727 [#uses=2]
	load ubyte* %1727		; <ubyte>:2729 [#uses=2]
	add ubyte %2729, 255		; <ubyte>:2730 [#uses=1]
	store ubyte %2730, ubyte* %1727
	seteq ubyte %2729, 1		; <bool>:1168 [#uses=1]
	br bool %1168, label %1169, label %1168

; <label>:1169		; preds = %1167, %1168
	add uint %767, 4294967292		; <uint>:786 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %786		; <ubyte*>:1728 [#uses=1]
	load ubyte* %1728		; <ubyte>:2731 [#uses=1]
	seteq ubyte %2731, 0		; <bool>:1169 [#uses=1]
	br bool %1169, label %1171, label %1170

; <label>:1170		; preds = %1169, %1170
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %785		; <ubyte*>:1729 [#uses=2]
	load ubyte* %1729		; <ubyte>:2732 [#uses=1]
	add ubyte %2732, 1		; <ubyte>:2733 [#uses=1]
	store ubyte %2733, ubyte* %1729
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %786		; <ubyte*>:1730 [#uses=2]
	load ubyte* %1730		; <ubyte>:2734 [#uses=2]
	add ubyte %2734, 255		; <ubyte>:2735 [#uses=1]
	store ubyte %2735, ubyte* %1730
	seteq ubyte %2734, 1		; <bool>:1170 [#uses=1]
	br bool %1170, label %1171, label %1170

; <label>:1171		; preds = %1169, %1170
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %768		; <ubyte*>:1731 [#uses=1]
	load ubyte* %1731		; <ubyte>:2736 [#uses=1]
	seteq ubyte %2736, 0		; <bool>:1171 [#uses=1]
	br bool %1171, label %1173, label %1172

; <label>:1172		; preds = %1171, %1172
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %768		; <ubyte*>:1732 [#uses=2]
	load ubyte* %1732		; <ubyte>:2737 [#uses=2]
	add ubyte %2737, 255		; <ubyte>:2738 [#uses=1]
	store ubyte %2738, ubyte* %1732
	seteq ubyte %2737, 1		; <bool>:1172 [#uses=1]
	br bool %1172, label %1173, label %1172

; <label>:1173		; preds = %1171, %1172
	add uint %767, 4294967187		; <uint>:787 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %787		; <ubyte*>:1733 [#uses=1]
	load ubyte* %1733		; <ubyte>:2739 [#uses=1]
	seteq ubyte %2739, 0		; <bool>:1173 [#uses=1]
	br bool %1173, label %1175, label %1174

; <label>:1174		; preds = %1173, %1174
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %787		; <ubyte*>:1734 [#uses=2]
	load ubyte* %1734		; <ubyte>:2740 [#uses=1]
	add ubyte %2740, 255		; <ubyte>:2741 [#uses=1]
	store ubyte %2741, ubyte* %1734
	add uint %767, 4294967188		; <uint>:788 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %788		; <ubyte*>:1735 [#uses=2]
	load ubyte* %1735		; <ubyte>:2742 [#uses=1]
	add ubyte %2742, 1		; <ubyte>:2743 [#uses=1]
	store ubyte %2743, ubyte* %1735
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %768		; <ubyte*>:1736 [#uses=2]
	load ubyte* %1736		; <ubyte>:2744 [#uses=1]
	add ubyte %2744, 1		; <ubyte>:2745 [#uses=1]
	store ubyte %2745, ubyte* %1736
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %787		; <ubyte*>:1737 [#uses=1]
	load ubyte* %1737		; <ubyte>:2746 [#uses=1]
	seteq ubyte %2746, 0		; <bool>:1174 [#uses=1]
	br bool %1174, label %1175, label %1174

; <label>:1175		; preds = %1173, %1174
	add uint %767, 4294967188		; <uint>:789 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %789		; <ubyte*>:1738 [#uses=1]
	load ubyte* %1738		; <ubyte>:2747 [#uses=1]
	seteq ubyte %2747, 0		; <bool>:1175 [#uses=1]
	br bool %1175, label %1177, label %1176

; <label>:1176		; preds = %1175, %1176
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %787		; <ubyte*>:1739 [#uses=2]
	load ubyte* %1739		; <ubyte>:2748 [#uses=1]
	add ubyte %2748, 1		; <ubyte>:2749 [#uses=1]
	store ubyte %2749, ubyte* %1739
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %789		; <ubyte*>:1740 [#uses=2]
	load ubyte* %1740		; <ubyte>:2750 [#uses=2]
	add ubyte %2750, 255		; <ubyte>:2751 [#uses=1]
	store ubyte %2751, ubyte* %1740
	seteq ubyte %2750, 1		; <bool>:1176 [#uses=1]
	br bool %1176, label %1177, label %1176

; <label>:1177		; preds = %1175, %1176
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %771		; <ubyte*>:1741 [#uses=1]
	load ubyte* %1741		; <ubyte>:2752 [#uses=1]
	seteq ubyte %2752, 0		; <bool>:1177 [#uses=1]
	br bool %1177, label %1179, label %1178

; <label>:1178		; preds = %1177, %1178
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %771		; <ubyte*>:1742 [#uses=2]
	load ubyte* %1742		; <ubyte>:2753 [#uses=2]
	add ubyte %2753, 255		; <ubyte>:2754 [#uses=1]
	store ubyte %2754, ubyte* %1742
	seteq ubyte %2753, 1		; <bool>:1178 [#uses=1]
	br bool %1178, label %1179, label %1178

; <label>:1179		; preds = %1177, %1178
	add uint %767, 4294967193		; <uint>:790 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %790		; <ubyte*>:1743 [#uses=1]
	load ubyte* %1743		; <ubyte>:2755 [#uses=1]
	seteq ubyte %2755, 0		; <bool>:1179 [#uses=1]
	br bool %1179, label %1181, label %1180

; <label>:1180		; preds = %1179, %1180
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %790		; <ubyte*>:1744 [#uses=2]
	load ubyte* %1744		; <ubyte>:2756 [#uses=1]
	add ubyte %2756, 255		; <ubyte>:2757 [#uses=1]
	store ubyte %2757, ubyte* %1744
	add uint %767, 4294967194		; <uint>:791 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %791		; <ubyte*>:1745 [#uses=2]
	load ubyte* %1745		; <ubyte>:2758 [#uses=1]
	add ubyte %2758, 1		; <ubyte>:2759 [#uses=1]
	store ubyte %2759, ubyte* %1745
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %771		; <ubyte*>:1746 [#uses=2]
	load ubyte* %1746		; <ubyte>:2760 [#uses=1]
	add ubyte %2760, 1		; <ubyte>:2761 [#uses=1]
	store ubyte %2761, ubyte* %1746
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %790		; <ubyte*>:1747 [#uses=1]
	load ubyte* %1747		; <ubyte>:2762 [#uses=1]
	seteq ubyte %2762, 0		; <bool>:1180 [#uses=1]
	br bool %1180, label %1181, label %1180

; <label>:1181		; preds = %1179, %1180
	add uint %767, 4294967194		; <uint>:792 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %792		; <ubyte*>:1748 [#uses=1]
	load ubyte* %1748		; <ubyte>:2763 [#uses=1]
	seteq ubyte %2763, 0		; <bool>:1181 [#uses=1]
	br bool %1181, label %1183, label %1182

; <label>:1182		; preds = %1181, %1182
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %790		; <ubyte*>:1749 [#uses=2]
	load ubyte* %1749		; <ubyte>:2764 [#uses=1]
	add ubyte %2764, 1		; <ubyte>:2765 [#uses=1]
	store ubyte %2765, ubyte* %1749
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %792		; <ubyte*>:1750 [#uses=2]
	load ubyte* %1750		; <ubyte>:2766 [#uses=2]
	add ubyte %2766, 255		; <ubyte>:2767 [#uses=1]
	store ubyte %2767, ubyte* %1750
	seteq ubyte %2766, 1		; <bool>:1182 [#uses=1]
	br bool %1182, label %1183, label %1182

; <label>:1183		; preds = %1181, %1182
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %772		; <ubyte*>:1751 [#uses=1]
	load ubyte* %1751		; <ubyte>:2768 [#uses=1]
	seteq ubyte %2768, 0		; <bool>:1183 [#uses=1]
	br bool %1183, label %1185, label %1184

; <label>:1184		; preds = %1183, %1184
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %772		; <ubyte*>:1752 [#uses=2]
	load ubyte* %1752		; <ubyte>:2769 [#uses=2]
	add ubyte %2769, 255		; <ubyte>:2770 [#uses=1]
	store ubyte %2770, ubyte* %1752
	seteq ubyte %2769, 1		; <bool>:1184 [#uses=1]
	br bool %1184, label %1185, label %1184

; <label>:1185		; preds = %1183, %1184
	add uint %767, 4294967199		; <uint>:793 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %793		; <ubyte*>:1753 [#uses=1]
	load ubyte* %1753		; <ubyte>:2771 [#uses=1]
	seteq ubyte %2771, 0		; <bool>:1185 [#uses=1]
	br bool %1185, label %1187, label %1186

; <label>:1186		; preds = %1185, %1186
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %793		; <ubyte*>:1754 [#uses=2]
	load ubyte* %1754		; <ubyte>:2772 [#uses=1]
	add ubyte %2772, 255		; <ubyte>:2773 [#uses=1]
	store ubyte %2773, ubyte* %1754
	add uint %767, 4294967200		; <uint>:794 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %794		; <ubyte*>:1755 [#uses=2]
	load ubyte* %1755		; <ubyte>:2774 [#uses=1]
	add ubyte %2774, 1		; <ubyte>:2775 [#uses=1]
	store ubyte %2775, ubyte* %1755
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %772		; <ubyte*>:1756 [#uses=2]
	load ubyte* %1756		; <ubyte>:2776 [#uses=1]
	add ubyte %2776, 1		; <ubyte>:2777 [#uses=1]
	store ubyte %2777, ubyte* %1756
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %793		; <ubyte*>:1757 [#uses=1]
	load ubyte* %1757		; <ubyte>:2778 [#uses=1]
	seteq ubyte %2778, 0		; <bool>:1186 [#uses=1]
	br bool %1186, label %1187, label %1186

; <label>:1187		; preds = %1185, %1186
	add uint %767, 4294967200		; <uint>:795 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %795		; <ubyte*>:1758 [#uses=1]
	load ubyte* %1758		; <ubyte>:2779 [#uses=1]
	seteq ubyte %2779, 0		; <bool>:1187 [#uses=1]
	br bool %1187, label %1189, label %1188

; <label>:1188		; preds = %1187, %1188
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %793		; <ubyte*>:1759 [#uses=2]
	load ubyte* %1759		; <ubyte>:2780 [#uses=1]
	add ubyte %2780, 1		; <ubyte>:2781 [#uses=1]
	store ubyte %2781, ubyte* %1759
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %795		; <ubyte*>:1760 [#uses=2]
	load ubyte* %1760		; <ubyte>:2782 [#uses=2]
	add ubyte %2782, 255		; <ubyte>:2783 [#uses=1]
	store ubyte %2783, ubyte* %1760
	seteq ubyte %2782, 1		; <bool>:1188 [#uses=1]
	br bool %1188, label %1189, label %1188

; <label>:1189		; preds = %1187, %1188
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %773		; <ubyte*>:1761 [#uses=1]
	load ubyte* %1761		; <ubyte>:2784 [#uses=1]
	seteq ubyte %2784, 0		; <bool>:1189 [#uses=1]
	br bool %1189, label %1191, label %1190

; <label>:1190		; preds = %1189, %1190
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %773		; <ubyte*>:1762 [#uses=2]
	load ubyte* %1762		; <ubyte>:2785 [#uses=2]
	add ubyte %2785, 255		; <ubyte>:2786 [#uses=1]
	store ubyte %2786, ubyte* %1762
	seteq ubyte %2785, 1		; <bool>:1190 [#uses=1]
	br bool %1190, label %1191, label %1190

; <label>:1191		; preds = %1189, %1190
	add uint %767, 4294967205		; <uint>:796 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %796		; <ubyte*>:1763 [#uses=1]
	load ubyte* %1763		; <ubyte>:2787 [#uses=1]
	seteq ubyte %2787, 0		; <bool>:1191 [#uses=1]
	br bool %1191, label %1193, label %1192

; <label>:1192		; preds = %1191, %1192
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %796		; <ubyte*>:1764 [#uses=2]
	load ubyte* %1764		; <ubyte>:2788 [#uses=1]
	add ubyte %2788, 255		; <ubyte>:2789 [#uses=1]
	store ubyte %2789, ubyte* %1764
	add uint %767, 4294967206		; <uint>:797 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %797		; <ubyte*>:1765 [#uses=2]
	load ubyte* %1765		; <ubyte>:2790 [#uses=1]
	add ubyte %2790, 1		; <ubyte>:2791 [#uses=1]
	store ubyte %2791, ubyte* %1765
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %773		; <ubyte*>:1766 [#uses=2]
	load ubyte* %1766		; <ubyte>:2792 [#uses=1]
	add ubyte %2792, 1		; <ubyte>:2793 [#uses=1]
	store ubyte %2793, ubyte* %1766
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %796		; <ubyte*>:1767 [#uses=1]
	load ubyte* %1767		; <ubyte>:2794 [#uses=1]
	seteq ubyte %2794, 0		; <bool>:1192 [#uses=1]
	br bool %1192, label %1193, label %1192

; <label>:1193		; preds = %1191, %1192
	add uint %767, 4294967206		; <uint>:798 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %798		; <ubyte*>:1768 [#uses=1]
	load ubyte* %1768		; <ubyte>:2795 [#uses=1]
	seteq ubyte %2795, 0		; <bool>:1193 [#uses=1]
	br bool %1193, label %1195, label %1194

; <label>:1194		; preds = %1193, %1194
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %796		; <ubyte*>:1769 [#uses=2]
	load ubyte* %1769		; <ubyte>:2796 [#uses=1]
	add ubyte %2796, 1		; <ubyte>:2797 [#uses=1]
	store ubyte %2797, ubyte* %1769
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %798		; <ubyte*>:1770 [#uses=2]
	load ubyte* %1770		; <ubyte>:2798 [#uses=2]
	add ubyte %2798, 255		; <ubyte>:2799 [#uses=1]
	store ubyte %2799, ubyte* %1770
	seteq ubyte %2798, 1		; <bool>:1194 [#uses=1]
	br bool %1194, label %1195, label %1194

; <label>:1195		; preds = %1193, %1194
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %774		; <ubyte*>:1771 [#uses=1]
	load ubyte* %1771		; <ubyte>:2800 [#uses=1]
	seteq ubyte %2800, 0		; <bool>:1195 [#uses=1]
	br bool %1195, label %1197, label %1196

; <label>:1196		; preds = %1195, %1196
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %774		; <ubyte*>:1772 [#uses=2]
	load ubyte* %1772		; <ubyte>:2801 [#uses=2]
	add ubyte %2801, 255		; <ubyte>:2802 [#uses=1]
	store ubyte %2802, ubyte* %1772
	seteq ubyte %2801, 1		; <bool>:1196 [#uses=1]
	br bool %1196, label %1197, label %1196

; <label>:1197		; preds = %1195, %1196
	add uint %767, 4294967211		; <uint>:799 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %799		; <ubyte*>:1773 [#uses=1]
	load ubyte* %1773		; <ubyte>:2803 [#uses=1]
	seteq ubyte %2803, 0		; <bool>:1197 [#uses=1]
	br bool %1197, label %1199, label %1198

; <label>:1198		; preds = %1197, %1198
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %799		; <ubyte*>:1774 [#uses=2]
	load ubyte* %1774		; <ubyte>:2804 [#uses=1]
	add ubyte %2804, 255		; <ubyte>:2805 [#uses=1]
	store ubyte %2805, ubyte* %1774
	add uint %767, 4294967212		; <uint>:800 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %800		; <ubyte*>:1775 [#uses=2]
	load ubyte* %1775		; <ubyte>:2806 [#uses=1]
	add ubyte %2806, 1		; <ubyte>:2807 [#uses=1]
	store ubyte %2807, ubyte* %1775
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %774		; <ubyte*>:1776 [#uses=2]
	load ubyte* %1776		; <ubyte>:2808 [#uses=1]
	add ubyte %2808, 1		; <ubyte>:2809 [#uses=1]
	store ubyte %2809, ubyte* %1776
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %799		; <ubyte*>:1777 [#uses=1]
	load ubyte* %1777		; <ubyte>:2810 [#uses=1]
	seteq ubyte %2810, 0		; <bool>:1198 [#uses=1]
	br bool %1198, label %1199, label %1198

; <label>:1199		; preds = %1197, %1198
	add uint %767, 4294967212		; <uint>:801 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %801		; <ubyte*>:1778 [#uses=1]
	load ubyte* %1778		; <ubyte>:2811 [#uses=1]
	seteq ubyte %2811, 0		; <bool>:1199 [#uses=1]
	br bool %1199, label %1201, label %1200

; <label>:1200		; preds = %1199, %1200
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %799		; <ubyte*>:1779 [#uses=2]
	load ubyte* %1779		; <ubyte>:2812 [#uses=1]
	add ubyte %2812, 1		; <ubyte>:2813 [#uses=1]
	store ubyte %2813, ubyte* %1779
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %801		; <ubyte*>:1780 [#uses=2]
	load ubyte* %1780		; <ubyte>:2814 [#uses=2]
	add ubyte %2814, 255		; <ubyte>:2815 [#uses=1]
	store ubyte %2815, ubyte* %1780
	seteq ubyte %2814, 1		; <bool>:1200 [#uses=1]
	br bool %1200, label %1201, label %1200

; <label>:1201		; preds = %1199, %1200
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %775		; <ubyte*>:1781 [#uses=1]
	load ubyte* %1781		; <ubyte>:2816 [#uses=1]
	seteq ubyte %2816, 0		; <bool>:1201 [#uses=1]
	br bool %1201, label %1203, label %1202

; <label>:1202		; preds = %1201, %1202
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %775		; <ubyte*>:1782 [#uses=2]
	load ubyte* %1782		; <ubyte>:2817 [#uses=2]
	add ubyte %2817, 255		; <ubyte>:2818 [#uses=1]
	store ubyte %2818, ubyte* %1782
	seteq ubyte %2817, 1		; <bool>:1202 [#uses=1]
	br bool %1202, label %1203, label %1202

; <label>:1203		; preds = %1201, %1202
	add uint %767, 4294967217		; <uint>:802 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %802		; <ubyte*>:1783 [#uses=1]
	load ubyte* %1783		; <ubyte>:2819 [#uses=1]
	seteq ubyte %2819, 0		; <bool>:1203 [#uses=1]
	br bool %1203, label %1205, label %1204

; <label>:1204		; preds = %1203, %1204
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %802		; <ubyte*>:1784 [#uses=2]
	load ubyte* %1784		; <ubyte>:2820 [#uses=1]
	add ubyte %2820, 255		; <ubyte>:2821 [#uses=1]
	store ubyte %2821, ubyte* %1784
	add uint %767, 4294967218		; <uint>:803 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %803		; <ubyte*>:1785 [#uses=2]
	load ubyte* %1785		; <ubyte>:2822 [#uses=1]
	add ubyte %2822, 1		; <ubyte>:2823 [#uses=1]
	store ubyte %2823, ubyte* %1785
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %775		; <ubyte*>:1786 [#uses=2]
	load ubyte* %1786		; <ubyte>:2824 [#uses=1]
	add ubyte %2824, 1		; <ubyte>:2825 [#uses=1]
	store ubyte %2825, ubyte* %1786
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %802		; <ubyte*>:1787 [#uses=1]
	load ubyte* %1787		; <ubyte>:2826 [#uses=1]
	seteq ubyte %2826, 0		; <bool>:1204 [#uses=1]
	br bool %1204, label %1205, label %1204

; <label>:1205		; preds = %1203, %1204
	add uint %767, 4294967218		; <uint>:804 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %804		; <ubyte*>:1788 [#uses=1]
	load ubyte* %1788		; <ubyte>:2827 [#uses=1]
	seteq ubyte %2827, 0		; <bool>:1205 [#uses=1]
	br bool %1205, label %1207, label %1206

; <label>:1206		; preds = %1205, %1206
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %802		; <ubyte*>:1789 [#uses=2]
	load ubyte* %1789		; <ubyte>:2828 [#uses=1]
	add ubyte %2828, 1		; <ubyte>:2829 [#uses=1]
	store ubyte %2829, ubyte* %1789
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %804		; <ubyte*>:1790 [#uses=2]
	load ubyte* %1790		; <ubyte>:2830 [#uses=2]
	add ubyte %2830, 255		; <ubyte>:2831 [#uses=1]
	store ubyte %2831, ubyte* %1790
	seteq ubyte %2830, 1		; <bool>:1206 [#uses=1]
	br bool %1206, label %1207, label %1206

; <label>:1207		; preds = %1205, %1206
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %776		; <ubyte*>:1791 [#uses=1]
	load ubyte* %1791		; <ubyte>:2832 [#uses=1]
	seteq ubyte %2832, 0		; <bool>:1207 [#uses=1]
	br bool %1207, label %1209, label %1208

; <label>:1208		; preds = %1207, %1208
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %776		; <ubyte*>:1792 [#uses=2]
	load ubyte* %1792		; <ubyte>:2833 [#uses=2]
	add ubyte %2833, 255		; <ubyte>:2834 [#uses=1]
	store ubyte %2834, ubyte* %1792
	seteq ubyte %2833, 1		; <bool>:1208 [#uses=1]
	br bool %1208, label %1209, label %1208

; <label>:1209		; preds = %1207, %1208
	add uint %767, 4294967223		; <uint>:805 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %805		; <ubyte*>:1793 [#uses=1]
	load ubyte* %1793		; <ubyte>:2835 [#uses=1]
	seteq ubyte %2835, 0		; <bool>:1209 [#uses=1]
	br bool %1209, label %1211, label %1210

; <label>:1210		; preds = %1209, %1210
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %805		; <ubyte*>:1794 [#uses=2]
	load ubyte* %1794		; <ubyte>:2836 [#uses=1]
	add ubyte %2836, 255		; <ubyte>:2837 [#uses=1]
	store ubyte %2837, ubyte* %1794
	add uint %767, 4294967224		; <uint>:806 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %806		; <ubyte*>:1795 [#uses=2]
	load ubyte* %1795		; <ubyte>:2838 [#uses=1]
	add ubyte %2838, 1		; <ubyte>:2839 [#uses=1]
	store ubyte %2839, ubyte* %1795
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %776		; <ubyte*>:1796 [#uses=2]
	load ubyte* %1796		; <ubyte>:2840 [#uses=1]
	add ubyte %2840, 1		; <ubyte>:2841 [#uses=1]
	store ubyte %2841, ubyte* %1796
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %805		; <ubyte*>:1797 [#uses=1]
	load ubyte* %1797		; <ubyte>:2842 [#uses=1]
	seteq ubyte %2842, 0		; <bool>:1210 [#uses=1]
	br bool %1210, label %1211, label %1210

; <label>:1211		; preds = %1209, %1210
	add uint %767, 4294967224		; <uint>:807 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %807		; <ubyte*>:1798 [#uses=1]
	load ubyte* %1798		; <ubyte>:2843 [#uses=1]
	seteq ubyte %2843, 0		; <bool>:1211 [#uses=1]
	br bool %1211, label %1213, label %1212

; <label>:1212		; preds = %1211, %1212
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %805		; <ubyte*>:1799 [#uses=2]
	load ubyte* %1799		; <ubyte>:2844 [#uses=1]
	add ubyte %2844, 1		; <ubyte>:2845 [#uses=1]
	store ubyte %2845, ubyte* %1799
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %807		; <ubyte*>:1800 [#uses=2]
	load ubyte* %1800		; <ubyte>:2846 [#uses=2]
	add ubyte %2846, 255		; <ubyte>:2847 [#uses=1]
	store ubyte %2847, ubyte* %1800
	seteq ubyte %2846, 1		; <bool>:1212 [#uses=1]
	br bool %1212, label %1213, label %1212

; <label>:1213		; preds = %1211, %1212
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %777		; <ubyte*>:1801 [#uses=1]
	load ubyte* %1801		; <ubyte>:2848 [#uses=1]
	seteq ubyte %2848, 0		; <bool>:1213 [#uses=1]
	br bool %1213, label %1215, label %1214

; <label>:1214		; preds = %1213, %1214
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %777		; <ubyte*>:1802 [#uses=2]
	load ubyte* %1802		; <ubyte>:2849 [#uses=2]
	add ubyte %2849, 255		; <ubyte>:2850 [#uses=1]
	store ubyte %2850, ubyte* %1802
	seteq ubyte %2849, 1		; <bool>:1214 [#uses=1]
	br bool %1214, label %1215, label %1214

; <label>:1215		; preds = %1213, %1214
	add uint %767, 4294967229		; <uint>:808 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %808		; <ubyte*>:1803 [#uses=1]
	load ubyte* %1803		; <ubyte>:2851 [#uses=1]
	seteq ubyte %2851, 0		; <bool>:1215 [#uses=1]
	br bool %1215, label %1217, label %1216

; <label>:1216		; preds = %1215, %1216
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %808		; <ubyte*>:1804 [#uses=2]
	load ubyte* %1804		; <ubyte>:2852 [#uses=1]
	add ubyte %2852, 255		; <ubyte>:2853 [#uses=1]
	store ubyte %2853, ubyte* %1804
	add uint %767, 4294967230		; <uint>:809 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %809		; <ubyte*>:1805 [#uses=2]
	load ubyte* %1805		; <ubyte>:2854 [#uses=1]
	add ubyte %2854, 1		; <ubyte>:2855 [#uses=1]
	store ubyte %2855, ubyte* %1805
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %777		; <ubyte*>:1806 [#uses=2]
	load ubyte* %1806		; <ubyte>:2856 [#uses=1]
	add ubyte %2856, 1		; <ubyte>:2857 [#uses=1]
	store ubyte %2857, ubyte* %1806
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %808		; <ubyte*>:1807 [#uses=1]
	load ubyte* %1807		; <ubyte>:2858 [#uses=1]
	seteq ubyte %2858, 0		; <bool>:1216 [#uses=1]
	br bool %1216, label %1217, label %1216

; <label>:1217		; preds = %1215, %1216
	add uint %767, 4294967230		; <uint>:810 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %810		; <ubyte*>:1808 [#uses=1]
	load ubyte* %1808		; <ubyte>:2859 [#uses=1]
	seteq ubyte %2859, 0		; <bool>:1217 [#uses=1]
	br bool %1217, label %1219, label %1218

; <label>:1218		; preds = %1217, %1218
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %808		; <ubyte*>:1809 [#uses=2]
	load ubyte* %1809		; <ubyte>:2860 [#uses=1]
	add ubyte %2860, 1		; <ubyte>:2861 [#uses=1]
	store ubyte %2861, ubyte* %1809
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %810		; <ubyte*>:1810 [#uses=2]
	load ubyte* %1810		; <ubyte>:2862 [#uses=2]
	add ubyte %2862, 255		; <ubyte>:2863 [#uses=1]
	store ubyte %2863, ubyte* %1810
	seteq ubyte %2862, 1		; <bool>:1218 [#uses=1]
	br bool %1218, label %1219, label %1218

; <label>:1219		; preds = %1217, %1218
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %778		; <ubyte*>:1811 [#uses=1]
	load ubyte* %1811		; <ubyte>:2864 [#uses=1]
	seteq ubyte %2864, 0		; <bool>:1219 [#uses=1]
	br bool %1219, label %1221, label %1220

; <label>:1220		; preds = %1219, %1220
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %778		; <ubyte*>:1812 [#uses=2]
	load ubyte* %1812		; <ubyte>:2865 [#uses=2]
	add ubyte %2865, 255		; <ubyte>:2866 [#uses=1]
	store ubyte %2866, ubyte* %1812
	seteq ubyte %2865, 1		; <bool>:1220 [#uses=1]
	br bool %1220, label %1221, label %1220

; <label>:1221		; preds = %1219, %1220
	add uint %767, 4294967235		; <uint>:811 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %811		; <ubyte*>:1813 [#uses=1]
	load ubyte* %1813		; <ubyte>:2867 [#uses=1]
	seteq ubyte %2867, 0		; <bool>:1221 [#uses=1]
	br bool %1221, label %1223, label %1222

; <label>:1222		; preds = %1221, %1222
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %811		; <ubyte*>:1814 [#uses=2]
	load ubyte* %1814		; <ubyte>:2868 [#uses=1]
	add ubyte %2868, 255		; <ubyte>:2869 [#uses=1]
	store ubyte %2869, ubyte* %1814
	add uint %767, 4294967236		; <uint>:812 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %812		; <ubyte*>:1815 [#uses=2]
	load ubyte* %1815		; <ubyte>:2870 [#uses=1]
	add ubyte %2870, 1		; <ubyte>:2871 [#uses=1]
	store ubyte %2871, ubyte* %1815
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %778		; <ubyte*>:1816 [#uses=2]
	load ubyte* %1816		; <ubyte>:2872 [#uses=1]
	add ubyte %2872, 1		; <ubyte>:2873 [#uses=1]
	store ubyte %2873, ubyte* %1816
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %811		; <ubyte*>:1817 [#uses=1]
	load ubyte* %1817		; <ubyte>:2874 [#uses=1]
	seteq ubyte %2874, 0		; <bool>:1222 [#uses=1]
	br bool %1222, label %1223, label %1222

; <label>:1223		; preds = %1221, %1222
	add uint %767, 4294967236		; <uint>:813 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %813		; <ubyte*>:1818 [#uses=1]
	load ubyte* %1818		; <ubyte>:2875 [#uses=1]
	seteq ubyte %2875, 0		; <bool>:1223 [#uses=1]
	br bool %1223, label %1225, label %1224

; <label>:1224		; preds = %1223, %1224
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %811		; <ubyte*>:1819 [#uses=2]
	load ubyte* %1819		; <ubyte>:2876 [#uses=1]
	add ubyte %2876, 1		; <ubyte>:2877 [#uses=1]
	store ubyte %2877, ubyte* %1819
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %813		; <ubyte*>:1820 [#uses=2]
	load ubyte* %1820		; <ubyte>:2878 [#uses=2]
	add ubyte %2878, 255		; <ubyte>:2879 [#uses=1]
	store ubyte %2879, ubyte* %1820
	seteq ubyte %2878, 1		; <bool>:1224 [#uses=1]
	br bool %1224, label %1225, label %1224

; <label>:1225		; preds = %1223, %1224
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %779		; <ubyte*>:1821 [#uses=1]
	load ubyte* %1821		; <ubyte>:2880 [#uses=1]
	seteq ubyte %2880, 0		; <bool>:1225 [#uses=1]
	br bool %1225, label %1227, label %1226

; <label>:1226		; preds = %1225, %1226
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %779		; <ubyte*>:1822 [#uses=2]
	load ubyte* %1822		; <ubyte>:2881 [#uses=2]
	add ubyte %2881, 255		; <ubyte>:2882 [#uses=1]
	store ubyte %2882, ubyte* %1822
	seteq ubyte %2881, 1		; <bool>:1226 [#uses=1]
	br bool %1226, label %1227, label %1226

; <label>:1227		; preds = %1225, %1226
	add uint %767, 4294967241		; <uint>:814 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %814		; <ubyte*>:1823 [#uses=1]
	load ubyte* %1823		; <ubyte>:2883 [#uses=1]
	seteq ubyte %2883, 0		; <bool>:1227 [#uses=1]
	br bool %1227, label %1229, label %1228

; <label>:1228		; preds = %1227, %1228
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %814		; <ubyte*>:1824 [#uses=2]
	load ubyte* %1824		; <ubyte>:2884 [#uses=1]
	add ubyte %2884, 255		; <ubyte>:2885 [#uses=1]
	store ubyte %2885, ubyte* %1824
	add uint %767, 4294967242		; <uint>:815 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %815		; <ubyte*>:1825 [#uses=2]
	load ubyte* %1825		; <ubyte>:2886 [#uses=1]
	add ubyte %2886, 1		; <ubyte>:2887 [#uses=1]
	store ubyte %2887, ubyte* %1825
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %779		; <ubyte*>:1826 [#uses=2]
	load ubyte* %1826		; <ubyte>:2888 [#uses=1]
	add ubyte %2888, 1		; <ubyte>:2889 [#uses=1]
	store ubyte %2889, ubyte* %1826
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %814		; <ubyte*>:1827 [#uses=1]
	load ubyte* %1827		; <ubyte>:2890 [#uses=1]
	seteq ubyte %2890, 0		; <bool>:1228 [#uses=1]
	br bool %1228, label %1229, label %1228

; <label>:1229		; preds = %1227, %1228
	add uint %767, 4294967242		; <uint>:816 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %816		; <ubyte*>:1828 [#uses=1]
	load ubyte* %1828		; <ubyte>:2891 [#uses=1]
	seteq ubyte %2891, 0		; <bool>:1229 [#uses=1]
	br bool %1229, label %1231, label %1230

; <label>:1230		; preds = %1229, %1230
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %814		; <ubyte*>:1829 [#uses=2]
	load ubyte* %1829		; <ubyte>:2892 [#uses=1]
	add ubyte %2892, 1		; <ubyte>:2893 [#uses=1]
	store ubyte %2893, ubyte* %1829
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %816		; <ubyte*>:1830 [#uses=2]
	load ubyte* %1830		; <ubyte>:2894 [#uses=2]
	add ubyte %2894, 255		; <ubyte>:2895 [#uses=1]
	store ubyte %2895, ubyte* %1830
	seteq ubyte %2894, 1		; <bool>:1230 [#uses=1]
	br bool %1230, label %1231, label %1230

; <label>:1231		; preds = %1229, %1230
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %780		; <ubyte*>:1831 [#uses=1]
	load ubyte* %1831		; <ubyte>:2896 [#uses=1]
	seteq ubyte %2896, 0		; <bool>:1231 [#uses=1]
	br bool %1231, label %1233, label %1232

; <label>:1232		; preds = %1231, %1232
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %780		; <ubyte*>:1832 [#uses=2]
	load ubyte* %1832		; <ubyte>:2897 [#uses=2]
	add ubyte %2897, 255		; <ubyte>:2898 [#uses=1]
	store ubyte %2898, ubyte* %1832
	seteq ubyte %2897, 1		; <bool>:1232 [#uses=1]
	br bool %1232, label %1233, label %1232

; <label>:1233		; preds = %1231, %1232
	add uint %767, 4294967247		; <uint>:817 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %817		; <ubyte*>:1833 [#uses=1]
	load ubyte* %1833		; <ubyte>:2899 [#uses=1]
	seteq ubyte %2899, 0		; <bool>:1233 [#uses=1]
	br bool %1233, label %1235, label %1234

; <label>:1234		; preds = %1233, %1234
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %817		; <ubyte*>:1834 [#uses=2]
	load ubyte* %1834		; <ubyte>:2900 [#uses=1]
	add ubyte %2900, 255		; <ubyte>:2901 [#uses=1]
	store ubyte %2901, ubyte* %1834
	add uint %767, 4294967248		; <uint>:818 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %818		; <ubyte*>:1835 [#uses=2]
	load ubyte* %1835		; <ubyte>:2902 [#uses=1]
	add ubyte %2902, 1		; <ubyte>:2903 [#uses=1]
	store ubyte %2903, ubyte* %1835
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %780		; <ubyte*>:1836 [#uses=2]
	load ubyte* %1836		; <ubyte>:2904 [#uses=1]
	add ubyte %2904, 1		; <ubyte>:2905 [#uses=1]
	store ubyte %2905, ubyte* %1836
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %817		; <ubyte*>:1837 [#uses=1]
	load ubyte* %1837		; <ubyte>:2906 [#uses=1]
	seteq ubyte %2906, 0		; <bool>:1234 [#uses=1]
	br bool %1234, label %1235, label %1234

; <label>:1235		; preds = %1233, %1234
	add uint %767, 4294967248		; <uint>:819 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %819		; <ubyte*>:1838 [#uses=1]
	load ubyte* %1838		; <ubyte>:2907 [#uses=1]
	seteq ubyte %2907, 0		; <bool>:1235 [#uses=1]
	br bool %1235, label %1237, label %1236

; <label>:1236		; preds = %1235, %1236
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %817		; <ubyte*>:1839 [#uses=2]
	load ubyte* %1839		; <ubyte>:2908 [#uses=1]
	add ubyte %2908, 1		; <ubyte>:2909 [#uses=1]
	store ubyte %2909, ubyte* %1839
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %819		; <ubyte*>:1840 [#uses=2]
	load ubyte* %1840		; <ubyte>:2910 [#uses=2]
	add ubyte %2910, 255		; <ubyte>:2911 [#uses=1]
	store ubyte %2911, ubyte* %1840
	seteq ubyte %2910, 1		; <bool>:1236 [#uses=1]
	br bool %1236, label %1237, label %1236

; <label>:1237		; preds = %1235, %1236
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %781		; <ubyte*>:1841 [#uses=1]
	load ubyte* %1841		; <ubyte>:2912 [#uses=1]
	seteq ubyte %2912, 0		; <bool>:1237 [#uses=1]
	br bool %1237, label %1239, label %1238

; <label>:1238		; preds = %1237, %1238
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %781		; <ubyte*>:1842 [#uses=2]
	load ubyte* %1842		; <ubyte>:2913 [#uses=2]
	add ubyte %2913, 255		; <ubyte>:2914 [#uses=1]
	store ubyte %2914, ubyte* %1842
	seteq ubyte %2913, 1		; <bool>:1238 [#uses=1]
	br bool %1238, label %1239, label %1238

; <label>:1239		; preds = %1237, %1238
	add uint %767, 4294967253		; <uint>:820 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %820		; <ubyte*>:1843 [#uses=1]
	load ubyte* %1843		; <ubyte>:2915 [#uses=1]
	seteq ubyte %2915, 0		; <bool>:1239 [#uses=1]
	br bool %1239, label %1241, label %1240

; <label>:1240		; preds = %1239, %1240
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %820		; <ubyte*>:1844 [#uses=2]
	load ubyte* %1844		; <ubyte>:2916 [#uses=1]
	add ubyte %2916, 255		; <ubyte>:2917 [#uses=1]
	store ubyte %2917, ubyte* %1844
	add uint %767, 4294967254		; <uint>:821 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %821		; <ubyte*>:1845 [#uses=2]
	load ubyte* %1845		; <ubyte>:2918 [#uses=1]
	add ubyte %2918, 1		; <ubyte>:2919 [#uses=1]
	store ubyte %2919, ubyte* %1845
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %781		; <ubyte*>:1846 [#uses=2]
	load ubyte* %1846		; <ubyte>:2920 [#uses=1]
	add ubyte %2920, 1		; <ubyte>:2921 [#uses=1]
	store ubyte %2921, ubyte* %1846
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %820		; <ubyte*>:1847 [#uses=1]
	load ubyte* %1847		; <ubyte>:2922 [#uses=1]
	seteq ubyte %2922, 0		; <bool>:1240 [#uses=1]
	br bool %1240, label %1241, label %1240

; <label>:1241		; preds = %1239, %1240
	add uint %767, 4294967254		; <uint>:822 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %822		; <ubyte*>:1848 [#uses=1]
	load ubyte* %1848		; <ubyte>:2923 [#uses=1]
	seteq ubyte %2923, 0		; <bool>:1241 [#uses=1]
	br bool %1241, label %1243, label %1242

; <label>:1242		; preds = %1241, %1242
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %820		; <ubyte*>:1849 [#uses=2]
	load ubyte* %1849		; <ubyte>:2924 [#uses=1]
	add ubyte %2924, 1		; <ubyte>:2925 [#uses=1]
	store ubyte %2925, ubyte* %1849
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %822		; <ubyte*>:1850 [#uses=2]
	load ubyte* %1850		; <ubyte>:2926 [#uses=2]
	add ubyte %2926, 255		; <ubyte>:2927 [#uses=1]
	store ubyte %2927, ubyte* %1850
	seteq ubyte %2926, 1		; <bool>:1242 [#uses=1]
	br bool %1242, label %1243, label %1242

; <label>:1243		; preds = %1241, %1242
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %782		; <ubyte*>:1851 [#uses=1]
	load ubyte* %1851		; <ubyte>:2928 [#uses=1]
	seteq ubyte %2928, 0		; <bool>:1243 [#uses=1]
	br bool %1243, label %1245, label %1244

; <label>:1244		; preds = %1243, %1244
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %782		; <ubyte*>:1852 [#uses=2]
	load ubyte* %1852		; <ubyte>:2929 [#uses=2]
	add ubyte %2929, 255		; <ubyte>:2930 [#uses=1]
	store ubyte %2930, ubyte* %1852
	seteq ubyte %2929, 1		; <bool>:1244 [#uses=1]
	br bool %1244, label %1245, label %1244

; <label>:1245		; preds = %1243, %1244
	add uint %767, 4294967259		; <uint>:823 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %823		; <ubyte*>:1853 [#uses=1]
	load ubyte* %1853		; <ubyte>:2931 [#uses=1]
	seteq ubyte %2931, 0		; <bool>:1245 [#uses=1]
	br bool %1245, label %1247, label %1246

; <label>:1246		; preds = %1245, %1246
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %823		; <ubyte*>:1854 [#uses=2]
	load ubyte* %1854		; <ubyte>:2932 [#uses=1]
	add ubyte %2932, 255		; <ubyte>:2933 [#uses=1]
	store ubyte %2933, ubyte* %1854
	add uint %767, 4294967260		; <uint>:824 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %824		; <ubyte*>:1855 [#uses=2]
	load ubyte* %1855		; <ubyte>:2934 [#uses=1]
	add ubyte %2934, 1		; <ubyte>:2935 [#uses=1]
	store ubyte %2935, ubyte* %1855
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %782		; <ubyte*>:1856 [#uses=2]
	load ubyte* %1856		; <ubyte>:2936 [#uses=1]
	add ubyte %2936, 1		; <ubyte>:2937 [#uses=1]
	store ubyte %2937, ubyte* %1856
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %823		; <ubyte*>:1857 [#uses=1]
	load ubyte* %1857		; <ubyte>:2938 [#uses=1]
	seteq ubyte %2938, 0		; <bool>:1246 [#uses=1]
	br bool %1246, label %1247, label %1246

; <label>:1247		; preds = %1245, %1246
	add uint %767, 4294967260		; <uint>:825 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %825		; <ubyte*>:1858 [#uses=1]
	load ubyte* %1858		; <ubyte>:2939 [#uses=1]
	seteq ubyte %2939, 0		; <bool>:1247 [#uses=1]
	br bool %1247, label %1249, label %1248

; <label>:1248		; preds = %1247, %1248
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %823		; <ubyte*>:1859 [#uses=2]
	load ubyte* %1859		; <ubyte>:2940 [#uses=1]
	add ubyte %2940, 1		; <ubyte>:2941 [#uses=1]
	store ubyte %2941, ubyte* %1859
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %825		; <ubyte*>:1860 [#uses=2]
	load ubyte* %1860		; <ubyte>:2942 [#uses=2]
	add ubyte %2942, 255		; <ubyte>:2943 [#uses=1]
	store ubyte %2943, ubyte* %1860
	seteq ubyte %2942, 1		; <bool>:1248 [#uses=1]
	br bool %1248, label %1249, label %1248

; <label>:1249		; preds = %1247, %1248
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %783		; <ubyte*>:1861 [#uses=1]
	load ubyte* %1861		; <ubyte>:2944 [#uses=1]
	seteq ubyte %2944, 0		; <bool>:1249 [#uses=1]
	br bool %1249, label %1251, label %1250

; <label>:1250		; preds = %1249, %1250
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %783		; <ubyte*>:1862 [#uses=2]
	load ubyte* %1862		; <ubyte>:2945 [#uses=2]
	add ubyte %2945, 255		; <ubyte>:2946 [#uses=1]
	store ubyte %2946, ubyte* %1862
	seteq ubyte %2945, 1		; <bool>:1250 [#uses=1]
	br bool %1250, label %1251, label %1250

; <label>:1251		; preds = %1249, %1250
	add uint %767, 4294967265		; <uint>:826 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %826		; <ubyte*>:1863 [#uses=1]
	load ubyte* %1863		; <ubyte>:2947 [#uses=1]
	seteq ubyte %2947, 0		; <bool>:1251 [#uses=1]
	br bool %1251, label %1253, label %1252

; <label>:1252		; preds = %1251, %1252
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %826		; <ubyte*>:1864 [#uses=2]
	load ubyte* %1864		; <ubyte>:2948 [#uses=1]
	add ubyte %2948, 255		; <ubyte>:2949 [#uses=1]
	store ubyte %2949, ubyte* %1864
	add uint %767, 4294967266		; <uint>:827 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %827		; <ubyte*>:1865 [#uses=2]
	load ubyte* %1865		; <ubyte>:2950 [#uses=1]
	add ubyte %2950, 1		; <ubyte>:2951 [#uses=1]
	store ubyte %2951, ubyte* %1865
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %783		; <ubyte*>:1866 [#uses=2]
	load ubyte* %1866		; <ubyte>:2952 [#uses=1]
	add ubyte %2952, 1		; <ubyte>:2953 [#uses=1]
	store ubyte %2953, ubyte* %1866
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %826		; <ubyte*>:1867 [#uses=1]
	load ubyte* %1867		; <ubyte>:2954 [#uses=1]
	seteq ubyte %2954, 0		; <bool>:1252 [#uses=1]
	br bool %1252, label %1253, label %1252

; <label>:1253		; preds = %1251, %1252
	add uint %767, 4294967266		; <uint>:828 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %828		; <ubyte*>:1868 [#uses=1]
	load ubyte* %1868		; <ubyte>:2955 [#uses=1]
	seteq ubyte %2955, 0		; <bool>:1253 [#uses=1]
	br bool %1253, label %1255, label %1254

; <label>:1254		; preds = %1253, %1254
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %826		; <ubyte*>:1869 [#uses=2]
	load ubyte* %1869		; <ubyte>:2956 [#uses=1]
	add ubyte %2956, 1		; <ubyte>:2957 [#uses=1]
	store ubyte %2957, ubyte* %1869
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %828		; <ubyte*>:1870 [#uses=2]
	load ubyte* %1870		; <ubyte>:2958 [#uses=2]
	add ubyte %2958, 255		; <ubyte>:2959 [#uses=1]
	store ubyte %2959, ubyte* %1870
	seteq ubyte %2958, 1		; <bool>:1254 [#uses=1]
	br bool %1254, label %1255, label %1254

; <label>:1255		; preds = %1253, %1254
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %784		; <ubyte*>:1871 [#uses=1]
	load ubyte* %1871		; <ubyte>:2960 [#uses=1]
	seteq ubyte %2960, 0		; <bool>:1255 [#uses=1]
	br bool %1255, label %1257, label %1256

; <label>:1256		; preds = %1255, %1256
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %784		; <ubyte*>:1872 [#uses=2]
	load ubyte* %1872		; <ubyte>:2961 [#uses=2]
	add ubyte %2961, 255		; <ubyte>:2962 [#uses=1]
	store ubyte %2962, ubyte* %1872
	seteq ubyte %2961, 1		; <bool>:1256 [#uses=1]
	br bool %1256, label %1257, label %1256

; <label>:1257		; preds = %1255, %1256
	add uint %767, 4294967271		; <uint>:829 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %829		; <ubyte*>:1873 [#uses=1]
	load ubyte* %1873		; <ubyte>:2963 [#uses=1]
	seteq ubyte %2963, 0		; <bool>:1257 [#uses=1]
	br bool %1257, label %1259, label %1258

; <label>:1258		; preds = %1257, %1258
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %829		; <ubyte*>:1874 [#uses=2]
	load ubyte* %1874		; <ubyte>:2964 [#uses=1]
	add ubyte %2964, 255		; <ubyte>:2965 [#uses=1]
	store ubyte %2965, ubyte* %1874
	add uint %767, 4294967272		; <uint>:830 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %830		; <ubyte*>:1875 [#uses=2]
	load ubyte* %1875		; <ubyte>:2966 [#uses=1]
	add ubyte %2966, 1		; <ubyte>:2967 [#uses=1]
	store ubyte %2967, ubyte* %1875
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %784		; <ubyte*>:1876 [#uses=2]
	load ubyte* %1876		; <ubyte>:2968 [#uses=1]
	add ubyte %2968, 1		; <ubyte>:2969 [#uses=1]
	store ubyte %2969, ubyte* %1876
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %829		; <ubyte*>:1877 [#uses=1]
	load ubyte* %1877		; <ubyte>:2970 [#uses=1]
	seteq ubyte %2970, 0		; <bool>:1258 [#uses=1]
	br bool %1258, label %1259, label %1258

; <label>:1259		; preds = %1257, %1258
	add uint %767, 4294967272		; <uint>:831 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %831		; <ubyte*>:1878 [#uses=1]
	load ubyte* %1878		; <ubyte>:2971 [#uses=1]
	seteq ubyte %2971, 0		; <bool>:1259 [#uses=1]
	br bool %1259, label %1261, label %1260

; <label>:1260		; preds = %1259, %1260
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %829		; <ubyte*>:1879 [#uses=2]
	load ubyte* %1879		; <ubyte>:2972 [#uses=1]
	add ubyte %2972, 1		; <ubyte>:2973 [#uses=1]
	store ubyte %2973, ubyte* %1879
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %831		; <ubyte*>:1880 [#uses=2]
	load ubyte* %1880		; <ubyte>:2974 [#uses=2]
	add ubyte %2974, 255		; <ubyte>:2975 [#uses=1]
	store ubyte %2975, ubyte* %1880
	seteq ubyte %2974, 1		; <bool>:1260 [#uses=1]
	br bool %1260, label %1261, label %1260

; <label>:1261		; preds = %1259, %1260
	add uint %767, 92		; <uint>:832 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %832		; <ubyte*>:1881 [#uses=1]
	load ubyte* %1881		; <ubyte>:2976 [#uses=1]
	seteq ubyte %2976, 0		; <bool>:1261 [#uses=1]
	br bool %1261, label %1263, label %1262

; <label>:1262		; preds = %1261, %1262
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %832		; <ubyte*>:1882 [#uses=2]
	load ubyte* %1882		; <ubyte>:2977 [#uses=2]
	add ubyte %2977, 255		; <ubyte>:2978 [#uses=1]
	store ubyte %2978, ubyte* %1882
	seteq ubyte %2977, 1		; <bool>:1262 [#uses=1]
	br bool %1262, label %1263, label %1262

; <label>:1263		; preds = %1261, %1262
	add uint %767, 4294967288		; <uint>:833 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %833		; <ubyte*>:1883 [#uses=1]
	load ubyte* %1883		; <ubyte>:2979 [#uses=1]
	seteq ubyte %2979, 0		; <bool>:1263 [#uses=1]
	br bool %1263, label %1265, label %1264

; <label>:1264		; preds = %1263, %1264
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %833		; <ubyte*>:1884 [#uses=2]
	load ubyte* %1884		; <ubyte>:2980 [#uses=1]
	add ubyte %2980, 255		; <ubyte>:2981 [#uses=1]
	store ubyte %2981, ubyte* %1884
	add uint %767, 4294967289		; <uint>:834 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %834		; <ubyte*>:1885 [#uses=2]
	load ubyte* %1885		; <ubyte>:2982 [#uses=1]
	add ubyte %2982, 1		; <ubyte>:2983 [#uses=1]
	store ubyte %2983, ubyte* %1885
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %832		; <ubyte*>:1886 [#uses=2]
	load ubyte* %1886		; <ubyte>:2984 [#uses=1]
	add ubyte %2984, 1		; <ubyte>:2985 [#uses=1]
	store ubyte %2985, ubyte* %1886
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %833		; <ubyte*>:1887 [#uses=1]
	load ubyte* %1887		; <ubyte>:2986 [#uses=1]
	seteq ubyte %2986, 0		; <bool>:1264 [#uses=1]
	br bool %1264, label %1265, label %1264

; <label>:1265		; preds = %1263, %1264
	add uint %767, 4294967289		; <uint>:835 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %835		; <ubyte*>:1888 [#uses=1]
	load ubyte* %1888		; <ubyte>:2987 [#uses=1]
	seteq ubyte %2987, 0		; <bool>:1265 [#uses=1]
	br bool %1265, label %1267, label %1266

; <label>:1266		; preds = %1265, %1266
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %833		; <ubyte*>:1889 [#uses=2]
	load ubyte* %1889		; <ubyte>:2988 [#uses=1]
	add ubyte %2988, 1		; <ubyte>:2989 [#uses=1]
	store ubyte %2989, ubyte* %1889
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %835		; <ubyte*>:1890 [#uses=2]
	load ubyte* %1890		; <ubyte>:2990 [#uses=2]
	add ubyte %2990, 255		; <ubyte>:2991 [#uses=1]
	store ubyte %2991, ubyte* %1890
	seteq ubyte %2990, 1		; <bool>:1266 [#uses=1]
	br bool %1266, label %1267, label %1266

; <label>:1267		; preds = %1265, %1266
	add uint %767, 6		; <uint>:836 [#uses=6]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %836		; <ubyte*>:1891 [#uses=1]
	load ubyte* %1891		; <ubyte>:2992 [#uses=1]
	seteq ubyte %2992, 0		; <bool>:1267 [#uses=1]
	br bool %1267, label %1269, label %1268

; <label>:1268		; preds = %1267, %1268
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %836		; <ubyte*>:1892 [#uses=2]
	load ubyte* %1892		; <ubyte>:2993 [#uses=2]
	add ubyte %2993, 255		; <ubyte>:2994 [#uses=1]
	store ubyte %2994, ubyte* %1892
	seteq ubyte %2993, 1		; <bool>:1268 [#uses=1]
	br bool %1268, label %1269, label %1268

; <label>:1269		; preds = %1267, %1268
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %832		; <ubyte*>:1893 [#uses=1]
	load ubyte* %1893		; <ubyte>:2995 [#uses=1]
	seteq ubyte %2995, 0		; <bool>:1269 [#uses=1]
	br bool %1269, label %1271, label %1270

; <label>:1270		; preds = %1269, %1270
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %836		; <ubyte*>:1894 [#uses=2]
	load ubyte* %1894		; <ubyte>:2996 [#uses=1]
	add ubyte %2996, 1		; <ubyte>:2997 [#uses=1]
	store ubyte %2997, ubyte* %1894
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %832		; <ubyte*>:1895 [#uses=2]
	load ubyte* %1895		; <ubyte>:2998 [#uses=2]
	add ubyte %2998, 255		; <ubyte>:2999 [#uses=1]
	store ubyte %2999, ubyte* %1895
	seteq ubyte %2998, 1		; <bool>:1270 [#uses=1]
	br bool %1270, label %1271, label %1270

; <label>:1271		; preds = %1269, %1270
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %836		; <ubyte*>:1896 [#uses=1]
	load ubyte* %1896		; <ubyte>:3000 [#uses=1]
	seteq ubyte %3000, 0		; <bool>:1271 [#uses=1]
	br bool %1271, label %1273, label %1272

; <label>:1272		; preds = %1271, %1275
	phi uint [ %836, %1271 ], [ %841, %1275 ]		; <uint>:837 [#uses=6]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %837		; <ubyte*>:1897 [#uses=1]
	load ubyte* %1897		; <ubyte>:3001 [#uses=1]
	seteq ubyte %3001, 0		; <bool>:1272 [#uses=1]
	br bool %1272, label %1275, label %1274

; <label>:1273		; preds = %1271, %1275
	phi uint [ %836, %1271 ], [ %841, %1275 ]		; <uint>:838 [#uses=7]
	add uint %838, 4294967292		; <uint>:839 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %839		; <ubyte*>:1898 [#uses=1]
	load ubyte* %1898		; <ubyte>:3002 [#uses=1]
	seteq ubyte %3002, 0		; <bool>:1273 [#uses=1]
	br bool %1273, label %1277, label %1276

; <label>:1274		; preds = %1272, %1274
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %837		; <ubyte*>:1899 [#uses=2]
	load ubyte* %1899		; <ubyte>:3003 [#uses=1]
	add ubyte %3003, 255		; <ubyte>:3004 [#uses=1]
	store ubyte %3004, ubyte* %1899
	add uint %837, 6		; <uint>:840 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %840		; <ubyte*>:1900 [#uses=2]
	load ubyte* %1900		; <ubyte>:3005 [#uses=1]
	add ubyte %3005, 1		; <ubyte>:3006 [#uses=1]
	store ubyte %3006, ubyte* %1900
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %837		; <ubyte*>:1901 [#uses=1]
	load ubyte* %1901		; <ubyte>:3007 [#uses=1]
	seteq ubyte %3007, 0		; <bool>:1274 [#uses=1]
	br bool %1274, label %1275, label %1274

; <label>:1275		; preds = %1272, %1274
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %837		; <ubyte*>:1902 [#uses=2]
	load ubyte* %1902		; <ubyte>:3008 [#uses=1]
	add ubyte %3008, 1		; <ubyte>:3009 [#uses=1]
	store ubyte %3009, ubyte* %1902
	add uint %837, 6		; <uint>:841 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %841		; <ubyte*>:1903 [#uses=2]
	load ubyte* %1903		; <ubyte>:3010 [#uses=2]
	add ubyte %3010, 255		; <ubyte>:3011 [#uses=1]
	store ubyte %3011, ubyte* %1903
	seteq ubyte %3010, 1		; <bool>:1275 [#uses=1]
	br bool %1275, label %1273, label %1272

; <label>:1276		; preds = %1273, %1276
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %839		; <ubyte*>:1904 [#uses=2]
	load ubyte* %1904		; <ubyte>:3012 [#uses=2]
	add ubyte %3012, 255		; <ubyte>:3013 [#uses=1]
	store ubyte %3013, ubyte* %1904
	seteq ubyte %3012, 1		; <bool>:1276 [#uses=1]
	br bool %1276, label %1277, label %1276

; <label>:1277		; preds = %1273, %1276
	add uint %838, 4294967294		; <uint>:842 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %842		; <ubyte*>:1905 [#uses=1]
	load ubyte* %1905		; <ubyte>:3014 [#uses=1]
	seteq ubyte %3014, 0		; <bool>:1277 [#uses=1]
	br bool %1277, label %1279, label %1278

; <label>:1278		; preds = %1277, %1278
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %839		; <ubyte*>:1906 [#uses=2]
	load ubyte* %1906		; <ubyte>:3015 [#uses=1]
	add ubyte %3015, 1		; <ubyte>:3016 [#uses=1]
	store ubyte %3016, ubyte* %1906
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %842		; <ubyte*>:1907 [#uses=2]
	load ubyte* %1907		; <ubyte>:3017 [#uses=1]
	add ubyte %3017, 255		; <ubyte>:3018 [#uses=1]
	store ubyte %3018, ubyte* %1907
	add uint %838, 4294967295		; <uint>:843 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %843		; <ubyte*>:1908 [#uses=2]
	load ubyte* %1908		; <ubyte>:3019 [#uses=1]
	add ubyte %3019, 1		; <ubyte>:3020 [#uses=1]
	store ubyte %3020, ubyte* %1908
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %842		; <ubyte*>:1909 [#uses=1]
	load ubyte* %1909		; <ubyte>:3021 [#uses=1]
	seteq ubyte %3021, 0		; <bool>:1278 [#uses=1]
	br bool %1278, label %1279, label %1278

; <label>:1279		; preds = %1277, %1278
	add uint %838, 4294967295		; <uint>:844 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %844		; <ubyte*>:1910 [#uses=1]
	load ubyte* %1910		; <ubyte>:3022 [#uses=1]
	seteq ubyte %3022, 0		; <bool>:1279 [#uses=1]
	br bool %1279, label %1281, label %1280

; <label>:1280		; preds = %1279, %1280
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %842		; <ubyte*>:1911 [#uses=2]
	load ubyte* %1911		; <ubyte>:3023 [#uses=1]
	add ubyte %3023, 1		; <ubyte>:3024 [#uses=1]
	store ubyte %3024, ubyte* %1911
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %844		; <ubyte*>:1912 [#uses=2]
	load ubyte* %1912		; <ubyte>:3025 [#uses=2]
	add ubyte %3025, 255		; <ubyte>:3026 [#uses=1]
	store ubyte %3026, ubyte* %1912
	seteq ubyte %3025, 1		; <bool>:1280 [#uses=1]
	br bool %1280, label %1281, label %1280

; <label>:1281		; preds = %1279, %1280
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %838		; <ubyte*>:1913 [#uses=2]
	load ubyte* %1913		; <ubyte>:3027 [#uses=2]
	add ubyte %3027, 1		; <ubyte>:3028 [#uses=1]
	store ubyte %3028, ubyte* %1913
	seteq ubyte %3027, 255		; <bool>:1281 [#uses=1]
	br bool %1281, label %1283, label %1282

; <label>:1282		; preds = %1281, %1287
	phi uint [ %838, %1281 ], [ %850, %1287 ]		; <uint>:845 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %845		; <ubyte*>:1914 [#uses=2]
	load ubyte* %1914		; <ubyte>:3029 [#uses=1]
	add ubyte %3029, 255		; <ubyte>:3030 [#uses=1]
	store ubyte %3030, ubyte* %1914
	add uint %845, 4294967286		; <uint>:846 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %846		; <ubyte*>:1915 [#uses=1]
	load ubyte* %1915		; <ubyte>:3031 [#uses=1]
	seteq ubyte %3031, 0		; <bool>:1282 [#uses=1]
	br bool %1282, label %1285, label %1284

; <label>:1283		; preds = %1281, %1287
	phi uint [ %838, %1281 ], [ %850, %1287 ]		; <uint>:847 [#uses=22]
	add uint %847, 4		; <uint>:848 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %848		; <ubyte*>:1916 [#uses=1]
	load ubyte* %1916		; <ubyte>:3032 [#uses=1]
	seteq ubyte %3032, 0		; <bool>:1283 [#uses=1]
	br bool %1283, label %1289, label %1288

; <label>:1284		; preds = %1282, %1284
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %846		; <ubyte*>:1917 [#uses=2]
	load ubyte* %1917		; <ubyte>:3033 [#uses=2]
	add ubyte %3033, 255		; <ubyte>:3034 [#uses=1]
	store ubyte %3034, ubyte* %1917
	seteq ubyte %3033, 1		; <bool>:1284 [#uses=1]
	br bool %1284, label %1285, label %1284

; <label>:1285		; preds = %1282, %1284
	add uint %845, 4294967292		; <uint>:849 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %849		; <ubyte*>:1918 [#uses=1]
	load ubyte* %1918		; <ubyte>:3035 [#uses=1]
	seteq ubyte %3035, 0		; <bool>:1285 [#uses=1]
	br bool %1285, label %1287, label %1286

; <label>:1286		; preds = %1285, %1286
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %846		; <ubyte*>:1919 [#uses=2]
	load ubyte* %1919		; <ubyte>:3036 [#uses=1]
	add ubyte %3036, 1		; <ubyte>:3037 [#uses=1]
	store ubyte %3037, ubyte* %1919
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %849		; <ubyte*>:1920 [#uses=2]
	load ubyte* %1920		; <ubyte>:3038 [#uses=2]
	add ubyte %3038, 255		; <ubyte>:3039 [#uses=1]
	store ubyte %3039, ubyte* %1920
	seteq ubyte %3038, 1		; <bool>:1286 [#uses=1]
	br bool %1286, label %1287, label %1286

; <label>:1287		; preds = %1285, %1286
	add uint %845, 4294967290		; <uint>:850 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %850		; <ubyte*>:1921 [#uses=1]
	load ubyte* %1921		; <ubyte>:3040 [#uses=1]
	seteq ubyte %3040, 0		; <bool>:1287 [#uses=1]
	br bool %1287, label %1283, label %1282

; <label>:1288		; preds = %1283, %1288
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %848		; <ubyte*>:1922 [#uses=2]
	load ubyte* %1922		; <ubyte>:3041 [#uses=2]
	add ubyte %3041, 255		; <ubyte>:3042 [#uses=1]
	store ubyte %3042, ubyte* %1922
	seteq ubyte %3041, 1		; <bool>:1288 [#uses=1]
	br bool %1288, label %1289, label %1288

; <label>:1289		; preds = %1283, %1288
	add uint %847, 10		; <uint>:851 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %851		; <ubyte*>:1923 [#uses=1]
	load ubyte* %1923		; <ubyte>:3043 [#uses=1]
	seteq ubyte %3043, 0		; <bool>:1289 [#uses=1]
	br bool %1289, label %1291, label %1290

; <label>:1290		; preds = %1289, %1290
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %851		; <ubyte*>:1924 [#uses=2]
	load ubyte* %1924		; <ubyte>:3044 [#uses=2]
	add ubyte %3044, 255		; <ubyte>:3045 [#uses=1]
	store ubyte %3045, ubyte* %1924
	seteq ubyte %3044, 1		; <bool>:1290 [#uses=1]
	br bool %1290, label %1291, label %1290

; <label>:1291		; preds = %1289, %1290
	add uint %847, 16		; <uint>:852 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %852		; <ubyte*>:1925 [#uses=1]
	load ubyte* %1925		; <ubyte>:3046 [#uses=1]
	seteq ubyte %3046, 0		; <bool>:1291 [#uses=1]
	br bool %1291, label %1293, label %1292

; <label>:1292		; preds = %1291, %1292
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %852		; <ubyte*>:1926 [#uses=2]
	load ubyte* %1926		; <ubyte>:3047 [#uses=2]
	add ubyte %3047, 255		; <ubyte>:3048 [#uses=1]
	store ubyte %3048, ubyte* %1926
	seteq ubyte %3047, 1		; <bool>:1292 [#uses=1]
	br bool %1292, label %1293, label %1292

; <label>:1293		; preds = %1291, %1292
	add uint %847, 22		; <uint>:853 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %853		; <ubyte*>:1927 [#uses=1]
	load ubyte* %1927		; <ubyte>:3049 [#uses=1]
	seteq ubyte %3049, 0		; <bool>:1293 [#uses=1]
	br bool %1293, label %1295, label %1294

; <label>:1294		; preds = %1293, %1294
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %853		; <ubyte*>:1928 [#uses=2]
	load ubyte* %1928		; <ubyte>:3050 [#uses=2]
	add ubyte %3050, 255		; <ubyte>:3051 [#uses=1]
	store ubyte %3051, ubyte* %1928
	seteq ubyte %3050, 1		; <bool>:1294 [#uses=1]
	br bool %1294, label %1295, label %1294

; <label>:1295		; preds = %1293, %1294
	add uint %847, 28		; <uint>:854 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %854		; <ubyte*>:1929 [#uses=1]
	load ubyte* %1929		; <ubyte>:3052 [#uses=1]
	seteq ubyte %3052, 0		; <bool>:1295 [#uses=1]
	br bool %1295, label %1297, label %1296

; <label>:1296		; preds = %1295, %1296
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %854		; <ubyte*>:1930 [#uses=2]
	load ubyte* %1930		; <ubyte>:3053 [#uses=2]
	add ubyte %3053, 255		; <ubyte>:3054 [#uses=1]
	store ubyte %3054, ubyte* %1930
	seteq ubyte %3053, 1		; <bool>:1296 [#uses=1]
	br bool %1296, label %1297, label %1296

; <label>:1297		; preds = %1295, %1296
	add uint %847, 34		; <uint>:855 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %855		; <ubyte*>:1931 [#uses=1]
	load ubyte* %1931		; <ubyte>:3055 [#uses=1]
	seteq ubyte %3055, 0		; <bool>:1297 [#uses=1]
	br bool %1297, label %1299, label %1298

; <label>:1298		; preds = %1297, %1298
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %855		; <ubyte*>:1932 [#uses=2]
	load ubyte* %1932		; <ubyte>:3056 [#uses=2]
	add ubyte %3056, 255		; <ubyte>:3057 [#uses=1]
	store ubyte %3057, ubyte* %1932
	seteq ubyte %3056, 1		; <bool>:1298 [#uses=1]
	br bool %1298, label %1299, label %1298

; <label>:1299		; preds = %1297, %1298
	add uint %847, 40		; <uint>:856 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %856		; <ubyte*>:1933 [#uses=1]
	load ubyte* %1933		; <ubyte>:3058 [#uses=1]
	seteq ubyte %3058, 0		; <bool>:1299 [#uses=1]
	br bool %1299, label %1301, label %1300

; <label>:1300		; preds = %1299, %1300
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %856		; <ubyte*>:1934 [#uses=2]
	load ubyte* %1934		; <ubyte>:3059 [#uses=2]
	add ubyte %3059, 255		; <ubyte>:3060 [#uses=1]
	store ubyte %3060, ubyte* %1934
	seteq ubyte %3059, 1		; <bool>:1300 [#uses=1]
	br bool %1300, label %1301, label %1300

; <label>:1301		; preds = %1299, %1300
	add uint %847, 46		; <uint>:857 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %857		; <ubyte*>:1935 [#uses=1]
	load ubyte* %1935		; <ubyte>:3061 [#uses=1]
	seteq ubyte %3061, 0		; <bool>:1301 [#uses=1]
	br bool %1301, label %1303, label %1302

; <label>:1302		; preds = %1301, %1302
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %857		; <ubyte*>:1936 [#uses=2]
	load ubyte* %1936		; <ubyte>:3062 [#uses=2]
	add ubyte %3062, 255		; <ubyte>:3063 [#uses=1]
	store ubyte %3063, ubyte* %1936
	seteq ubyte %3062, 1		; <bool>:1302 [#uses=1]
	br bool %1302, label %1303, label %1302

; <label>:1303		; preds = %1301, %1302
	add uint %847, 52		; <uint>:858 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %858		; <ubyte*>:1937 [#uses=1]
	load ubyte* %1937		; <ubyte>:3064 [#uses=1]
	seteq ubyte %3064, 0		; <bool>:1303 [#uses=1]
	br bool %1303, label %1305, label %1304

; <label>:1304		; preds = %1303, %1304
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %858		; <ubyte*>:1938 [#uses=2]
	load ubyte* %1938		; <ubyte>:3065 [#uses=2]
	add ubyte %3065, 255		; <ubyte>:3066 [#uses=1]
	store ubyte %3066, ubyte* %1938
	seteq ubyte %3065, 1		; <bool>:1304 [#uses=1]
	br bool %1304, label %1305, label %1304

; <label>:1305		; preds = %1303, %1304
	add uint %847, 58		; <uint>:859 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %859		; <ubyte*>:1939 [#uses=1]
	load ubyte* %1939		; <ubyte>:3067 [#uses=1]
	seteq ubyte %3067, 0		; <bool>:1305 [#uses=1]
	br bool %1305, label %1307, label %1306

; <label>:1306		; preds = %1305, %1306
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %859		; <ubyte*>:1940 [#uses=2]
	load ubyte* %1940		; <ubyte>:3068 [#uses=2]
	add ubyte %3068, 255		; <ubyte>:3069 [#uses=1]
	store ubyte %3069, ubyte* %1940
	seteq ubyte %3068, 1		; <bool>:1306 [#uses=1]
	br bool %1306, label %1307, label %1306

; <label>:1307		; preds = %1305, %1306
	add uint %847, 64		; <uint>:860 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %860		; <ubyte*>:1941 [#uses=1]
	load ubyte* %1941		; <ubyte>:3070 [#uses=1]
	seteq ubyte %3070, 0		; <bool>:1307 [#uses=1]
	br bool %1307, label %1309, label %1308

; <label>:1308		; preds = %1307, %1308
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %860		; <ubyte*>:1942 [#uses=2]
	load ubyte* %1942		; <ubyte>:3071 [#uses=2]
	add ubyte %3071, 255		; <ubyte>:3072 [#uses=1]
	store ubyte %3072, ubyte* %1942
	seteq ubyte %3071, 1		; <bool>:1308 [#uses=1]
	br bool %1308, label %1309, label %1308

; <label>:1309		; preds = %1307, %1308
	add uint %847, 70		; <uint>:861 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %861		; <ubyte*>:1943 [#uses=1]
	load ubyte* %1943		; <ubyte>:3073 [#uses=1]
	seteq ubyte %3073, 0		; <bool>:1309 [#uses=1]
	br bool %1309, label %1311, label %1310

; <label>:1310		; preds = %1309, %1310
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %861		; <ubyte*>:1944 [#uses=2]
	load ubyte* %1944		; <ubyte>:3074 [#uses=2]
	add ubyte %3074, 255		; <ubyte>:3075 [#uses=1]
	store ubyte %3075, ubyte* %1944
	seteq ubyte %3074, 1		; <bool>:1310 [#uses=1]
	br bool %1310, label %1311, label %1310

; <label>:1311		; preds = %1309, %1310
	add uint %847, 76		; <uint>:862 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %862		; <ubyte*>:1945 [#uses=1]
	load ubyte* %1945		; <ubyte>:3076 [#uses=1]
	seteq ubyte %3076, 0		; <bool>:1311 [#uses=1]
	br bool %1311, label %1313, label %1312

; <label>:1312		; preds = %1311, %1312
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %862		; <ubyte*>:1946 [#uses=2]
	load ubyte* %1946		; <ubyte>:3077 [#uses=2]
	add ubyte %3077, 255		; <ubyte>:3078 [#uses=1]
	store ubyte %3078, ubyte* %1946
	seteq ubyte %3077, 1		; <bool>:1312 [#uses=1]
	br bool %1312, label %1313, label %1312

; <label>:1313		; preds = %1311, %1312
	add uint %847, 82		; <uint>:863 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %863		; <ubyte*>:1947 [#uses=1]
	load ubyte* %1947		; <ubyte>:3079 [#uses=1]
	seteq ubyte %3079, 0		; <bool>:1313 [#uses=1]
	br bool %1313, label %1315, label %1314

; <label>:1314		; preds = %1313, %1314
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %863		; <ubyte*>:1948 [#uses=2]
	load ubyte* %1948		; <ubyte>:3080 [#uses=2]
	add ubyte %3080, 255		; <ubyte>:3081 [#uses=1]
	store ubyte %3081, ubyte* %1948
	seteq ubyte %3080, 1		; <bool>:1314 [#uses=1]
	br bool %1314, label %1315, label %1314

; <label>:1315		; preds = %1313, %1314
	add uint %847, 88		; <uint>:864 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %864		; <ubyte*>:1949 [#uses=1]
	load ubyte* %1949		; <ubyte>:3082 [#uses=1]
	seteq ubyte %3082, 0		; <bool>:1315 [#uses=1]
	br bool %1315, label %1317, label %1316

; <label>:1316		; preds = %1315, %1316
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %864		; <ubyte*>:1950 [#uses=2]
	load ubyte* %1950		; <ubyte>:3083 [#uses=2]
	add ubyte %3083, 255		; <ubyte>:3084 [#uses=1]
	store ubyte %3084, ubyte* %1950
	seteq ubyte %3083, 1		; <bool>:1316 [#uses=1]
	br bool %1316, label %1317, label %1316

; <label>:1317		; preds = %1315, %1316
	add uint %847, 4294967294		; <uint>:865 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %865		; <ubyte*>:1951 [#uses=1]
	load ubyte* %1951		; <ubyte>:3085 [#uses=1]
	seteq ubyte %3085, 0		; <bool>:1317 [#uses=1]
	br bool %1317, label %1319, label %1318

; <label>:1318		; preds = %1317, %1318
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %865		; <ubyte*>:1952 [#uses=2]
	load ubyte* %1952		; <ubyte>:3086 [#uses=2]
	add ubyte %3086, 255		; <ubyte>:3087 [#uses=1]
	store ubyte %3087, ubyte* %1952
	seteq ubyte %3086, 1		; <bool>:1318 [#uses=1]
	br bool %1318, label %1319, label %1318

; <label>:1319		; preds = %1317, %1318
	add uint %847, 4294967286		; <uint>:866 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %866		; <ubyte*>:1953 [#uses=1]
	load ubyte* %1953		; <ubyte>:3088 [#uses=1]
	seteq ubyte %3088, 0		; <bool>:1319 [#uses=1]
	br bool %1319, label %1321, label %1320

; <label>:1320		; preds = %1319, %1320
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %866		; <ubyte*>:1954 [#uses=2]
	load ubyte* %1954		; <ubyte>:3089 [#uses=1]
	add ubyte %3089, 255		; <ubyte>:3090 [#uses=1]
	store ubyte %3090, ubyte* %1954
	add uint %847, 4294967287		; <uint>:867 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %867		; <ubyte*>:1955 [#uses=2]
	load ubyte* %1955		; <ubyte>:3091 [#uses=1]
	add ubyte %3091, 1		; <ubyte>:3092 [#uses=1]
	store ubyte %3092, ubyte* %1955
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %865		; <ubyte*>:1956 [#uses=2]
	load ubyte* %1956		; <ubyte>:3093 [#uses=1]
	add ubyte %3093, 1		; <ubyte>:3094 [#uses=1]
	store ubyte %3094, ubyte* %1956
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %866		; <ubyte*>:1957 [#uses=1]
	load ubyte* %1957		; <ubyte>:3095 [#uses=1]
	seteq ubyte %3095, 0		; <bool>:1320 [#uses=1]
	br bool %1320, label %1321, label %1320

; <label>:1321		; preds = %1319, %1320
	add uint %847, 4294967287		; <uint>:868 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %868		; <ubyte*>:1958 [#uses=1]
	load ubyte* %1958		; <ubyte>:3096 [#uses=1]
	seteq ubyte %3096, 0		; <bool>:1321 [#uses=1]
	br bool %1321, label %1323, label %1322

; <label>:1322		; preds = %1321, %1322
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %866		; <ubyte*>:1959 [#uses=2]
	load ubyte* %1959		; <ubyte>:3097 [#uses=1]
	add ubyte %3097, 1		; <ubyte>:3098 [#uses=1]
	store ubyte %3098, ubyte* %1959
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %868		; <ubyte*>:1960 [#uses=2]
	load ubyte* %1960		; <ubyte>:3099 [#uses=2]
	add ubyte %3099, 255		; <ubyte>:3100 [#uses=1]
	store ubyte %3100, ubyte* %1960
	seteq ubyte %3099, 1		; <bool>:1322 [#uses=1]
	br bool %1322, label %1323, label %1322

; <label>:1323		; preds = %1321, %1322
	add uint %847, 4294967189		; <uint>:869 [#uses=6]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %869		; <ubyte*>:1961 [#uses=1]
	load ubyte* %1961		; <ubyte>:3101 [#uses=1]
	seteq ubyte %3101, 0		; <bool>:1323 [#uses=1]
	br bool %1323, label %1325, label %1324

; <label>:1324		; preds = %1323, %1324
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %869		; <ubyte*>:1962 [#uses=2]
	load ubyte* %1962		; <ubyte>:3102 [#uses=2]
	add ubyte %3102, 255		; <ubyte>:3103 [#uses=1]
	store ubyte %3103, ubyte* %1962
	seteq ubyte %3102, 1		; <bool>:1324 [#uses=1]
	br bool %1324, label %1325, label %1324

; <label>:1325		; preds = %1323, %1324
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %865		; <ubyte*>:1963 [#uses=1]
	load ubyte* %1963		; <ubyte>:3104 [#uses=1]
	seteq ubyte %3104, 0		; <bool>:1325 [#uses=1]
	br bool %1325, label %1327, label %1326

; <label>:1326		; preds = %1325, %1326
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %869		; <ubyte*>:1964 [#uses=2]
	load ubyte* %1964		; <ubyte>:3105 [#uses=1]
	add ubyte %3105, 1		; <ubyte>:3106 [#uses=1]
	store ubyte %3106, ubyte* %1964
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %865		; <ubyte*>:1965 [#uses=2]
	load ubyte* %1965		; <ubyte>:3107 [#uses=2]
	add ubyte %3107, 255		; <ubyte>:3108 [#uses=1]
	store ubyte %3108, ubyte* %1965
	seteq ubyte %3107, 1		; <bool>:1326 [#uses=1]
	br bool %1326, label %1327, label %1326

; <label>:1327		; preds = %1325, %1326
	add uint %847, 4294967179		; <uint>:870 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %870		; <ubyte*>:1966 [#uses=1]
	load ubyte* %1966		; <ubyte>:3109 [#uses=1]
	seteq ubyte %3109, 0		; <bool>:1327 [#uses=1]
	br bool %1327, label %1329, label %1328

; <label>:1328		; preds = %1327, %1328
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %870		; <ubyte*>:1967 [#uses=2]
	load ubyte* %1967		; <ubyte>:3110 [#uses=2]
	add ubyte %3110, 255		; <ubyte>:3111 [#uses=1]
	store ubyte %3111, ubyte* %1967
	seteq ubyte %3110, 1		; <bool>:1328 [#uses=1]
	br bool %1328, label %1329, label %1328

; <label>:1329		; preds = %1327, %1328
	add uint %847, 4294967292		; <uint>:871 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %871		; <ubyte*>:1968 [#uses=1]
	load ubyte* %1968		; <ubyte>:3112 [#uses=1]
	seteq ubyte %3112, 0		; <bool>:1329 [#uses=1]
	br bool %1329, label %1331, label %1330

; <label>:1330		; preds = %1329, %1330
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %870		; <ubyte*>:1969 [#uses=2]
	load ubyte* %1969		; <ubyte>:3113 [#uses=1]
	add ubyte %3113, 1		; <ubyte>:3114 [#uses=1]
	store ubyte %3114, ubyte* %1969
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %871		; <ubyte*>:1970 [#uses=2]
	load ubyte* %1970		; <ubyte>:3115 [#uses=2]
	add ubyte %3115, 255		; <ubyte>:3116 [#uses=1]
	store ubyte %3116, ubyte* %1970
	seteq ubyte %3115, 1		; <bool>:1330 [#uses=1]
	br bool %1330, label %1331, label %1330

; <label>:1331		; preds = %1329, %1330
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %869		; <ubyte*>:1971 [#uses=1]
	load ubyte* %1971		; <ubyte>:3117 [#uses=1]
	seteq ubyte %3117, 0		; <bool>:1331 [#uses=1]
	br bool %1331, label %1333, label %1332

; <label>:1332		; preds = %1331, %1339
	phi uint [ %869, %1331 ], [ %878, %1339 ]		; <uint>:872 [#uses=8]
	add uint %872, 4294967292		; <uint>:873 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %873		; <ubyte*>:1972 [#uses=1]
	load ubyte* %1972		; <ubyte>:3118 [#uses=1]
	seteq ubyte %3118, 0		; <bool>:1332 [#uses=1]
	br bool %1332, label %1335, label %1334

; <label>:1333		; preds = %1331, %1339
	phi uint [ %869, %1331 ], [ %878, %1339 ]		; <uint>:874 [#uses=5]
	add uint %874, 4294967294		; <uint>:875 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %875		; <ubyte*>:1973 [#uses=1]
	load ubyte* %1973		; <ubyte>:3119 [#uses=1]
	seteq ubyte %3119, 0		; <bool>:1333 [#uses=1]
	br bool %1333, label %1341, label %1340

; <label>:1334		; preds = %1332, %1334
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %873		; <ubyte*>:1974 [#uses=2]
	load ubyte* %1974		; <ubyte>:3120 [#uses=2]
	add ubyte %3120, 255		; <ubyte>:3121 [#uses=1]
	store ubyte %3121, ubyte* %1974
	seteq ubyte %3120, 1		; <bool>:1334 [#uses=1]
	br bool %1334, label %1335, label %1334

; <label>:1335		; preds = %1332, %1334
	add uint %872, 4294967286		; <uint>:876 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %876		; <ubyte*>:1975 [#uses=1]
	load ubyte* %1975		; <ubyte>:3122 [#uses=1]
	seteq ubyte %3122, 0		; <bool>:1335 [#uses=1]
	br bool %1335, label %1337, label %1336

; <label>:1336		; preds = %1335, %1336
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %876		; <ubyte*>:1976 [#uses=2]
	load ubyte* %1976		; <ubyte>:3123 [#uses=1]
	add ubyte %3123, 255		; <ubyte>:3124 [#uses=1]
	store ubyte %3124, ubyte* %1976
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %873		; <ubyte*>:1977 [#uses=2]
	load ubyte* %1977		; <ubyte>:3125 [#uses=1]
	add ubyte %3125, 1		; <ubyte>:3126 [#uses=1]
	store ubyte %3126, ubyte* %1977
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %876		; <ubyte*>:1978 [#uses=1]
	load ubyte* %1978		; <ubyte>:3127 [#uses=1]
	seteq ubyte %3127, 0		; <bool>:1336 [#uses=1]
	br bool %1336, label %1337, label %1336

; <label>:1337		; preds = %1335, %1336
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %872		; <ubyte*>:1979 [#uses=1]
	load ubyte* %1979		; <ubyte>:3128 [#uses=1]
	seteq ubyte %3128, 0		; <bool>:1337 [#uses=1]
	br bool %1337, label %1339, label %1338

; <label>:1338		; preds = %1337, %1338
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %872		; <ubyte*>:1980 [#uses=2]
	load ubyte* %1980		; <ubyte>:3129 [#uses=1]
	add ubyte %3129, 255		; <ubyte>:3130 [#uses=1]
	store ubyte %3130, ubyte* %1980
	add uint %872, 6		; <uint>:877 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %877		; <ubyte*>:1981 [#uses=2]
	load ubyte* %1981		; <ubyte>:3131 [#uses=1]
	add ubyte %3131, 1		; <ubyte>:3132 [#uses=1]
	store ubyte %3132, ubyte* %1981
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %872		; <ubyte*>:1982 [#uses=1]
	load ubyte* %1982		; <ubyte>:3133 [#uses=1]
	seteq ubyte %3133, 0		; <bool>:1338 [#uses=1]
	br bool %1338, label %1339, label %1338

; <label>:1339		; preds = %1337, %1338
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %872		; <ubyte*>:1983 [#uses=2]
	load ubyte* %1983		; <ubyte>:3134 [#uses=1]
	add ubyte %3134, 1		; <ubyte>:3135 [#uses=1]
	store ubyte %3135, ubyte* %1983
	add uint %872, 6		; <uint>:878 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %878		; <ubyte*>:1984 [#uses=2]
	load ubyte* %1984		; <ubyte>:3136 [#uses=2]
	add ubyte %3136, 255		; <ubyte>:3137 [#uses=1]
	store ubyte %3137, ubyte* %1984
	seteq ubyte %3136, 1		; <bool>:1339 [#uses=1]
	br bool %1339, label %1333, label %1332

; <label>:1340		; preds = %1333, %1340
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %875		; <ubyte*>:1985 [#uses=2]
	load ubyte* %1985		; <ubyte>:3138 [#uses=2]
	add ubyte %3138, 255		; <ubyte>:3139 [#uses=1]
	store ubyte %3139, ubyte* %1985
	seteq ubyte %3138, 1		; <bool>:1340 [#uses=1]
	br bool %1340, label %1341, label %1340

; <label>:1341		; preds = %1333, %1340
	add uint %874, 4294967286		; <uint>:879 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %879		; <ubyte*>:1986 [#uses=1]
	load ubyte* %1986		; <ubyte>:3140 [#uses=1]
	seteq ubyte %3140, 0		; <bool>:1341 [#uses=1]
	br bool %1341, label %1343, label %1342

; <label>:1342		; preds = %1341, %1342
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %879		; <ubyte*>:1987 [#uses=2]
	load ubyte* %1987		; <ubyte>:3141 [#uses=1]
	add ubyte %3141, 255		; <ubyte>:3142 [#uses=1]
	store ubyte %3142, ubyte* %1987
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %875		; <ubyte*>:1988 [#uses=2]
	load ubyte* %1988		; <ubyte>:3143 [#uses=1]
	add ubyte %3143, 1		; <ubyte>:3144 [#uses=1]
	store ubyte %3144, ubyte* %1988
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %879		; <ubyte*>:1989 [#uses=1]
	load ubyte* %1989		; <ubyte>:3145 [#uses=1]
	seteq ubyte %3145, 0		; <bool>:1342 [#uses=1]
	br bool %1342, label %1343, label %1342

; <label>:1343		; preds = %1341, %1342
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %874		; <ubyte*>:1990 [#uses=2]
	load ubyte* %1990		; <ubyte>:3146 [#uses=2]
	add ubyte %3146, 1		; <ubyte>:3147 [#uses=1]
	store ubyte %3147, ubyte* %1990
	seteq ubyte %3146, 255		; <bool>:1343 [#uses=1]
	br bool %1343, label %1345, label %1344

; <label>:1344		; preds = %1343, %1344
	phi uint [ %874, %1343 ], [ %881, %1344 ]		; <uint>:880 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %880		; <ubyte*>:1991 [#uses=2]
	load ubyte* %1991		; <ubyte>:3148 [#uses=1]
	add ubyte %3148, 255		; <ubyte>:3149 [#uses=1]
	store ubyte %3149, ubyte* %1991
	add uint %880, 4294967290		; <uint>:881 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %881		; <ubyte*>:1992 [#uses=1]
	load ubyte* %1992		; <ubyte>:3150 [#uses=1]
	seteq ubyte %3150, 0		; <bool>:1344 [#uses=1]
	br bool %1344, label %1345, label %1344

; <label>:1345		; preds = %1343, %1344
	phi uint [ %874, %1343 ], [ %881, %1344 ]		; <uint>:882 [#uses=10]
	add uint %882, 109		; <uint>:883 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %883		; <ubyte*>:1993 [#uses=1]
	load ubyte* %1993		; <ubyte>:3151 [#uses=1]
	seteq ubyte %3151, 0		; <bool>:1345 [#uses=1]
	br bool %1345, label %1347, label %1346

; <label>:1346		; preds = %1345, %1346
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %883		; <ubyte*>:1994 [#uses=2]
	load ubyte* %1994		; <ubyte>:3152 [#uses=2]
	add ubyte %3152, 255		; <ubyte>:3153 [#uses=1]
	store ubyte %3153, ubyte* %1994
	seteq ubyte %3152, 1		; <bool>:1346 [#uses=1]
	br bool %1346, label %1347, label %1346

; <label>:1347		; preds = %1345, %1346
	add uint %882, 107		; <uint>:884 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %884		; <ubyte*>:1995 [#uses=1]
	load ubyte* %1995		; <ubyte>:3154 [#uses=1]
	seteq ubyte %3154, 0		; <bool>:1347 [#uses=1]
	br bool %1347, label %1349, label %1348

; <label>:1348		; preds = %1347, %1348
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %884		; <ubyte*>:1996 [#uses=2]
	load ubyte* %1996		; <ubyte>:3155 [#uses=1]
	add ubyte %3155, 255		; <ubyte>:3156 [#uses=1]
	store ubyte %3156, ubyte* %1996
	add uint %882, 108		; <uint>:885 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %885		; <ubyte*>:1997 [#uses=2]
	load ubyte* %1997		; <ubyte>:3157 [#uses=1]
	add ubyte %3157, 1		; <ubyte>:3158 [#uses=1]
	store ubyte %3158, ubyte* %1997
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %883		; <ubyte*>:1998 [#uses=2]
	load ubyte* %1998		; <ubyte>:3159 [#uses=1]
	add ubyte %3159, 1		; <ubyte>:3160 [#uses=1]
	store ubyte %3160, ubyte* %1998
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %884		; <ubyte*>:1999 [#uses=1]
	load ubyte* %1999		; <ubyte>:3161 [#uses=1]
	seteq ubyte %3161, 0		; <bool>:1348 [#uses=1]
	br bool %1348, label %1349, label %1348

; <label>:1349		; preds = %1347, %1348
	add uint %882, 108		; <uint>:886 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %886		; <ubyte*>:2000 [#uses=1]
	load ubyte* %2000		; <ubyte>:3162 [#uses=1]
	seteq ubyte %3162, 0		; <bool>:1349 [#uses=1]
	br bool %1349, label %1351, label %1350

; <label>:1350		; preds = %1349, %1350
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %884		; <ubyte*>:2001 [#uses=2]
	load ubyte* %2001		; <ubyte>:3163 [#uses=1]
	add ubyte %3163, 1		; <ubyte>:3164 [#uses=1]
	store ubyte %3164, ubyte* %2001
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %886		; <ubyte*>:2002 [#uses=2]
	load ubyte* %2002		; <ubyte>:3165 [#uses=2]
	add ubyte %3165, 255		; <ubyte>:3166 [#uses=1]
	store ubyte %3166, ubyte* %2002
	seteq ubyte %3165, 1		; <bool>:1350 [#uses=1]
	br bool %1350, label %1351, label %1350

; <label>:1351		; preds = %1349, %1350
	add uint %882, 111		; <uint>:887 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %887		; <ubyte*>:2003 [#uses=1]
	load ubyte* %2003		; <ubyte>:3167 [#uses=1]
	seteq ubyte %3167, 0		; <bool>:1351 [#uses=1]
	br bool %1351, label %1353, label %1352

; <label>:1352		; preds = %1351, %1352
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %887		; <ubyte*>:2004 [#uses=2]
	load ubyte* %2004		; <ubyte>:3168 [#uses=2]
	add ubyte %3168, 255		; <ubyte>:3169 [#uses=1]
	store ubyte %3169, ubyte* %2004
	seteq ubyte %3168, 1		; <bool>:1352 [#uses=1]
	br bool %1352, label %1353, label %1352

; <label>:1353		; preds = %1351, %1352
	add uint %882, 105		; <uint>:888 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %888		; <ubyte*>:2005 [#uses=1]
	load ubyte* %2005		; <ubyte>:3170 [#uses=1]
	seteq ubyte %3170, 0		; <bool>:1353 [#uses=1]
	br bool %1353, label %1355, label %1354

; <label>:1354		; preds = %1353, %1354
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %888		; <ubyte*>:2006 [#uses=2]
	load ubyte* %2006		; <ubyte>:3171 [#uses=1]
	add ubyte %3171, 255		; <ubyte>:3172 [#uses=1]
	store ubyte %3172, ubyte* %2006
	add uint %882, 106		; <uint>:889 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %889		; <ubyte*>:2007 [#uses=2]
	load ubyte* %2007		; <ubyte>:3173 [#uses=1]
	add ubyte %3173, 1		; <ubyte>:3174 [#uses=1]
	store ubyte %3174, ubyte* %2007
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %887		; <ubyte*>:2008 [#uses=2]
	load ubyte* %2008		; <ubyte>:3175 [#uses=1]
	add ubyte %3175, 1		; <ubyte>:3176 [#uses=1]
	store ubyte %3176, ubyte* %2008
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %888		; <ubyte*>:2009 [#uses=1]
	load ubyte* %2009		; <ubyte>:3177 [#uses=1]
	seteq ubyte %3177, 0		; <bool>:1354 [#uses=1]
	br bool %1354, label %1355, label %1354

; <label>:1355		; preds = %1353, %1354
	add uint %882, 106		; <uint>:890 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %890		; <ubyte*>:2010 [#uses=1]
	load ubyte* %2010		; <ubyte>:3178 [#uses=1]
	seteq ubyte %3178, 0		; <bool>:1355 [#uses=1]
	br bool %1355, label %1357, label %1356

; <label>:1356		; preds = %1355, %1356
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %888		; <ubyte*>:2011 [#uses=2]
	load ubyte* %2011		; <ubyte>:3179 [#uses=1]
	add ubyte %3179, 1		; <ubyte>:3180 [#uses=1]
	store ubyte %3180, ubyte* %2011
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %890		; <ubyte*>:2012 [#uses=2]
	load ubyte* %2012		; <ubyte>:3181 [#uses=2]
	add ubyte %3181, 255		; <ubyte>:3182 [#uses=1]
	store ubyte %3182, ubyte* %2012
	seteq ubyte %3181, 1		; <bool>:1356 [#uses=1]
	br bool %1356, label %1357, label %1356

; <label>:1357		; preds = %1355, %1356
	add uint %882, 6		; <uint>:891 [#uses=6]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %891		; <ubyte*>:2013 [#uses=1]
	load ubyte* %2013		; <ubyte>:3183 [#uses=1]
	seteq ubyte %3183, 0		; <bool>:1357 [#uses=1]
	br bool %1357, label %1359, label %1358

; <label>:1358		; preds = %1357, %1358
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %891		; <ubyte*>:2014 [#uses=2]
	load ubyte* %2014		; <ubyte>:3184 [#uses=2]
	add ubyte %3184, 255		; <ubyte>:3185 [#uses=1]
	store ubyte %3185, ubyte* %2014
	seteq ubyte %3184, 1		; <bool>:1358 [#uses=1]
	br bool %1358, label %1359, label %1358

; <label>:1359		; preds = %1357, %1358
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %887		; <ubyte*>:2015 [#uses=1]
	load ubyte* %2015		; <ubyte>:3186 [#uses=1]
	seteq ubyte %3186, 0		; <bool>:1359 [#uses=1]
	br bool %1359, label %1361, label %1360

; <label>:1360		; preds = %1359, %1360
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %891		; <ubyte*>:2016 [#uses=2]
	load ubyte* %2016		; <ubyte>:3187 [#uses=1]
	add ubyte %3187, 1		; <ubyte>:3188 [#uses=1]
	store ubyte %3188, ubyte* %2016
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %887		; <ubyte*>:2017 [#uses=2]
	load ubyte* %2017		; <ubyte>:3189 [#uses=2]
	add ubyte %3189, 255		; <ubyte>:3190 [#uses=1]
	store ubyte %3190, ubyte* %2017
	seteq ubyte %3189, 1		; <bool>:1360 [#uses=1]
	br bool %1360, label %1361, label %1360

; <label>:1361		; preds = %1359, %1360
	add uint %882, 4294967292		; <uint>:892 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %892		; <ubyte*>:2018 [#uses=1]
	load ubyte* %2018		; <ubyte>:3191 [#uses=1]
	seteq ubyte %3191, 0		; <bool>:1361 [#uses=1]
	br bool %1361, label %1363, label %1362

; <label>:1362		; preds = %1361, %1362
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %892		; <ubyte*>:2019 [#uses=2]
	load ubyte* %2019		; <ubyte>:3192 [#uses=2]
	add ubyte %3192, 255		; <ubyte>:3193 [#uses=1]
	store ubyte %3193, ubyte* %2019
	seteq ubyte %3192, 1		; <bool>:1362 [#uses=1]
	br bool %1362, label %1363, label %1362

; <label>:1363		; preds = %1361, %1362
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %883		; <ubyte*>:2020 [#uses=1]
	load ubyte* %2020		; <ubyte>:3194 [#uses=1]
	seteq ubyte %3194, 0		; <bool>:1363 [#uses=1]
	br bool %1363, label %1365, label %1364

; <label>:1364		; preds = %1363, %1364
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %892		; <ubyte*>:2021 [#uses=2]
	load ubyte* %2021		; <ubyte>:3195 [#uses=1]
	add ubyte %3195, 1		; <ubyte>:3196 [#uses=1]
	store ubyte %3196, ubyte* %2021
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %883		; <ubyte*>:2022 [#uses=2]
	load ubyte* %2022		; <ubyte>:3197 [#uses=2]
	add ubyte %3197, 255		; <ubyte>:3198 [#uses=1]
	store ubyte %3198, ubyte* %2022
	seteq ubyte %3197, 1		; <bool>:1364 [#uses=1]
	br bool %1364, label %1365, label %1364

; <label>:1365		; preds = %1363, %1364
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %891		; <ubyte*>:2023 [#uses=1]
	load ubyte* %2023		; <ubyte>:3199 [#uses=1]
	seteq ubyte %3199, 0		; <bool>:1365 [#uses=1]
	br bool %1365, label %1367, label %1366

; <label>:1366		; preds = %1365, %1373
	phi uint [ %891, %1365 ], [ %899, %1373 ]		; <uint>:893 [#uses=8]
	add uint %893, 4294967292		; <uint>:894 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %894		; <ubyte*>:2024 [#uses=1]
	load ubyte* %2024		; <ubyte>:3200 [#uses=1]
	seteq ubyte %3200, 0		; <bool>:1366 [#uses=1]
	br bool %1366, label %1369, label %1368

; <label>:1367		; preds = %1365, %1373
	phi uint [ %891, %1365 ], [ %899, %1373 ]		; <uint>:895 [#uses=5]
	add uint %895, 4294967294		; <uint>:896 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %896		; <ubyte*>:2025 [#uses=1]
	load ubyte* %2025		; <ubyte>:3201 [#uses=1]
	seteq ubyte %3201, 0		; <bool>:1367 [#uses=1]
	br bool %1367, label %1375, label %1374

; <label>:1368		; preds = %1366, %1368
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %894		; <ubyte*>:2026 [#uses=2]
	load ubyte* %2026		; <ubyte>:3202 [#uses=2]
	add ubyte %3202, 255		; <ubyte>:3203 [#uses=1]
	store ubyte %3203, ubyte* %2026
	seteq ubyte %3202, 1		; <bool>:1368 [#uses=1]
	br bool %1368, label %1369, label %1368

; <label>:1369		; preds = %1366, %1368
	add uint %893, 4294967286		; <uint>:897 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %897		; <ubyte*>:2027 [#uses=1]
	load ubyte* %2027		; <ubyte>:3204 [#uses=1]
	seteq ubyte %3204, 0		; <bool>:1369 [#uses=1]
	br bool %1369, label %1371, label %1370

; <label>:1370		; preds = %1369, %1370
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %897		; <ubyte*>:2028 [#uses=2]
	load ubyte* %2028		; <ubyte>:3205 [#uses=1]
	add ubyte %3205, 255		; <ubyte>:3206 [#uses=1]
	store ubyte %3206, ubyte* %2028
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %894		; <ubyte*>:2029 [#uses=2]
	load ubyte* %2029		; <ubyte>:3207 [#uses=1]
	add ubyte %3207, 1		; <ubyte>:3208 [#uses=1]
	store ubyte %3208, ubyte* %2029
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %897		; <ubyte*>:2030 [#uses=1]
	load ubyte* %2030		; <ubyte>:3209 [#uses=1]
	seteq ubyte %3209, 0		; <bool>:1370 [#uses=1]
	br bool %1370, label %1371, label %1370

; <label>:1371		; preds = %1369, %1370
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %893		; <ubyte*>:2031 [#uses=1]
	load ubyte* %2031		; <ubyte>:3210 [#uses=1]
	seteq ubyte %3210, 0		; <bool>:1371 [#uses=1]
	br bool %1371, label %1373, label %1372

; <label>:1372		; preds = %1371, %1372
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %893		; <ubyte*>:2032 [#uses=2]
	load ubyte* %2032		; <ubyte>:3211 [#uses=1]
	add ubyte %3211, 255		; <ubyte>:3212 [#uses=1]
	store ubyte %3212, ubyte* %2032
	add uint %893, 6		; <uint>:898 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %898		; <ubyte*>:2033 [#uses=2]
	load ubyte* %2033		; <ubyte>:3213 [#uses=1]
	add ubyte %3213, 1		; <ubyte>:3214 [#uses=1]
	store ubyte %3214, ubyte* %2033
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %893		; <ubyte*>:2034 [#uses=1]
	load ubyte* %2034		; <ubyte>:3215 [#uses=1]
	seteq ubyte %3215, 0		; <bool>:1372 [#uses=1]
	br bool %1372, label %1373, label %1372

; <label>:1373		; preds = %1371, %1372
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %893		; <ubyte*>:2035 [#uses=2]
	load ubyte* %2035		; <ubyte>:3216 [#uses=1]
	add ubyte %3216, 1		; <ubyte>:3217 [#uses=1]
	store ubyte %3217, ubyte* %2035
	add uint %893, 6		; <uint>:899 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %899		; <ubyte*>:2036 [#uses=2]
	load ubyte* %2036		; <ubyte>:3218 [#uses=2]
	add ubyte %3218, 255		; <ubyte>:3219 [#uses=1]
	store ubyte %3219, ubyte* %2036
	seteq ubyte %3218, 1		; <bool>:1373 [#uses=1]
	br bool %1373, label %1367, label %1366

; <label>:1374		; preds = %1367, %1374
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %896		; <ubyte*>:2037 [#uses=2]
	load ubyte* %2037		; <ubyte>:3220 [#uses=2]
	add ubyte %3220, 255		; <ubyte>:3221 [#uses=1]
	store ubyte %3221, ubyte* %2037
	seteq ubyte %3220, 1		; <bool>:1374 [#uses=1]
	br bool %1374, label %1375, label %1374

; <label>:1375		; preds = %1367, %1374
	add uint %895, 4294967286		; <uint>:900 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %900		; <ubyte*>:2038 [#uses=1]
	load ubyte* %2038		; <ubyte>:3222 [#uses=1]
	seteq ubyte %3222, 0		; <bool>:1375 [#uses=1]
	br bool %1375, label %1377, label %1376

; <label>:1376		; preds = %1375, %1376
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %900		; <ubyte*>:2039 [#uses=2]
	load ubyte* %2039		; <ubyte>:3223 [#uses=1]
	add ubyte %3223, 255		; <ubyte>:3224 [#uses=1]
	store ubyte %3224, ubyte* %2039
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %896		; <ubyte*>:2040 [#uses=2]
	load ubyte* %2040		; <ubyte>:3225 [#uses=1]
	add ubyte %3225, 1		; <ubyte>:3226 [#uses=1]
	store ubyte %3226, ubyte* %2040
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %900		; <ubyte*>:2041 [#uses=1]
	load ubyte* %2041		; <ubyte>:3227 [#uses=1]
	seteq ubyte %3227, 0		; <bool>:1376 [#uses=1]
	br bool %1376, label %1377, label %1376

; <label>:1377		; preds = %1375, %1376
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %895		; <ubyte*>:2042 [#uses=2]
	load ubyte* %2042		; <ubyte>:3228 [#uses=2]
	add ubyte %3228, 1		; <ubyte>:3229 [#uses=1]
	store ubyte %3229, ubyte* %2042
	seteq ubyte %3228, 255		; <bool>:1377 [#uses=1]
	br bool %1377, label %1379, label %1378

; <label>:1378		; preds = %1377, %1378
	phi uint [ %895, %1377 ], [ %902, %1378 ]		; <uint>:901 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %901		; <ubyte*>:2043 [#uses=2]
	load ubyte* %2043		; <ubyte>:3230 [#uses=1]
	add ubyte %3230, 255		; <ubyte>:3231 [#uses=1]
	store ubyte %3231, ubyte* %2043
	add uint %901, 4294967290		; <uint>:902 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %902		; <ubyte*>:2044 [#uses=1]
	load ubyte* %2044		; <ubyte>:3232 [#uses=1]
	seteq ubyte %3232, 0		; <bool>:1378 [#uses=1]
	br bool %1378, label %1379, label %1378

; <label>:1379		; preds = %1377, %1378
	phi uint [ %895, %1377 ], [ %902, %1378 ]		; <uint>:903 [#uses=2]
	add uint %903, 98		; <uint>:904 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %904		; <ubyte*>:2045 [#uses=2]
	load ubyte* %2045		; <ubyte>:3233 [#uses=1]
	add ubyte %3233, 5		; <ubyte>:3234 [#uses=1]
	store ubyte %3234, ubyte* %2045
	add uint %903, 100		; <uint>:905 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %905		; <ubyte*>:2046 [#uses=1]
	load ubyte* %2046		; <ubyte>:3235 [#uses=1]
	seteq ubyte %3235, 0		; <bool>:1379 [#uses=1]
	br bool %1379, label %1009, label %1008

; <label>:1380		; preds = %583, %1393
	phi uint [ %402, %583 ], [ %915, %1393 ]		; <uint>:906 [#uses=8]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %906		; <ubyte*>:2047 [#uses=2]
	load ubyte* %2047		; <ubyte>:3236 [#uses=1]
	add ubyte %3236, 255		; <ubyte>:3237 [#uses=1]
	store ubyte %3237, ubyte* %2047
	add uint %906, 10		; <uint>:907 [#uses=6]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %907		; <ubyte*>:2048 [#uses=1]
	load ubyte* %2048		; <ubyte>:3238 [#uses=1]
	seteq ubyte %3238, 0		; <bool>:1380 [#uses=1]
	br bool %1380, label %1383, label %1382

; <label>:1381		; preds = %583, %1393
	phi uint [ %402, %583 ], [ %915, %1393 ]		; <uint>:908 [#uses=1]
	add uint %908, 4294967295		; <uint>:909 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %909		; <ubyte*>:2049 [#uses=1]
	load ubyte* %2049		; <ubyte>:3239 [#uses=1]
	seteq ubyte %3239, 0		; <bool>:1381 [#uses=1]
	br bool %1381, label %581, label %580

; <label>:1382		; preds = %1380, %1382
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %907		; <ubyte*>:2050 [#uses=2]
	load ubyte* %2050		; <ubyte>:3240 [#uses=2]
	add ubyte %3240, 255		; <ubyte>:3241 [#uses=1]
	store ubyte %3241, ubyte* %2050
	seteq ubyte %3240, 1		; <bool>:1382 [#uses=1]
	br bool %1382, label %1383, label %1382

; <label>:1383		; preds = %1380, %1382
	add uint %906, 4		; <uint>:910 [#uses=7]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %910		; <ubyte*>:2051 [#uses=1]
	load ubyte* %2051		; <ubyte>:3242 [#uses=1]
	seteq ubyte %3242, 0		; <bool>:1383 [#uses=1]
	br bool %1383, label %1385, label %1384

; <label>:1384		; preds = %1383, %1384
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %910		; <ubyte*>:2052 [#uses=2]
	load ubyte* %2052		; <ubyte>:3243 [#uses=1]
	add ubyte %3243, 255		; <ubyte>:3244 [#uses=1]
	store ubyte %3244, ubyte* %2052
	add uint %906, 5		; <uint>:911 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %911		; <ubyte*>:2053 [#uses=2]
	load ubyte* %2053		; <ubyte>:3245 [#uses=1]
	add ubyte %3245, 1		; <ubyte>:3246 [#uses=1]
	store ubyte %3246, ubyte* %2053
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %907		; <ubyte*>:2054 [#uses=2]
	load ubyte* %2054		; <ubyte>:3247 [#uses=1]
	add ubyte %3247, 1		; <ubyte>:3248 [#uses=1]
	store ubyte %3248, ubyte* %2054
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %910		; <ubyte*>:2055 [#uses=1]
	load ubyte* %2055		; <ubyte>:3249 [#uses=1]
	seteq ubyte %3249, 0		; <bool>:1384 [#uses=1]
	br bool %1384, label %1385, label %1384

; <label>:1385		; preds = %1383, %1384
	add uint %906, 5		; <uint>:912 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %912		; <ubyte*>:2056 [#uses=1]
	load ubyte* %2056		; <ubyte>:3250 [#uses=1]
	seteq ubyte %3250, 0		; <bool>:1385 [#uses=1]
	br bool %1385, label %1387, label %1386

; <label>:1386		; preds = %1385, %1386
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %910		; <ubyte*>:2057 [#uses=2]
	load ubyte* %2057		; <ubyte>:3251 [#uses=1]
	add ubyte %3251, 1		; <ubyte>:3252 [#uses=1]
	store ubyte %3252, ubyte* %2057
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %912		; <ubyte*>:2058 [#uses=2]
	load ubyte* %2058		; <ubyte>:3253 [#uses=2]
	add ubyte %3253, 255		; <ubyte>:3254 [#uses=1]
	store ubyte %3254, ubyte* %2058
	seteq ubyte %3253, 1		; <bool>:1386 [#uses=1]
	br bool %1386, label %1387, label %1386

; <label>:1387		; preds = %1385, %1386
	add uint %906, 12		; <uint>:913 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %913		; <ubyte*>:2059 [#uses=2]
	load ubyte* %2059		; <ubyte>:3255 [#uses=2]
	add ubyte %3255, 1		; <ubyte>:3256 [#uses=1]
	store ubyte %3256, ubyte* %2059
	seteq ubyte %3255, 255		; <bool>:1387 [#uses=1]
	br bool %1387, label %1389, label %1388

; <label>:1388		; preds = %1387, %1388
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %907		; <ubyte*>:2060 [#uses=2]
	load ubyte* %2060		; <ubyte>:3257 [#uses=1]
	add ubyte %3257, 1		; <ubyte>:3258 [#uses=1]
	store ubyte %3258, ubyte* %2060
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %913		; <ubyte*>:2061 [#uses=2]
	load ubyte* %2061		; <ubyte>:3259 [#uses=2]
	add ubyte %3259, 255		; <ubyte>:3260 [#uses=1]
	store ubyte %3260, ubyte* %2061
	seteq ubyte %3259, 1		; <bool>:1388 [#uses=1]
	br bool %1388, label %1389, label %1388

; <label>:1389		; preds = %1387, %1388
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %910		; <ubyte*>:2062 [#uses=1]
	load ubyte* %2062		; <ubyte>:3261 [#uses=1]
	seteq ubyte %3261, 0		; <bool>:1389 [#uses=1]
	br bool %1389, label %1391, label %1390

; <label>:1390		; preds = %1389, %1390
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %910		; <ubyte*>:2063 [#uses=2]
	load ubyte* %2063		; <ubyte>:3262 [#uses=2]
	add ubyte %3262, 255		; <ubyte>:3263 [#uses=1]
	store ubyte %3263, ubyte* %2063
	seteq ubyte %3262, 1		; <bool>:1390 [#uses=1]
	br bool %1390, label %1391, label %1390

; <label>:1391		; preds = %1389, %1390
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %907		; <ubyte*>:2064 [#uses=1]
	load ubyte* %2064		; <ubyte>:3264 [#uses=1]
	seteq ubyte %3264, 0		; <bool>:1391 [#uses=1]
	br bool %1391, label %1393, label %1392

; <label>:1392		; preds = %1391, %1392
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %910		; <ubyte*>:2065 [#uses=2]
	load ubyte* %2065		; <ubyte>:3265 [#uses=1]
	add ubyte %3265, 1		; <ubyte>:3266 [#uses=1]
	store ubyte %3266, ubyte* %2065
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %907		; <ubyte*>:2066 [#uses=2]
	load ubyte* %2066		; <ubyte>:3267 [#uses=2]
	add ubyte %3267, 255		; <ubyte>:3268 [#uses=1]
	store ubyte %3268, ubyte* %2066
	seteq ubyte %3267, 1		; <bool>:1392 [#uses=1]
	br bool %1392, label %1393, label %1392

; <label>:1393		; preds = %1391, %1392
	add uint %906, 4294967295		; <uint>:914 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %914		; <ubyte*>:2067 [#uses=2]
	load ubyte* %2067		; <ubyte>:3269 [#uses=1]
	add ubyte %3269, 5		; <ubyte>:3270 [#uses=1]
	store ubyte %3270, ubyte* %2067
	add uint %906, 1		; <uint>:915 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %915		; <ubyte*>:2068 [#uses=1]
	load ubyte* %2068		; <ubyte>:3271 [#uses=1]
	seteq ubyte %3271, 0		; <bool>:1393 [#uses=1]
	br bool %1393, label %1381, label %1380

; <label>:1394		; preds = %581, %1394
	phi uint [ %399, %581 ], [ %918, %1394 ]		; <uint>:916 [#uses=3]
	add uint %916, 4294967295		; <uint>:917 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %917		; <ubyte*>:2069 [#uses=2]
	load ubyte* %2069		; <ubyte>:3272 [#uses=1]
	add ubyte %3272, 3		; <ubyte>:3273 [#uses=1]
	store ubyte %3273, ubyte* %2069
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %916		; <ubyte*>:2070 [#uses=2]
	load ubyte* %2070		; <ubyte>:3274 [#uses=1]
	add ubyte %3274, 255		; <ubyte>:3275 [#uses=1]
	store ubyte %3275, ubyte* %2070
	add uint %916, 1		; <uint>:918 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %918		; <ubyte*>:2071 [#uses=1]
	load ubyte* %2071		; <ubyte>:3276 [#uses=1]
	seteq ubyte %3276, 0		; <bool>:1394 [#uses=1]
	br bool %1394, label %1395, label %1394

; <label>:1395		; preds = %581, %1394
	phi uint [ %399, %581 ], [ %918, %1394 ]		; <uint>:919 [#uses=1]
	add uint %919, 4294967295		; <uint>:920 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %920		; <ubyte*>:2072 [#uses=1]
	load ubyte* %2072		; <ubyte>:3277 [#uses=1]
	seteq ubyte %3277, 0		; <bool>:1395 [#uses=1]
	br bool %1395, label %579, label %578

; <label>:1396		; preds = %579, %1589
	phi uint [ %396, %579 ], [ %1034, %1589 ]		; <uint>:921 [#uses=66]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %921		; <ubyte*>:2073 [#uses=2]
	load ubyte* %2073		; <ubyte>:3278 [#uses=1]
	add ubyte %3278, 255		; <ubyte>:3279 [#uses=1]
	store ubyte %3279, ubyte* %2073
	add uint %921, 18		; <uint>:922 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %922		; <ubyte*>:2074 [#uses=1]
	load ubyte* %2074		; <ubyte>:3280 [#uses=1]
	seteq ubyte %3280, 0		; <bool>:1396 [#uses=1]
	br bool %1396, label %1399, label %1398

; <label>:1397		; preds = %579, %1589
	phi uint [ %396, %579 ], [ %1034, %1589 ]		; <uint>:923 [#uses=1]
	add uint %923, 4294967295		; <uint>:924 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %924		; <ubyte*>:2075 [#uses=1]
	load ubyte* %2075		; <ubyte>:3281 [#uses=1]
	seteq ubyte %3281, 0		; <bool>:1397 [#uses=1]
	br bool %1397, label %577, label %576

; <label>:1398		; preds = %1396, %1398
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %922		; <ubyte*>:2076 [#uses=2]
	load ubyte* %2076		; <ubyte>:3282 [#uses=2]
	add ubyte %3282, 255		; <ubyte>:3283 [#uses=1]
	store ubyte %3283, ubyte* %2076
	seteq ubyte %3282, 1		; <bool>:1398 [#uses=1]
	br bool %1398, label %1399, label %1398

; <label>:1399		; preds = %1396, %1398
	add uint %921, 4294967201		; <uint>:925 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %925		; <ubyte*>:2077 [#uses=1]
	load ubyte* %2077		; <ubyte>:3284 [#uses=1]
	seteq ubyte %3284, 0		; <bool>:1399 [#uses=1]
	br bool %1399, label %1401, label %1400

; <label>:1400		; preds = %1399, %1400
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %925		; <ubyte*>:2078 [#uses=2]
	load ubyte* %2078		; <ubyte>:3285 [#uses=1]
	add ubyte %3285, 255		; <ubyte>:3286 [#uses=1]
	store ubyte %3286, ubyte* %2078
	add uint %921, 4294967202		; <uint>:926 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %926		; <ubyte*>:2079 [#uses=2]
	load ubyte* %2079		; <ubyte>:3287 [#uses=1]
	add ubyte %3287, 1		; <ubyte>:3288 [#uses=1]
	store ubyte %3288, ubyte* %2079
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %922		; <ubyte*>:2080 [#uses=2]
	load ubyte* %2080		; <ubyte>:3289 [#uses=1]
	add ubyte %3289, 1		; <ubyte>:3290 [#uses=1]
	store ubyte %3290, ubyte* %2080
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %925		; <ubyte*>:2081 [#uses=1]
	load ubyte* %2081		; <ubyte>:3291 [#uses=1]
	seteq ubyte %3291, 0		; <bool>:1400 [#uses=1]
	br bool %1400, label %1401, label %1400

; <label>:1401		; preds = %1399, %1400
	add uint %921, 4294967202		; <uint>:927 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %927		; <ubyte*>:2082 [#uses=1]
	load ubyte* %2082		; <ubyte>:3292 [#uses=1]
	seteq ubyte %3292, 0		; <bool>:1401 [#uses=1]
	br bool %1401, label %1403, label %1402

; <label>:1402		; preds = %1401, %1402
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %925		; <ubyte*>:2083 [#uses=2]
	load ubyte* %2083		; <ubyte>:3293 [#uses=1]
	add ubyte %3293, 1		; <ubyte>:3294 [#uses=1]
	store ubyte %3294, ubyte* %2083
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %927		; <ubyte*>:2084 [#uses=2]
	load ubyte* %2084		; <ubyte>:3295 [#uses=2]
	add ubyte %3295, 255		; <ubyte>:3296 [#uses=1]
	store ubyte %3296, ubyte* %2084
	seteq ubyte %3295, 1		; <bool>:1402 [#uses=1]
	br bool %1402, label %1403, label %1402

; <label>:1403		; preds = %1401, %1402
	add uint %921, 24		; <uint>:928 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %928		; <ubyte*>:2085 [#uses=1]
	load ubyte* %2085		; <ubyte>:3297 [#uses=1]
	seteq ubyte %3297, 0		; <bool>:1403 [#uses=1]
	br bool %1403, label %1405, label %1404

; <label>:1404		; preds = %1403, %1404
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %928		; <ubyte*>:2086 [#uses=2]
	load ubyte* %2086		; <ubyte>:3298 [#uses=2]
	add ubyte %3298, 255		; <ubyte>:3299 [#uses=1]
	store ubyte %3299, ubyte* %2086
	seteq ubyte %3298, 1		; <bool>:1404 [#uses=1]
	br bool %1404, label %1405, label %1404

; <label>:1405		; preds = %1403, %1404
	add uint %921, 4294967207		; <uint>:929 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %929		; <ubyte*>:2087 [#uses=1]
	load ubyte* %2087		; <ubyte>:3300 [#uses=1]
	seteq ubyte %3300, 0		; <bool>:1405 [#uses=1]
	br bool %1405, label %1407, label %1406

; <label>:1406		; preds = %1405, %1406
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %929		; <ubyte*>:2088 [#uses=2]
	load ubyte* %2088		; <ubyte>:3301 [#uses=1]
	add ubyte %3301, 255		; <ubyte>:3302 [#uses=1]
	store ubyte %3302, ubyte* %2088
	add uint %921, 4294967208		; <uint>:930 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %930		; <ubyte*>:2089 [#uses=2]
	load ubyte* %2089		; <ubyte>:3303 [#uses=1]
	add ubyte %3303, 1		; <ubyte>:3304 [#uses=1]
	store ubyte %3304, ubyte* %2089
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %928		; <ubyte*>:2090 [#uses=2]
	load ubyte* %2090		; <ubyte>:3305 [#uses=1]
	add ubyte %3305, 1		; <ubyte>:3306 [#uses=1]
	store ubyte %3306, ubyte* %2090
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %929		; <ubyte*>:2091 [#uses=1]
	load ubyte* %2091		; <ubyte>:3307 [#uses=1]
	seteq ubyte %3307, 0		; <bool>:1406 [#uses=1]
	br bool %1406, label %1407, label %1406

; <label>:1407		; preds = %1405, %1406
	add uint %921, 4294967208		; <uint>:931 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %931		; <ubyte*>:2092 [#uses=1]
	load ubyte* %2092		; <ubyte>:3308 [#uses=1]
	seteq ubyte %3308, 0		; <bool>:1407 [#uses=1]
	br bool %1407, label %1409, label %1408

; <label>:1408		; preds = %1407, %1408
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %929		; <ubyte*>:2093 [#uses=2]
	load ubyte* %2093		; <ubyte>:3309 [#uses=1]
	add ubyte %3309, 1		; <ubyte>:3310 [#uses=1]
	store ubyte %3310, ubyte* %2093
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %931		; <ubyte*>:2094 [#uses=2]
	load ubyte* %2094		; <ubyte>:3311 [#uses=2]
	add ubyte %3311, 255		; <ubyte>:3312 [#uses=1]
	store ubyte %3312, ubyte* %2094
	seteq ubyte %3311, 1		; <bool>:1408 [#uses=1]
	br bool %1408, label %1409, label %1408

; <label>:1409		; preds = %1407, %1408
	add uint %921, 30		; <uint>:932 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %932		; <ubyte*>:2095 [#uses=1]
	load ubyte* %2095		; <ubyte>:3313 [#uses=1]
	seteq ubyte %3313, 0		; <bool>:1409 [#uses=1]
	br bool %1409, label %1411, label %1410

; <label>:1410		; preds = %1409, %1410
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %932		; <ubyte*>:2096 [#uses=2]
	load ubyte* %2096		; <ubyte>:3314 [#uses=2]
	add ubyte %3314, 255		; <ubyte>:3315 [#uses=1]
	store ubyte %3315, ubyte* %2096
	seteq ubyte %3314, 1		; <bool>:1410 [#uses=1]
	br bool %1410, label %1411, label %1410

; <label>:1411		; preds = %1409, %1410
	add uint %921, 4294967213		; <uint>:933 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %933		; <ubyte*>:2097 [#uses=1]
	load ubyte* %2097		; <ubyte>:3316 [#uses=1]
	seteq ubyte %3316, 0		; <bool>:1411 [#uses=1]
	br bool %1411, label %1413, label %1412

; <label>:1412		; preds = %1411, %1412
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %933		; <ubyte*>:2098 [#uses=2]
	load ubyte* %2098		; <ubyte>:3317 [#uses=1]
	add ubyte %3317, 255		; <ubyte>:3318 [#uses=1]
	store ubyte %3318, ubyte* %2098
	add uint %921, 4294967214		; <uint>:934 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %934		; <ubyte*>:2099 [#uses=2]
	load ubyte* %2099		; <ubyte>:3319 [#uses=1]
	add ubyte %3319, 1		; <ubyte>:3320 [#uses=1]
	store ubyte %3320, ubyte* %2099
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %932		; <ubyte*>:2100 [#uses=2]
	load ubyte* %2100		; <ubyte>:3321 [#uses=1]
	add ubyte %3321, 1		; <ubyte>:3322 [#uses=1]
	store ubyte %3322, ubyte* %2100
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %933		; <ubyte*>:2101 [#uses=1]
	load ubyte* %2101		; <ubyte>:3323 [#uses=1]
	seteq ubyte %3323, 0		; <bool>:1412 [#uses=1]
	br bool %1412, label %1413, label %1412

; <label>:1413		; preds = %1411, %1412
	add uint %921, 4294967214		; <uint>:935 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %935		; <ubyte*>:2102 [#uses=1]
	load ubyte* %2102		; <ubyte>:3324 [#uses=1]
	seteq ubyte %3324, 0		; <bool>:1413 [#uses=1]
	br bool %1413, label %1415, label %1414

; <label>:1414		; preds = %1413, %1414
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %933		; <ubyte*>:2103 [#uses=2]
	load ubyte* %2103		; <ubyte>:3325 [#uses=1]
	add ubyte %3325, 1		; <ubyte>:3326 [#uses=1]
	store ubyte %3326, ubyte* %2103
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %935		; <ubyte*>:2104 [#uses=2]
	load ubyte* %2104		; <ubyte>:3327 [#uses=2]
	add ubyte %3327, 255		; <ubyte>:3328 [#uses=1]
	store ubyte %3328, ubyte* %2104
	seteq ubyte %3327, 1		; <bool>:1414 [#uses=1]
	br bool %1414, label %1415, label %1414

; <label>:1415		; preds = %1413, %1414
	add uint %921, 36		; <uint>:936 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %936		; <ubyte*>:2105 [#uses=1]
	load ubyte* %2105		; <ubyte>:3329 [#uses=1]
	seteq ubyte %3329, 0		; <bool>:1415 [#uses=1]
	br bool %1415, label %1417, label %1416

; <label>:1416		; preds = %1415, %1416
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %936		; <ubyte*>:2106 [#uses=2]
	load ubyte* %2106		; <ubyte>:3330 [#uses=2]
	add ubyte %3330, 255		; <ubyte>:3331 [#uses=1]
	store ubyte %3331, ubyte* %2106
	seteq ubyte %3330, 1		; <bool>:1416 [#uses=1]
	br bool %1416, label %1417, label %1416

; <label>:1417		; preds = %1415, %1416
	add uint %921, 4294967219		; <uint>:937 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %937		; <ubyte*>:2107 [#uses=1]
	load ubyte* %2107		; <ubyte>:3332 [#uses=1]
	seteq ubyte %3332, 0		; <bool>:1417 [#uses=1]
	br bool %1417, label %1419, label %1418

; <label>:1418		; preds = %1417, %1418
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %937		; <ubyte*>:2108 [#uses=2]
	load ubyte* %2108		; <ubyte>:3333 [#uses=1]
	add ubyte %3333, 255		; <ubyte>:3334 [#uses=1]
	store ubyte %3334, ubyte* %2108
	add uint %921, 4294967220		; <uint>:938 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %938		; <ubyte*>:2109 [#uses=2]
	load ubyte* %2109		; <ubyte>:3335 [#uses=1]
	add ubyte %3335, 1		; <ubyte>:3336 [#uses=1]
	store ubyte %3336, ubyte* %2109
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %936		; <ubyte*>:2110 [#uses=2]
	load ubyte* %2110		; <ubyte>:3337 [#uses=1]
	add ubyte %3337, 1		; <ubyte>:3338 [#uses=1]
	store ubyte %3338, ubyte* %2110
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %937		; <ubyte*>:2111 [#uses=1]
	load ubyte* %2111		; <ubyte>:3339 [#uses=1]
	seteq ubyte %3339, 0		; <bool>:1418 [#uses=1]
	br bool %1418, label %1419, label %1418

; <label>:1419		; preds = %1417, %1418
	add uint %921, 4294967220		; <uint>:939 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %939		; <ubyte*>:2112 [#uses=1]
	load ubyte* %2112		; <ubyte>:3340 [#uses=1]
	seteq ubyte %3340, 0		; <bool>:1419 [#uses=1]
	br bool %1419, label %1421, label %1420

; <label>:1420		; preds = %1419, %1420
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %937		; <ubyte*>:2113 [#uses=2]
	load ubyte* %2113		; <ubyte>:3341 [#uses=1]
	add ubyte %3341, 1		; <ubyte>:3342 [#uses=1]
	store ubyte %3342, ubyte* %2113
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %939		; <ubyte*>:2114 [#uses=2]
	load ubyte* %2114		; <ubyte>:3343 [#uses=2]
	add ubyte %3343, 255		; <ubyte>:3344 [#uses=1]
	store ubyte %3344, ubyte* %2114
	seteq ubyte %3343, 1		; <bool>:1420 [#uses=1]
	br bool %1420, label %1421, label %1420

; <label>:1421		; preds = %1419, %1420
	add uint %921, 42		; <uint>:940 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %940		; <ubyte*>:2115 [#uses=1]
	load ubyte* %2115		; <ubyte>:3345 [#uses=1]
	seteq ubyte %3345, 0		; <bool>:1421 [#uses=1]
	br bool %1421, label %1423, label %1422

; <label>:1422		; preds = %1421, %1422
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %940		; <ubyte*>:2116 [#uses=2]
	load ubyte* %2116		; <ubyte>:3346 [#uses=2]
	add ubyte %3346, 255		; <ubyte>:3347 [#uses=1]
	store ubyte %3347, ubyte* %2116
	seteq ubyte %3346, 1		; <bool>:1422 [#uses=1]
	br bool %1422, label %1423, label %1422

; <label>:1423		; preds = %1421, %1422
	add uint %921, 4294967225		; <uint>:941 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %941		; <ubyte*>:2117 [#uses=1]
	load ubyte* %2117		; <ubyte>:3348 [#uses=1]
	seteq ubyte %3348, 0		; <bool>:1423 [#uses=1]
	br bool %1423, label %1425, label %1424

; <label>:1424		; preds = %1423, %1424
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %941		; <ubyte*>:2118 [#uses=2]
	load ubyte* %2118		; <ubyte>:3349 [#uses=1]
	add ubyte %3349, 255		; <ubyte>:3350 [#uses=1]
	store ubyte %3350, ubyte* %2118
	add uint %921, 4294967226		; <uint>:942 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %942		; <ubyte*>:2119 [#uses=2]
	load ubyte* %2119		; <ubyte>:3351 [#uses=1]
	add ubyte %3351, 1		; <ubyte>:3352 [#uses=1]
	store ubyte %3352, ubyte* %2119
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %940		; <ubyte*>:2120 [#uses=2]
	load ubyte* %2120		; <ubyte>:3353 [#uses=1]
	add ubyte %3353, 1		; <ubyte>:3354 [#uses=1]
	store ubyte %3354, ubyte* %2120
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %941		; <ubyte*>:2121 [#uses=1]
	load ubyte* %2121		; <ubyte>:3355 [#uses=1]
	seteq ubyte %3355, 0		; <bool>:1424 [#uses=1]
	br bool %1424, label %1425, label %1424

; <label>:1425		; preds = %1423, %1424
	add uint %921, 4294967226		; <uint>:943 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %943		; <ubyte*>:2122 [#uses=1]
	load ubyte* %2122		; <ubyte>:3356 [#uses=1]
	seteq ubyte %3356, 0		; <bool>:1425 [#uses=1]
	br bool %1425, label %1427, label %1426

; <label>:1426		; preds = %1425, %1426
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %941		; <ubyte*>:2123 [#uses=2]
	load ubyte* %2123		; <ubyte>:3357 [#uses=1]
	add ubyte %3357, 1		; <ubyte>:3358 [#uses=1]
	store ubyte %3358, ubyte* %2123
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %943		; <ubyte*>:2124 [#uses=2]
	load ubyte* %2124		; <ubyte>:3359 [#uses=2]
	add ubyte %3359, 255		; <ubyte>:3360 [#uses=1]
	store ubyte %3360, ubyte* %2124
	seteq ubyte %3359, 1		; <bool>:1426 [#uses=1]
	br bool %1426, label %1427, label %1426

; <label>:1427		; preds = %1425, %1426
	add uint %921, 48		; <uint>:944 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %944		; <ubyte*>:2125 [#uses=1]
	load ubyte* %2125		; <ubyte>:3361 [#uses=1]
	seteq ubyte %3361, 0		; <bool>:1427 [#uses=1]
	br bool %1427, label %1429, label %1428

; <label>:1428		; preds = %1427, %1428
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %944		; <ubyte*>:2126 [#uses=2]
	load ubyte* %2126		; <ubyte>:3362 [#uses=2]
	add ubyte %3362, 255		; <ubyte>:3363 [#uses=1]
	store ubyte %3363, ubyte* %2126
	seteq ubyte %3362, 1		; <bool>:1428 [#uses=1]
	br bool %1428, label %1429, label %1428

; <label>:1429		; preds = %1427, %1428
	add uint %921, 4294967231		; <uint>:945 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %945		; <ubyte*>:2127 [#uses=1]
	load ubyte* %2127		; <ubyte>:3364 [#uses=1]
	seteq ubyte %3364, 0		; <bool>:1429 [#uses=1]
	br bool %1429, label %1431, label %1430

; <label>:1430		; preds = %1429, %1430
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %945		; <ubyte*>:2128 [#uses=2]
	load ubyte* %2128		; <ubyte>:3365 [#uses=1]
	add ubyte %3365, 255		; <ubyte>:3366 [#uses=1]
	store ubyte %3366, ubyte* %2128
	add uint %921, 4294967232		; <uint>:946 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %946		; <ubyte*>:2129 [#uses=2]
	load ubyte* %2129		; <ubyte>:3367 [#uses=1]
	add ubyte %3367, 1		; <ubyte>:3368 [#uses=1]
	store ubyte %3368, ubyte* %2129
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %944		; <ubyte*>:2130 [#uses=2]
	load ubyte* %2130		; <ubyte>:3369 [#uses=1]
	add ubyte %3369, 1		; <ubyte>:3370 [#uses=1]
	store ubyte %3370, ubyte* %2130
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %945		; <ubyte*>:2131 [#uses=1]
	load ubyte* %2131		; <ubyte>:3371 [#uses=1]
	seteq ubyte %3371, 0		; <bool>:1430 [#uses=1]
	br bool %1430, label %1431, label %1430

; <label>:1431		; preds = %1429, %1430
	add uint %921, 4294967232		; <uint>:947 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %947		; <ubyte*>:2132 [#uses=1]
	load ubyte* %2132		; <ubyte>:3372 [#uses=1]
	seteq ubyte %3372, 0		; <bool>:1431 [#uses=1]
	br bool %1431, label %1433, label %1432

; <label>:1432		; preds = %1431, %1432
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %945		; <ubyte*>:2133 [#uses=2]
	load ubyte* %2133		; <ubyte>:3373 [#uses=1]
	add ubyte %3373, 1		; <ubyte>:3374 [#uses=1]
	store ubyte %3374, ubyte* %2133
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %947		; <ubyte*>:2134 [#uses=2]
	load ubyte* %2134		; <ubyte>:3375 [#uses=2]
	add ubyte %3375, 255		; <ubyte>:3376 [#uses=1]
	store ubyte %3376, ubyte* %2134
	seteq ubyte %3375, 1		; <bool>:1432 [#uses=1]
	br bool %1432, label %1433, label %1432

; <label>:1433		; preds = %1431, %1432
	add uint %921, 54		; <uint>:948 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %948		; <ubyte*>:2135 [#uses=1]
	load ubyte* %2135		; <ubyte>:3377 [#uses=1]
	seteq ubyte %3377, 0		; <bool>:1433 [#uses=1]
	br bool %1433, label %1435, label %1434

; <label>:1434		; preds = %1433, %1434
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %948		; <ubyte*>:2136 [#uses=2]
	load ubyte* %2136		; <ubyte>:3378 [#uses=2]
	add ubyte %3378, 255		; <ubyte>:3379 [#uses=1]
	store ubyte %3379, ubyte* %2136
	seteq ubyte %3378, 1		; <bool>:1434 [#uses=1]
	br bool %1434, label %1435, label %1434

; <label>:1435		; preds = %1433, %1434
	add uint %921, 4294967237		; <uint>:949 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %949		; <ubyte*>:2137 [#uses=1]
	load ubyte* %2137		; <ubyte>:3380 [#uses=1]
	seteq ubyte %3380, 0		; <bool>:1435 [#uses=1]
	br bool %1435, label %1437, label %1436

; <label>:1436		; preds = %1435, %1436
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %949		; <ubyte*>:2138 [#uses=2]
	load ubyte* %2138		; <ubyte>:3381 [#uses=1]
	add ubyte %3381, 255		; <ubyte>:3382 [#uses=1]
	store ubyte %3382, ubyte* %2138
	add uint %921, 4294967238		; <uint>:950 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %950		; <ubyte*>:2139 [#uses=2]
	load ubyte* %2139		; <ubyte>:3383 [#uses=1]
	add ubyte %3383, 1		; <ubyte>:3384 [#uses=1]
	store ubyte %3384, ubyte* %2139
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %948		; <ubyte*>:2140 [#uses=2]
	load ubyte* %2140		; <ubyte>:3385 [#uses=1]
	add ubyte %3385, 1		; <ubyte>:3386 [#uses=1]
	store ubyte %3386, ubyte* %2140
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %949		; <ubyte*>:2141 [#uses=1]
	load ubyte* %2141		; <ubyte>:3387 [#uses=1]
	seteq ubyte %3387, 0		; <bool>:1436 [#uses=1]
	br bool %1436, label %1437, label %1436

; <label>:1437		; preds = %1435, %1436
	add uint %921, 4294967238		; <uint>:951 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %951		; <ubyte*>:2142 [#uses=1]
	load ubyte* %2142		; <ubyte>:3388 [#uses=1]
	seteq ubyte %3388, 0		; <bool>:1437 [#uses=1]
	br bool %1437, label %1439, label %1438

; <label>:1438		; preds = %1437, %1438
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %949		; <ubyte*>:2143 [#uses=2]
	load ubyte* %2143		; <ubyte>:3389 [#uses=1]
	add ubyte %3389, 1		; <ubyte>:3390 [#uses=1]
	store ubyte %3390, ubyte* %2143
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %951		; <ubyte*>:2144 [#uses=2]
	load ubyte* %2144		; <ubyte>:3391 [#uses=2]
	add ubyte %3391, 255		; <ubyte>:3392 [#uses=1]
	store ubyte %3392, ubyte* %2144
	seteq ubyte %3391, 1		; <bool>:1438 [#uses=1]
	br bool %1438, label %1439, label %1438

; <label>:1439		; preds = %1437, %1438
	add uint %921, 60		; <uint>:952 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %952		; <ubyte*>:2145 [#uses=1]
	load ubyte* %2145		; <ubyte>:3393 [#uses=1]
	seteq ubyte %3393, 0		; <bool>:1439 [#uses=1]
	br bool %1439, label %1441, label %1440

; <label>:1440		; preds = %1439, %1440
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %952		; <ubyte*>:2146 [#uses=2]
	load ubyte* %2146		; <ubyte>:3394 [#uses=2]
	add ubyte %3394, 255		; <ubyte>:3395 [#uses=1]
	store ubyte %3395, ubyte* %2146
	seteq ubyte %3394, 1		; <bool>:1440 [#uses=1]
	br bool %1440, label %1441, label %1440

; <label>:1441		; preds = %1439, %1440
	add uint %921, 4294967243		; <uint>:953 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %953		; <ubyte*>:2147 [#uses=1]
	load ubyte* %2147		; <ubyte>:3396 [#uses=1]
	seteq ubyte %3396, 0		; <bool>:1441 [#uses=1]
	br bool %1441, label %1443, label %1442

; <label>:1442		; preds = %1441, %1442
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %953		; <ubyte*>:2148 [#uses=2]
	load ubyte* %2148		; <ubyte>:3397 [#uses=1]
	add ubyte %3397, 255		; <ubyte>:3398 [#uses=1]
	store ubyte %3398, ubyte* %2148
	add uint %921, 4294967244		; <uint>:954 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %954		; <ubyte*>:2149 [#uses=2]
	load ubyte* %2149		; <ubyte>:3399 [#uses=1]
	add ubyte %3399, 1		; <ubyte>:3400 [#uses=1]
	store ubyte %3400, ubyte* %2149
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %952		; <ubyte*>:2150 [#uses=2]
	load ubyte* %2150		; <ubyte>:3401 [#uses=1]
	add ubyte %3401, 1		; <ubyte>:3402 [#uses=1]
	store ubyte %3402, ubyte* %2150
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %953		; <ubyte*>:2151 [#uses=1]
	load ubyte* %2151		; <ubyte>:3403 [#uses=1]
	seteq ubyte %3403, 0		; <bool>:1442 [#uses=1]
	br bool %1442, label %1443, label %1442

; <label>:1443		; preds = %1441, %1442
	add uint %921, 4294967244		; <uint>:955 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %955		; <ubyte*>:2152 [#uses=1]
	load ubyte* %2152		; <ubyte>:3404 [#uses=1]
	seteq ubyte %3404, 0		; <bool>:1443 [#uses=1]
	br bool %1443, label %1445, label %1444

; <label>:1444		; preds = %1443, %1444
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %953		; <ubyte*>:2153 [#uses=2]
	load ubyte* %2153		; <ubyte>:3405 [#uses=1]
	add ubyte %3405, 1		; <ubyte>:3406 [#uses=1]
	store ubyte %3406, ubyte* %2153
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %955		; <ubyte*>:2154 [#uses=2]
	load ubyte* %2154		; <ubyte>:3407 [#uses=2]
	add ubyte %3407, 255		; <ubyte>:3408 [#uses=1]
	store ubyte %3408, ubyte* %2154
	seteq ubyte %3407, 1		; <bool>:1444 [#uses=1]
	br bool %1444, label %1445, label %1444

; <label>:1445		; preds = %1443, %1444
	add uint %921, 66		; <uint>:956 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %956		; <ubyte*>:2155 [#uses=1]
	load ubyte* %2155		; <ubyte>:3409 [#uses=1]
	seteq ubyte %3409, 0		; <bool>:1445 [#uses=1]
	br bool %1445, label %1447, label %1446

; <label>:1446		; preds = %1445, %1446
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %956		; <ubyte*>:2156 [#uses=2]
	load ubyte* %2156		; <ubyte>:3410 [#uses=2]
	add ubyte %3410, 255		; <ubyte>:3411 [#uses=1]
	store ubyte %3411, ubyte* %2156
	seteq ubyte %3410, 1		; <bool>:1446 [#uses=1]
	br bool %1446, label %1447, label %1446

; <label>:1447		; preds = %1445, %1446
	add uint %921, 4294967249		; <uint>:957 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %957		; <ubyte*>:2157 [#uses=1]
	load ubyte* %2157		; <ubyte>:3412 [#uses=1]
	seteq ubyte %3412, 0		; <bool>:1447 [#uses=1]
	br bool %1447, label %1449, label %1448

; <label>:1448		; preds = %1447, %1448
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %957		; <ubyte*>:2158 [#uses=2]
	load ubyte* %2158		; <ubyte>:3413 [#uses=1]
	add ubyte %3413, 255		; <ubyte>:3414 [#uses=1]
	store ubyte %3414, ubyte* %2158
	add uint %921, 4294967250		; <uint>:958 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %958		; <ubyte*>:2159 [#uses=2]
	load ubyte* %2159		; <ubyte>:3415 [#uses=1]
	add ubyte %3415, 1		; <ubyte>:3416 [#uses=1]
	store ubyte %3416, ubyte* %2159
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %956		; <ubyte*>:2160 [#uses=2]
	load ubyte* %2160		; <ubyte>:3417 [#uses=1]
	add ubyte %3417, 1		; <ubyte>:3418 [#uses=1]
	store ubyte %3418, ubyte* %2160
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %957		; <ubyte*>:2161 [#uses=1]
	load ubyte* %2161		; <ubyte>:3419 [#uses=1]
	seteq ubyte %3419, 0		; <bool>:1448 [#uses=1]
	br bool %1448, label %1449, label %1448

; <label>:1449		; preds = %1447, %1448
	add uint %921, 4294967250		; <uint>:959 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %959		; <ubyte*>:2162 [#uses=1]
	load ubyte* %2162		; <ubyte>:3420 [#uses=1]
	seteq ubyte %3420, 0		; <bool>:1449 [#uses=1]
	br bool %1449, label %1451, label %1450

; <label>:1450		; preds = %1449, %1450
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %957		; <ubyte*>:2163 [#uses=2]
	load ubyte* %2163		; <ubyte>:3421 [#uses=1]
	add ubyte %3421, 1		; <ubyte>:3422 [#uses=1]
	store ubyte %3422, ubyte* %2163
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %959		; <ubyte*>:2164 [#uses=2]
	load ubyte* %2164		; <ubyte>:3423 [#uses=2]
	add ubyte %3423, 255		; <ubyte>:3424 [#uses=1]
	store ubyte %3424, ubyte* %2164
	seteq ubyte %3423, 1		; <bool>:1450 [#uses=1]
	br bool %1450, label %1451, label %1450

; <label>:1451		; preds = %1449, %1450
	add uint %921, 72		; <uint>:960 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %960		; <ubyte*>:2165 [#uses=1]
	load ubyte* %2165		; <ubyte>:3425 [#uses=1]
	seteq ubyte %3425, 0		; <bool>:1451 [#uses=1]
	br bool %1451, label %1453, label %1452

; <label>:1452		; preds = %1451, %1452
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %960		; <ubyte*>:2166 [#uses=2]
	load ubyte* %2166		; <ubyte>:3426 [#uses=2]
	add ubyte %3426, 255		; <ubyte>:3427 [#uses=1]
	store ubyte %3427, ubyte* %2166
	seteq ubyte %3426, 1		; <bool>:1452 [#uses=1]
	br bool %1452, label %1453, label %1452

; <label>:1453		; preds = %1451, %1452
	add uint %921, 4294967255		; <uint>:961 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %961		; <ubyte*>:2167 [#uses=1]
	load ubyte* %2167		; <ubyte>:3428 [#uses=1]
	seteq ubyte %3428, 0		; <bool>:1453 [#uses=1]
	br bool %1453, label %1455, label %1454

; <label>:1454		; preds = %1453, %1454
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %961		; <ubyte*>:2168 [#uses=2]
	load ubyte* %2168		; <ubyte>:3429 [#uses=1]
	add ubyte %3429, 255		; <ubyte>:3430 [#uses=1]
	store ubyte %3430, ubyte* %2168
	add uint %921, 4294967256		; <uint>:962 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %962		; <ubyte*>:2169 [#uses=2]
	load ubyte* %2169		; <ubyte>:3431 [#uses=1]
	add ubyte %3431, 1		; <ubyte>:3432 [#uses=1]
	store ubyte %3432, ubyte* %2169
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %960		; <ubyte*>:2170 [#uses=2]
	load ubyte* %2170		; <ubyte>:3433 [#uses=1]
	add ubyte %3433, 1		; <ubyte>:3434 [#uses=1]
	store ubyte %3434, ubyte* %2170
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %961		; <ubyte*>:2171 [#uses=1]
	load ubyte* %2171		; <ubyte>:3435 [#uses=1]
	seteq ubyte %3435, 0		; <bool>:1454 [#uses=1]
	br bool %1454, label %1455, label %1454

; <label>:1455		; preds = %1453, %1454
	add uint %921, 4294967256		; <uint>:963 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %963		; <ubyte*>:2172 [#uses=1]
	load ubyte* %2172		; <ubyte>:3436 [#uses=1]
	seteq ubyte %3436, 0		; <bool>:1455 [#uses=1]
	br bool %1455, label %1457, label %1456

; <label>:1456		; preds = %1455, %1456
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %961		; <ubyte*>:2173 [#uses=2]
	load ubyte* %2173		; <ubyte>:3437 [#uses=1]
	add ubyte %3437, 1		; <ubyte>:3438 [#uses=1]
	store ubyte %3438, ubyte* %2173
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %963		; <ubyte*>:2174 [#uses=2]
	load ubyte* %2174		; <ubyte>:3439 [#uses=2]
	add ubyte %3439, 255		; <ubyte>:3440 [#uses=1]
	store ubyte %3440, ubyte* %2174
	seteq ubyte %3439, 1		; <bool>:1456 [#uses=1]
	br bool %1456, label %1457, label %1456

; <label>:1457		; preds = %1455, %1456
	add uint %921, 78		; <uint>:964 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %964		; <ubyte*>:2175 [#uses=1]
	load ubyte* %2175		; <ubyte>:3441 [#uses=1]
	seteq ubyte %3441, 0		; <bool>:1457 [#uses=1]
	br bool %1457, label %1459, label %1458

; <label>:1458		; preds = %1457, %1458
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %964		; <ubyte*>:2176 [#uses=2]
	load ubyte* %2176		; <ubyte>:3442 [#uses=2]
	add ubyte %3442, 255		; <ubyte>:3443 [#uses=1]
	store ubyte %3443, ubyte* %2176
	seteq ubyte %3442, 1		; <bool>:1458 [#uses=1]
	br bool %1458, label %1459, label %1458

; <label>:1459		; preds = %1457, %1458
	add uint %921, 4294967261		; <uint>:965 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %965		; <ubyte*>:2177 [#uses=1]
	load ubyte* %2177		; <ubyte>:3444 [#uses=1]
	seteq ubyte %3444, 0		; <bool>:1459 [#uses=1]
	br bool %1459, label %1461, label %1460

; <label>:1460		; preds = %1459, %1460
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %965		; <ubyte*>:2178 [#uses=2]
	load ubyte* %2178		; <ubyte>:3445 [#uses=1]
	add ubyte %3445, 255		; <ubyte>:3446 [#uses=1]
	store ubyte %3446, ubyte* %2178
	add uint %921, 4294967262		; <uint>:966 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %966		; <ubyte*>:2179 [#uses=2]
	load ubyte* %2179		; <ubyte>:3447 [#uses=1]
	add ubyte %3447, 1		; <ubyte>:3448 [#uses=1]
	store ubyte %3448, ubyte* %2179
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %964		; <ubyte*>:2180 [#uses=2]
	load ubyte* %2180		; <ubyte>:3449 [#uses=1]
	add ubyte %3449, 1		; <ubyte>:3450 [#uses=1]
	store ubyte %3450, ubyte* %2180
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %965		; <ubyte*>:2181 [#uses=1]
	load ubyte* %2181		; <ubyte>:3451 [#uses=1]
	seteq ubyte %3451, 0		; <bool>:1460 [#uses=1]
	br bool %1460, label %1461, label %1460

; <label>:1461		; preds = %1459, %1460
	add uint %921, 4294967262		; <uint>:967 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %967		; <ubyte*>:2182 [#uses=1]
	load ubyte* %2182		; <ubyte>:3452 [#uses=1]
	seteq ubyte %3452, 0		; <bool>:1461 [#uses=1]
	br bool %1461, label %1463, label %1462

; <label>:1462		; preds = %1461, %1462
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %965		; <ubyte*>:2183 [#uses=2]
	load ubyte* %2183		; <ubyte>:3453 [#uses=1]
	add ubyte %3453, 1		; <ubyte>:3454 [#uses=1]
	store ubyte %3454, ubyte* %2183
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %967		; <ubyte*>:2184 [#uses=2]
	load ubyte* %2184		; <ubyte>:3455 [#uses=2]
	add ubyte %3455, 255		; <ubyte>:3456 [#uses=1]
	store ubyte %3456, ubyte* %2184
	seteq ubyte %3455, 1		; <bool>:1462 [#uses=1]
	br bool %1462, label %1463, label %1462

; <label>:1463		; preds = %1461, %1462
	add uint %921, 84		; <uint>:968 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %968		; <ubyte*>:2185 [#uses=1]
	load ubyte* %2185		; <ubyte>:3457 [#uses=1]
	seteq ubyte %3457, 0		; <bool>:1463 [#uses=1]
	br bool %1463, label %1465, label %1464

; <label>:1464		; preds = %1463, %1464
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %968		; <ubyte*>:2186 [#uses=2]
	load ubyte* %2186		; <ubyte>:3458 [#uses=2]
	add ubyte %3458, 255		; <ubyte>:3459 [#uses=1]
	store ubyte %3459, ubyte* %2186
	seteq ubyte %3458, 1		; <bool>:1464 [#uses=1]
	br bool %1464, label %1465, label %1464

; <label>:1465		; preds = %1463, %1464
	add uint %921, 4294967267		; <uint>:969 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %969		; <ubyte*>:2187 [#uses=1]
	load ubyte* %2187		; <ubyte>:3460 [#uses=1]
	seteq ubyte %3460, 0		; <bool>:1465 [#uses=1]
	br bool %1465, label %1467, label %1466

; <label>:1466		; preds = %1465, %1466
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %969		; <ubyte*>:2188 [#uses=2]
	load ubyte* %2188		; <ubyte>:3461 [#uses=1]
	add ubyte %3461, 255		; <ubyte>:3462 [#uses=1]
	store ubyte %3462, ubyte* %2188
	add uint %921, 4294967268		; <uint>:970 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %970		; <ubyte*>:2189 [#uses=2]
	load ubyte* %2189		; <ubyte>:3463 [#uses=1]
	add ubyte %3463, 1		; <ubyte>:3464 [#uses=1]
	store ubyte %3464, ubyte* %2189
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %968		; <ubyte*>:2190 [#uses=2]
	load ubyte* %2190		; <ubyte>:3465 [#uses=1]
	add ubyte %3465, 1		; <ubyte>:3466 [#uses=1]
	store ubyte %3466, ubyte* %2190
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %969		; <ubyte*>:2191 [#uses=1]
	load ubyte* %2191		; <ubyte>:3467 [#uses=1]
	seteq ubyte %3467, 0		; <bool>:1466 [#uses=1]
	br bool %1466, label %1467, label %1466

; <label>:1467		; preds = %1465, %1466
	add uint %921, 4294967268		; <uint>:971 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %971		; <ubyte*>:2192 [#uses=1]
	load ubyte* %2192		; <ubyte>:3468 [#uses=1]
	seteq ubyte %3468, 0		; <bool>:1467 [#uses=1]
	br bool %1467, label %1469, label %1468

; <label>:1468		; preds = %1467, %1468
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %969		; <ubyte*>:2193 [#uses=2]
	load ubyte* %2193		; <ubyte>:3469 [#uses=1]
	add ubyte %3469, 1		; <ubyte>:3470 [#uses=1]
	store ubyte %3470, ubyte* %2193
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %971		; <ubyte*>:2194 [#uses=2]
	load ubyte* %2194		; <ubyte>:3471 [#uses=2]
	add ubyte %3471, 255		; <ubyte>:3472 [#uses=1]
	store ubyte %3472, ubyte* %2194
	seteq ubyte %3471, 1		; <bool>:1468 [#uses=1]
	br bool %1468, label %1469, label %1468

; <label>:1469		; preds = %1467, %1468
	add uint %921, 90		; <uint>:972 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %972		; <ubyte*>:2195 [#uses=1]
	load ubyte* %2195		; <ubyte>:3473 [#uses=1]
	seteq ubyte %3473, 0		; <bool>:1469 [#uses=1]
	br bool %1469, label %1471, label %1470

; <label>:1470		; preds = %1469, %1470
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %972		; <ubyte*>:2196 [#uses=2]
	load ubyte* %2196		; <ubyte>:3474 [#uses=2]
	add ubyte %3474, 255		; <ubyte>:3475 [#uses=1]
	store ubyte %3475, ubyte* %2196
	seteq ubyte %3474, 1		; <bool>:1470 [#uses=1]
	br bool %1470, label %1471, label %1470

; <label>:1471		; preds = %1469, %1470
	add uint %921, 4294967273		; <uint>:973 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %973		; <ubyte*>:2197 [#uses=1]
	load ubyte* %2197		; <ubyte>:3476 [#uses=1]
	seteq ubyte %3476, 0		; <bool>:1471 [#uses=1]
	br bool %1471, label %1473, label %1472

; <label>:1472		; preds = %1471, %1472
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %973		; <ubyte*>:2198 [#uses=2]
	load ubyte* %2198		; <ubyte>:3477 [#uses=1]
	add ubyte %3477, 255		; <ubyte>:3478 [#uses=1]
	store ubyte %3478, ubyte* %2198
	add uint %921, 4294967274		; <uint>:974 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %974		; <ubyte*>:2199 [#uses=2]
	load ubyte* %2199		; <ubyte>:3479 [#uses=1]
	add ubyte %3479, 1		; <ubyte>:3480 [#uses=1]
	store ubyte %3480, ubyte* %2199
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %972		; <ubyte*>:2200 [#uses=2]
	load ubyte* %2200		; <ubyte>:3481 [#uses=1]
	add ubyte %3481, 1		; <ubyte>:3482 [#uses=1]
	store ubyte %3482, ubyte* %2200
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %973		; <ubyte*>:2201 [#uses=1]
	load ubyte* %2201		; <ubyte>:3483 [#uses=1]
	seteq ubyte %3483, 0		; <bool>:1472 [#uses=1]
	br bool %1472, label %1473, label %1472

; <label>:1473		; preds = %1471, %1472
	add uint %921, 4294967274		; <uint>:975 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %975		; <ubyte*>:2202 [#uses=1]
	load ubyte* %2202		; <ubyte>:3484 [#uses=1]
	seteq ubyte %3484, 0		; <bool>:1473 [#uses=1]
	br bool %1473, label %1475, label %1474

; <label>:1474		; preds = %1473, %1474
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %973		; <ubyte*>:2203 [#uses=2]
	load ubyte* %2203		; <ubyte>:3485 [#uses=1]
	add ubyte %3485, 1		; <ubyte>:3486 [#uses=1]
	store ubyte %3486, ubyte* %2203
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %975		; <ubyte*>:2204 [#uses=2]
	load ubyte* %2204		; <ubyte>:3487 [#uses=2]
	add ubyte %3487, 255		; <ubyte>:3488 [#uses=1]
	store ubyte %3488, ubyte* %2204
	seteq ubyte %3487, 1		; <bool>:1474 [#uses=1]
	br bool %1474, label %1475, label %1474

; <label>:1475		; preds = %1473, %1474
	add uint %921, 96		; <uint>:976 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %976		; <ubyte*>:2205 [#uses=1]
	load ubyte* %2205		; <ubyte>:3489 [#uses=1]
	seteq ubyte %3489, 0		; <bool>:1475 [#uses=1]
	br bool %1475, label %1477, label %1476

; <label>:1476		; preds = %1475, %1476
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %976		; <ubyte*>:2206 [#uses=2]
	load ubyte* %2206		; <ubyte>:3490 [#uses=2]
	add ubyte %3490, 255		; <ubyte>:3491 [#uses=1]
	store ubyte %3491, ubyte* %2206
	seteq ubyte %3490, 1		; <bool>:1476 [#uses=1]
	br bool %1476, label %1477, label %1476

; <label>:1477		; preds = %1475, %1476
	add uint %921, 4294967279		; <uint>:977 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %977		; <ubyte*>:2207 [#uses=1]
	load ubyte* %2207		; <ubyte>:3492 [#uses=1]
	seteq ubyte %3492, 0		; <bool>:1477 [#uses=1]
	br bool %1477, label %1479, label %1478

; <label>:1478		; preds = %1477, %1478
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %977		; <ubyte*>:2208 [#uses=2]
	load ubyte* %2208		; <ubyte>:3493 [#uses=1]
	add ubyte %3493, 255		; <ubyte>:3494 [#uses=1]
	store ubyte %3494, ubyte* %2208
	add uint %921, 4294967280		; <uint>:978 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %978		; <ubyte*>:2209 [#uses=2]
	load ubyte* %2209		; <ubyte>:3495 [#uses=1]
	add ubyte %3495, 1		; <ubyte>:3496 [#uses=1]
	store ubyte %3496, ubyte* %2209
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %976		; <ubyte*>:2210 [#uses=2]
	load ubyte* %2210		; <ubyte>:3497 [#uses=1]
	add ubyte %3497, 1		; <ubyte>:3498 [#uses=1]
	store ubyte %3498, ubyte* %2210
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %977		; <ubyte*>:2211 [#uses=1]
	load ubyte* %2211		; <ubyte>:3499 [#uses=1]
	seteq ubyte %3499, 0		; <bool>:1478 [#uses=1]
	br bool %1478, label %1479, label %1478

; <label>:1479		; preds = %1477, %1478
	add uint %921, 4294967280		; <uint>:979 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %979		; <ubyte*>:2212 [#uses=1]
	load ubyte* %2212		; <ubyte>:3500 [#uses=1]
	seteq ubyte %3500, 0		; <bool>:1479 [#uses=1]
	br bool %1479, label %1481, label %1480

; <label>:1480		; preds = %1479, %1480
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %977		; <ubyte*>:2213 [#uses=2]
	load ubyte* %2213		; <ubyte>:3501 [#uses=1]
	add ubyte %3501, 1		; <ubyte>:3502 [#uses=1]
	store ubyte %3502, ubyte* %2213
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %979		; <ubyte*>:2214 [#uses=2]
	load ubyte* %2214		; <ubyte>:3503 [#uses=2]
	add ubyte %3503, 255		; <ubyte>:3504 [#uses=1]
	store ubyte %3504, ubyte* %2214
	seteq ubyte %3503, 1		; <bool>:1480 [#uses=1]
	br bool %1480, label %1481, label %1480

; <label>:1481		; preds = %1479, %1480
	add uint %921, 102		; <uint>:980 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %980		; <ubyte*>:2215 [#uses=1]
	load ubyte* %2215		; <ubyte>:3505 [#uses=1]
	seteq ubyte %3505, 0		; <bool>:1481 [#uses=1]
	br bool %1481, label %1483, label %1482

; <label>:1482		; preds = %1481, %1482
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %980		; <ubyte*>:2216 [#uses=2]
	load ubyte* %2216		; <ubyte>:3506 [#uses=2]
	add ubyte %3506, 255		; <ubyte>:3507 [#uses=1]
	store ubyte %3507, ubyte* %2216
	seteq ubyte %3506, 1		; <bool>:1482 [#uses=1]
	br bool %1482, label %1483, label %1482

; <label>:1483		; preds = %1481, %1482
	add uint %921, 4294967285		; <uint>:981 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %981		; <ubyte*>:2217 [#uses=1]
	load ubyte* %2217		; <ubyte>:3508 [#uses=1]
	seteq ubyte %3508, 0		; <bool>:1483 [#uses=1]
	br bool %1483, label %1485, label %1484

; <label>:1484		; preds = %1483, %1484
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %981		; <ubyte*>:2218 [#uses=2]
	load ubyte* %2218		; <ubyte>:3509 [#uses=1]
	add ubyte %3509, 255		; <ubyte>:3510 [#uses=1]
	store ubyte %3510, ubyte* %2218
	add uint %921, 4294967286		; <uint>:982 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %982		; <ubyte*>:2219 [#uses=2]
	load ubyte* %2219		; <ubyte>:3511 [#uses=1]
	add ubyte %3511, 1		; <ubyte>:3512 [#uses=1]
	store ubyte %3512, ubyte* %2219
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %980		; <ubyte*>:2220 [#uses=2]
	load ubyte* %2220		; <ubyte>:3513 [#uses=1]
	add ubyte %3513, 1		; <ubyte>:3514 [#uses=1]
	store ubyte %3514, ubyte* %2220
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %981		; <ubyte*>:2221 [#uses=1]
	load ubyte* %2221		; <ubyte>:3515 [#uses=1]
	seteq ubyte %3515, 0		; <bool>:1484 [#uses=1]
	br bool %1484, label %1485, label %1484

; <label>:1485		; preds = %1483, %1484
	add uint %921, 4294967286		; <uint>:983 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %983		; <ubyte*>:2222 [#uses=1]
	load ubyte* %2222		; <ubyte>:3516 [#uses=1]
	seteq ubyte %3516, 0		; <bool>:1485 [#uses=1]
	br bool %1485, label %1487, label %1486

; <label>:1486		; preds = %1485, %1486
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %981		; <ubyte*>:2223 [#uses=2]
	load ubyte* %2223		; <ubyte>:3517 [#uses=1]
	add ubyte %3517, 1		; <ubyte>:3518 [#uses=1]
	store ubyte %3518, ubyte* %2223
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %983		; <ubyte*>:2224 [#uses=2]
	load ubyte* %2224		; <ubyte>:3519 [#uses=2]
	add ubyte %3519, 255		; <ubyte>:3520 [#uses=1]
	store ubyte %3520, ubyte* %2224
	seteq ubyte %3519, 1		; <bool>:1486 [#uses=1]
	br bool %1486, label %1487, label %1486

; <label>:1487		; preds = %1485, %1486
	add uint %921, 106		; <uint>:984 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %984		; <ubyte*>:2225 [#uses=1]
	load ubyte* %2225		; <ubyte>:3521 [#uses=1]
	seteq ubyte %3521, 0		; <bool>:1487 [#uses=1]
	br bool %1487, label %1489, label %1488

; <label>:1488		; preds = %1487, %1488
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %984		; <ubyte*>:2226 [#uses=2]
	load ubyte* %2226		; <ubyte>:3522 [#uses=2]
	add ubyte %3522, 255		; <ubyte>:3523 [#uses=1]
	store ubyte %3523, ubyte* %2226
	seteq ubyte %3522, 1		; <bool>:1488 [#uses=1]
	br bool %1488, label %1489, label %1488

; <label>:1489		; preds = %1487, %1488
	add uint %921, 4		; <uint>:985 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %985		; <ubyte*>:2227 [#uses=1]
	load ubyte* %2227		; <ubyte>:3524 [#uses=1]
	seteq ubyte %3524, 0		; <bool>:1489 [#uses=1]
	br bool %1489, label %1491, label %1490

; <label>:1490		; preds = %1489, %1490
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %985		; <ubyte*>:2228 [#uses=2]
	load ubyte* %2228		; <ubyte>:3525 [#uses=1]
	add ubyte %3525, 255		; <ubyte>:3526 [#uses=1]
	store ubyte %3526, ubyte* %2228
	add uint %921, 5		; <uint>:986 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %986		; <ubyte*>:2229 [#uses=2]
	load ubyte* %2229		; <ubyte>:3527 [#uses=1]
	add ubyte %3527, 1		; <ubyte>:3528 [#uses=1]
	store ubyte %3528, ubyte* %2229
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %984		; <ubyte*>:2230 [#uses=2]
	load ubyte* %2230		; <ubyte>:3529 [#uses=1]
	add ubyte %3529, 1		; <ubyte>:3530 [#uses=1]
	store ubyte %3530, ubyte* %2230
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %985		; <ubyte*>:2231 [#uses=1]
	load ubyte* %2231		; <ubyte>:3531 [#uses=1]
	seteq ubyte %3531, 0		; <bool>:1490 [#uses=1]
	br bool %1490, label %1491, label %1490

; <label>:1491		; preds = %1489, %1490
	add uint %921, 5		; <uint>:987 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %987		; <ubyte*>:2232 [#uses=1]
	load ubyte* %2232		; <ubyte>:3532 [#uses=1]
	seteq ubyte %3532, 0		; <bool>:1491 [#uses=1]
	br bool %1491, label %1493, label %1492

; <label>:1492		; preds = %1491, %1492
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %985		; <ubyte*>:2233 [#uses=2]
	load ubyte* %2233		; <ubyte>:3533 [#uses=1]
	add ubyte %3533, 1		; <ubyte>:3534 [#uses=1]
	store ubyte %3534, ubyte* %2233
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %987		; <ubyte*>:2234 [#uses=2]
	load ubyte* %2234		; <ubyte>:3535 [#uses=2]
	add ubyte %3535, 255		; <ubyte>:3536 [#uses=1]
	store ubyte %3536, ubyte* %2234
	seteq ubyte %3535, 1		; <bool>:1492 [#uses=1]
	br bool %1492, label %1493, label %1492

; <label>:1493		; preds = %1491, %1492
	add uint %921, 20		; <uint>:988 [#uses=6]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %988		; <ubyte*>:2235 [#uses=1]
	load ubyte* %2235		; <ubyte>:3537 [#uses=1]
	seteq ubyte %3537, 0		; <bool>:1493 [#uses=1]
	br bool %1493, label %1495, label %1494

; <label>:1494		; preds = %1493, %1494
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %988		; <ubyte*>:2236 [#uses=2]
	load ubyte* %2236		; <ubyte>:3538 [#uses=2]
	add ubyte %3538, 255		; <ubyte>:3539 [#uses=1]
	store ubyte %3539, ubyte* %2236
	seteq ubyte %3538, 1		; <bool>:1494 [#uses=1]
	br bool %1494, label %1495, label %1494

; <label>:1495		; preds = %1493, %1494
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %984		; <ubyte*>:2237 [#uses=1]
	load ubyte* %2237		; <ubyte>:3540 [#uses=1]
	seteq ubyte %3540, 0		; <bool>:1495 [#uses=1]
	br bool %1495, label %1497, label %1496

; <label>:1496		; preds = %1495, %1496
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %988		; <ubyte*>:2238 [#uses=2]
	load ubyte* %2238		; <ubyte>:3541 [#uses=1]
	add ubyte %3541, 1		; <ubyte>:3542 [#uses=1]
	store ubyte %3542, ubyte* %2238
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %984		; <ubyte*>:2239 [#uses=2]
	load ubyte* %2239		; <ubyte>:3543 [#uses=2]
	add ubyte %3543, 255		; <ubyte>:3544 [#uses=1]
	store ubyte %3544, ubyte* %2239
	seteq ubyte %3543, 1		; <bool>:1496 [#uses=1]
	br bool %1496, label %1497, label %1496

; <label>:1497		; preds = %1495, %1496
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %988		; <ubyte*>:2240 [#uses=1]
	load ubyte* %2240		; <ubyte>:3545 [#uses=1]
	seteq ubyte %3545, 0		; <bool>:1497 [#uses=1]
	br bool %1497, label %1499, label %1498

; <label>:1498		; preds = %1497, %1501
	phi uint [ %988, %1497 ], [ %993, %1501 ]		; <uint>:989 [#uses=6]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %989		; <ubyte*>:2241 [#uses=1]
	load ubyte* %2241		; <ubyte>:3546 [#uses=1]
	seteq ubyte %3546, 0		; <bool>:1498 [#uses=1]
	br bool %1498, label %1501, label %1500

; <label>:1499		; preds = %1497, %1501
	phi uint [ %988, %1497 ], [ %993, %1501 ]		; <uint>:990 [#uses=7]
	add uint %990, 4294967292		; <uint>:991 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %991		; <ubyte*>:2242 [#uses=1]
	load ubyte* %2242		; <ubyte>:3547 [#uses=1]
	seteq ubyte %3547, 0		; <bool>:1499 [#uses=1]
	br bool %1499, label %1503, label %1502

; <label>:1500		; preds = %1498, %1500
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %989		; <ubyte*>:2243 [#uses=2]
	load ubyte* %2243		; <ubyte>:3548 [#uses=1]
	add ubyte %3548, 255		; <ubyte>:3549 [#uses=1]
	store ubyte %3549, ubyte* %2243
	add uint %989, 6		; <uint>:992 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %992		; <ubyte*>:2244 [#uses=2]
	load ubyte* %2244		; <ubyte>:3550 [#uses=1]
	add ubyte %3550, 1		; <ubyte>:3551 [#uses=1]
	store ubyte %3551, ubyte* %2244
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %989		; <ubyte*>:2245 [#uses=1]
	load ubyte* %2245		; <ubyte>:3552 [#uses=1]
	seteq ubyte %3552, 0		; <bool>:1500 [#uses=1]
	br bool %1500, label %1501, label %1500

; <label>:1501		; preds = %1498, %1500
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %989		; <ubyte*>:2246 [#uses=2]
	load ubyte* %2246		; <ubyte>:3553 [#uses=1]
	add ubyte %3553, 1		; <ubyte>:3554 [#uses=1]
	store ubyte %3554, ubyte* %2246
	add uint %989, 6		; <uint>:993 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %993		; <ubyte*>:2247 [#uses=2]
	load ubyte* %2247		; <ubyte>:3555 [#uses=2]
	add ubyte %3555, 255		; <ubyte>:3556 [#uses=1]
	store ubyte %3556, ubyte* %2247
	seteq ubyte %3555, 1		; <bool>:1501 [#uses=1]
	br bool %1501, label %1499, label %1498

; <label>:1502		; preds = %1499, %1502
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %991		; <ubyte*>:2248 [#uses=2]
	load ubyte* %2248		; <ubyte>:3557 [#uses=2]
	add ubyte %3557, 255		; <ubyte>:3558 [#uses=1]
	store ubyte %3558, ubyte* %2248
	seteq ubyte %3557, 1		; <bool>:1502 [#uses=1]
	br bool %1502, label %1503, label %1502

; <label>:1503		; preds = %1499, %1502
	add uint %990, 4294967294		; <uint>:994 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %994		; <ubyte*>:2249 [#uses=1]
	load ubyte* %2249		; <ubyte>:3559 [#uses=1]
	seteq ubyte %3559, 0		; <bool>:1503 [#uses=1]
	br bool %1503, label %1505, label %1504

; <label>:1504		; preds = %1503, %1504
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %991		; <ubyte*>:2250 [#uses=2]
	load ubyte* %2250		; <ubyte>:3560 [#uses=1]
	add ubyte %3560, 1		; <ubyte>:3561 [#uses=1]
	store ubyte %3561, ubyte* %2250
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %994		; <ubyte*>:2251 [#uses=2]
	load ubyte* %2251		; <ubyte>:3562 [#uses=1]
	add ubyte %3562, 255		; <ubyte>:3563 [#uses=1]
	store ubyte %3563, ubyte* %2251
	add uint %990, 4294967295		; <uint>:995 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %995		; <ubyte*>:2252 [#uses=2]
	load ubyte* %2252		; <ubyte>:3564 [#uses=1]
	add ubyte %3564, 1		; <ubyte>:3565 [#uses=1]
	store ubyte %3565, ubyte* %2252
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %994		; <ubyte*>:2253 [#uses=1]
	load ubyte* %2253		; <ubyte>:3566 [#uses=1]
	seteq ubyte %3566, 0		; <bool>:1504 [#uses=1]
	br bool %1504, label %1505, label %1504

; <label>:1505		; preds = %1503, %1504
	add uint %990, 4294967295		; <uint>:996 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %996		; <ubyte*>:2254 [#uses=1]
	load ubyte* %2254		; <ubyte>:3567 [#uses=1]
	seteq ubyte %3567, 0		; <bool>:1505 [#uses=1]
	br bool %1505, label %1507, label %1506

; <label>:1506		; preds = %1505, %1506
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %994		; <ubyte*>:2255 [#uses=2]
	load ubyte* %2255		; <ubyte>:3568 [#uses=1]
	add ubyte %3568, 1		; <ubyte>:3569 [#uses=1]
	store ubyte %3569, ubyte* %2255
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %996		; <ubyte*>:2256 [#uses=2]
	load ubyte* %2256		; <ubyte>:3570 [#uses=2]
	add ubyte %3570, 255		; <ubyte>:3571 [#uses=1]
	store ubyte %3571, ubyte* %2256
	seteq ubyte %3570, 1		; <bool>:1506 [#uses=1]
	br bool %1506, label %1507, label %1506

; <label>:1507		; preds = %1505, %1506
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %990		; <ubyte*>:2257 [#uses=2]
	load ubyte* %2257		; <ubyte>:3572 [#uses=2]
	add ubyte %3572, 1		; <ubyte>:3573 [#uses=1]
	store ubyte %3573, ubyte* %2257
	seteq ubyte %3572, 255		; <bool>:1507 [#uses=1]
	br bool %1507, label %1509, label %1508

; <label>:1508		; preds = %1507, %1513
	phi uint [ %990, %1507 ], [ %1002, %1513 ]		; <uint>:997 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %997		; <ubyte*>:2258 [#uses=2]
	load ubyte* %2258		; <ubyte>:3574 [#uses=1]
	add ubyte %3574, 255		; <ubyte>:3575 [#uses=1]
	store ubyte %3575, ubyte* %2258
	add uint %997, 4294967286		; <uint>:998 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %998		; <ubyte*>:2259 [#uses=1]
	load ubyte* %2259		; <ubyte>:3576 [#uses=1]
	seteq ubyte %3576, 0		; <bool>:1508 [#uses=1]
	br bool %1508, label %1511, label %1510

; <label>:1509		; preds = %1507, %1513
	phi uint [ %990, %1507 ], [ %1002, %1513 ]		; <uint>:999 [#uses=43]
	add uint %999, 4		; <uint>:1000 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1000		; <ubyte*>:2260 [#uses=1]
	load ubyte* %2260		; <ubyte>:3577 [#uses=1]
	seteq ubyte %3577, 0		; <bool>:1509 [#uses=1]
	br bool %1509, label %1515, label %1514

; <label>:1510		; preds = %1508, %1510
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %998		; <ubyte*>:2261 [#uses=2]
	load ubyte* %2261		; <ubyte>:3578 [#uses=2]
	add ubyte %3578, 255		; <ubyte>:3579 [#uses=1]
	store ubyte %3579, ubyte* %2261
	seteq ubyte %3578, 1		; <bool>:1510 [#uses=1]
	br bool %1510, label %1511, label %1510

; <label>:1511		; preds = %1508, %1510
	add uint %997, 4294967292		; <uint>:1001 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1001		; <ubyte*>:2262 [#uses=1]
	load ubyte* %2262		; <ubyte>:3580 [#uses=1]
	seteq ubyte %3580, 0		; <bool>:1511 [#uses=1]
	br bool %1511, label %1513, label %1512

; <label>:1512		; preds = %1511, %1512
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %998		; <ubyte*>:2263 [#uses=2]
	load ubyte* %2263		; <ubyte>:3581 [#uses=1]
	add ubyte %3581, 1		; <ubyte>:3582 [#uses=1]
	store ubyte %3582, ubyte* %2263
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1001		; <ubyte*>:2264 [#uses=2]
	load ubyte* %2264		; <ubyte>:3583 [#uses=2]
	add ubyte %3583, 255		; <ubyte>:3584 [#uses=1]
	store ubyte %3584, ubyte* %2264
	seteq ubyte %3583, 1		; <bool>:1512 [#uses=1]
	br bool %1512, label %1513, label %1512

; <label>:1513		; preds = %1511, %1512
	add uint %997, 4294967290		; <uint>:1002 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1002		; <ubyte*>:2265 [#uses=1]
	load ubyte* %2265		; <ubyte>:3585 [#uses=1]
	seteq ubyte %3585, 0		; <bool>:1513 [#uses=1]
	br bool %1513, label %1509, label %1508

; <label>:1514		; preds = %1509, %1514
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1000		; <ubyte*>:2266 [#uses=2]
	load ubyte* %2266		; <ubyte>:3586 [#uses=2]
	add ubyte %3586, 255		; <ubyte>:3587 [#uses=1]
	store ubyte %3587, ubyte* %2266
	seteq ubyte %3586, 1		; <bool>:1514 [#uses=1]
	br bool %1514, label %1515, label %1514

; <label>:1515		; preds = %1509, %1514
	add uint %999, 10		; <uint>:1003 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1003		; <ubyte*>:2267 [#uses=1]
	load ubyte* %2267		; <ubyte>:3588 [#uses=1]
	seteq ubyte %3588, 0		; <bool>:1515 [#uses=1]
	br bool %1515, label %1517, label %1516

; <label>:1516		; preds = %1515, %1516
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1003		; <ubyte*>:2268 [#uses=2]
	load ubyte* %2268		; <ubyte>:3589 [#uses=2]
	add ubyte %3589, 255		; <ubyte>:3590 [#uses=1]
	store ubyte %3590, ubyte* %2268
	seteq ubyte %3589, 1		; <bool>:1516 [#uses=1]
	br bool %1516, label %1517, label %1516

; <label>:1517		; preds = %1515, %1516
	add uint %999, 16		; <uint>:1004 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1004		; <ubyte*>:2269 [#uses=1]
	load ubyte* %2269		; <ubyte>:3591 [#uses=1]
	seteq ubyte %3591, 0		; <bool>:1517 [#uses=1]
	br bool %1517, label %1519, label %1518

; <label>:1518		; preds = %1517, %1518
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1004		; <ubyte*>:2270 [#uses=2]
	load ubyte* %2270		; <ubyte>:3592 [#uses=2]
	add ubyte %3592, 255		; <ubyte>:3593 [#uses=1]
	store ubyte %3593, ubyte* %2270
	seteq ubyte %3592, 1		; <bool>:1518 [#uses=1]
	br bool %1518, label %1519, label %1518

; <label>:1519		; preds = %1517, %1518
	add uint %999, 22		; <uint>:1005 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1005		; <ubyte*>:2271 [#uses=1]
	load ubyte* %2271		; <ubyte>:3594 [#uses=1]
	seteq ubyte %3594, 0		; <bool>:1519 [#uses=1]
	br bool %1519, label %1521, label %1520

; <label>:1520		; preds = %1519, %1520
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1005		; <ubyte*>:2272 [#uses=2]
	load ubyte* %2272		; <ubyte>:3595 [#uses=2]
	add ubyte %3595, 255		; <ubyte>:3596 [#uses=1]
	store ubyte %3596, ubyte* %2272
	seteq ubyte %3595, 1		; <bool>:1520 [#uses=1]
	br bool %1520, label %1521, label %1520

; <label>:1521		; preds = %1519, %1520
	add uint %999, 28		; <uint>:1006 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1006		; <ubyte*>:2273 [#uses=1]
	load ubyte* %2273		; <ubyte>:3597 [#uses=1]
	seteq ubyte %3597, 0		; <bool>:1521 [#uses=1]
	br bool %1521, label %1523, label %1522

; <label>:1522		; preds = %1521, %1522
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1006		; <ubyte*>:2274 [#uses=2]
	load ubyte* %2274		; <ubyte>:3598 [#uses=2]
	add ubyte %3598, 255		; <ubyte>:3599 [#uses=1]
	store ubyte %3599, ubyte* %2274
	seteq ubyte %3598, 1		; <bool>:1522 [#uses=1]
	br bool %1522, label %1523, label %1522

; <label>:1523		; preds = %1521, %1522
	add uint %999, 34		; <uint>:1007 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1007		; <ubyte*>:2275 [#uses=1]
	load ubyte* %2275		; <ubyte>:3600 [#uses=1]
	seteq ubyte %3600, 0		; <bool>:1523 [#uses=1]
	br bool %1523, label %1525, label %1524

; <label>:1524		; preds = %1523, %1524
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1007		; <ubyte*>:2276 [#uses=2]
	load ubyte* %2276		; <ubyte>:3601 [#uses=2]
	add ubyte %3601, 255		; <ubyte>:3602 [#uses=1]
	store ubyte %3602, ubyte* %2276
	seteq ubyte %3601, 1		; <bool>:1524 [#uses=1]
	br bool %1524, label %1525, label %1524

; <label>:1525		; preds = %1523, %1524
	add uint %999, 40		; <uint>:1008 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1008		; <ubyte*>:2277 [#uses=1]
	load ubyte* %2277		; <ubyte>:3603 [#uses=1]
	seteq ubyte %3603, 0		; <bool>:1525 [#uses=1]
	br bool %1525, label %1527, label %1526

; <label>:1526		; preds = %1525, %1526
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1008		; <ubyte*>:2278 [#uses=2]
	load ubyte* %2278		; <ubyte>:3604 [#uses=2]
	add ubyte %3604, 255		; <ubyte>:3605 [#uses=1]
	store ubyte %3605, ubyte* %2278
	seteq ubyte %3604, 1		; <bool>:1526 [#uses=1]
	br bool %1526, label %1527, label %1526

; <label>:1527		; preds = %1525, %1526
	add uint %999, 46		; <uint>:1009 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1009		; <ubyte*>:2279 [#uses=1]
	load ubyte* %2279		; <ubyte>:3606 [#uses=1]
	seteq ubyte %3606, 0		; <bool>:1527 [#uses=1]
	br bool %1527, label %1529, label %1528

; <label>:1528		; preds = %1527, %1528
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1009		; <ubyte*>:2280 [#uses=2]
	load ubyte* %2280		; <ubyte>:3607 [#uses=2]
	add ubyte %3607, 255		; <ubyte>:3608 [#uses=1]
	store ubyte %3608, ubyte* %2280
	seteq ubyte %3607, 1		; <bool>:1528 [#uses=1]
	br bool %1528, label %1529, label %1528

; <label>:1529		; preds = %1527, %1528
	add uint %999, 52		; <uint>:1010 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1010		; <ubyte*>:2281 [#uses=1]
	load ubyte* %2281		; <ubyte>:3609 [#uses=1]
	seteq ubyte %3609, 0		; <bool>:1529 [#uses=1]
	br bool %1529, label %1531, label %1530

; <label>:1530		; preds = %1529, %1530
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1010		; <ubyte*>:2282 [#uses=2]
	load ubyte* %2282		; <ubyte>:3610 [#uses=2]
	add ubyte %3610, 255		; <ubyte>:3611 [#uses=1]
	store ubyte %3611, ubyte* %2282
	seteq ubyte %3610, 1		; <bool>:1530 [#uses=1]
	br bool %1530, label %1531, label %1530

; <label>:1531		; preds = %1529, %1530
	add uint %999, 58		; <uint>:1011 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1011		; <ubyte*>:2283 [#uses=1]
	load ubyte* %2283		; <ubyte>:3612 [#uses=1]
	seteq ubyte %3612, 0		; <bool>:1531 [#uses=1]
	br bool %1531, label %1533, label %1532

; <label>:1532		; preds = %1531, %1532
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1011		; <ubyte*>:2284 [#uses=2]
	load ubyte* %2284		; <ubyte>:3613 [#uses=2]
	add ubyte %3613, 255		; <ubyte>:3614 [#uses=1]
	store ubyte %3614, ubyte* %2284
	seteq ubyte %3613, 1		; <bool>:1532 [#uses=1]
	br bool %1532, label %1533, label %1532

; <label>:1533		; preds = %1531, %1532
	add uint %999, 64		; <uint>:1012 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1012		; <ubyte*>:2285 [#uses=1]
	load ubyte* %2285		; <ubyte>:3615 [#uses=1]
	seteq ubyte %3615, 0		; <bool>:1533 [#uses=1]
	br bool %1533, label %1535, label %1534

; <label>:1534		; preds = %1533, %1534
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1012		; <ubyte*>:2286 [#uses=2]
	load ubyte* %2286		; <ubyte>:3616 [#uses=2]
	add ubyte %3616, 255		; <ubyte>:3617 [#uses=1]
	store ubyte %3617, ubyte* %2286
	seteq ubyte %3616, 1		; <bool>:1534 [#uses=1]
	br bool %1534, label %1535, label %1534

; <label>:1535		; preds = %1533, %1534
	add uint %999, 70		; <uint>:1013 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1013		; <ubyte*>:2287 [#uses=1]
	load ubyte* %2287		; <ubyte>:3618 [#uses=1]
	seteq ubyte %3618, 0		; <bool>:1535 [#uses=1]
	br bool %1535, label %1537, label %1536

; <label>:1536		; preds = %1535, %1536
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1013		; <ubyte*>:2288 [#uses=2]
	load ubyte* %2288		; <ubyte>:3619 [#uses=2]
	add ubyte %3619, 255		; <ubyte>:3620 [#uses=1]
	store ubyte %3620, ubyte* %2288
	seteq ubyte %3619, 1		; <bool>:1536 [#uses=1]
	br bool %1536, label %1537, label %1536

; <label>:1537		; preds = %1535, %1536
	add uint %999, 76		; <uint>:1014 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1014		; <ubyte*>:2289 [#uses=1]
	load ubyte* %2289		; <ubyte>:3621 [#uses=1]
	seteq ubyte %3621, 0		; <bool>:1537 [#uses=1]
	br bool %1537, label %1539, label %1538

; <label>:1538		; preds = %1537, %1538
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1014		; <ubyte*>:2290 [#uses=2]
	load ubyte* %2290		; <ubyte>:3622 [#uses=2]
	add ubyte %3622, 255		; <ubyte>:3623 [#uses=1]
	store ubyte %3623, ubyte* %2290
	seteq ubyte %3622, 1		; <bool>:1538 [#uses=1]
	br bool %1538, label %1539, label %1538

; <label>:1539		; preds = %1537, %1538
	add uint %999, 82		; <uint>:1015 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1015		; <ubyte*>:2291 [#uses=1]
	load ubyte* %2291		; <ubyte>:3624 [#uses=1]
	seteq ubyte %3624, 0		; <bool>:1539 [#uses=1]
	br bool %1539, label %1541, label %1540

; <label>:1540		; preds = %1539, %1540
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1015		; <ubyte*>:2292 [#uses=2]
	load ubyte* %2292		; <ubyte>:3625 [#uses=2]
	add ubyte %3625, 255		; <ubyte>:3626 [#uses=1]
	store ubyte %3626, ubyte* %2292
	seteq ubyte %3625, 1		; <bool>:1540 [#uses=1]
	br bool %1540, label %1541, label %1540

; <label>:1541		; preds = %1539, %1540
	add uint %999, 88		; <uint>:1016 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1016		; <ubyte*>:2293 [#uses=1]
	load ubyte* %2293		; <ubyte>:3627 [#uses=1]
	seteq ubyte %3627, 0		; <bool>:1541 [#uses=1]
	br bool %1541, label %1543, label %1542

; <label>:1542		; preds = %1541, %1542
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1016		; <ubyte*>:2294 [#uses=2]
	load ubyte* %2294		; <ubyte>:3628 [#uses=2]
	add ubyte %3628, 255		; <ubyte>:3629 [#uses=1]
	store ubyte %3629, ubyte* %2294
	seteq ubyte %3628, 1		; <bool>:1542 [#uses=1]
	br bool %1542, label %1543, label %1542

; <label>:1543		; preds = %1541, %1542
	add uint %999, 4294967294		; <uint>:1017 [#uses=12]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1017		; <ubyte*>:2295 [#uses=1]
	load ubyte* %2295		; <ubyte>:3630 [#uses=1]
	seteq ubyte %3630, 0		; <bool>:1543 [#uses=1]
	br bool %1543, label %1545, label %1544

; <label>:1544		; preds = %1543, %1544
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1017		; <ubyte*>:2296 [#uses=2]
	load ubyte* %2296		; <ubyte>:3631 [#uses=2]
	add ubyte %3631, 255		; <ubyte>:3632 [#uses=1]
	store ubyte %3632, ubyte* %2296
	seteq ubyte %3631, 1		; <bool>:1544 [#uses=1]
	br bool %1544, label %1545, label %1544

; <label>:1545		; preds = %1543, %1544
	add uint %999, 4294967284		; <uint>:1018 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1018		; <ubyte*>:2297 [#uses=1]
	load ubyte* %2297		; <ubyte>:3633 [#uses=1]
	seteq ubyte %3633, 0		; <bool>:1545 [#uses=1]
	br bool %1545, label %1547, label %1546

; <label>:1546		; preds = %1545, %1546
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1018		; <ubyte*>:2298 [#uses=2]
	load ubyte* %2298		; <ubyte>:3634 [#uses=1]
	add ubyte %3634, 255		; <ubyte>:3635 [#uses=1]
	store ubyte %3635, ubyte* %2298
	add uint %999, 4294967285		; <uint>:1019 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1019		; <ubyte*>:2299 [#uses=2]
	load ubyte* %2299		; <ubyte>:3636 [#uses=1]
	add ubyte %3636, 1		; <ubyte>:3637 [#uses=1]
	store ubyte %3637, ubyte* %2299
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1017		; <ubyte*>:2300 [#uses=2]
	load ubyte* %2300		; <ubyte>:3638 [#uses=1]
	add ubyte %3638, 1		; <ubyte>:3639 [#uses=1]
	store ubyte %3639, ubyte* %2300
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1018		; <ubyte*>:2301 [#uses=1]
	load ubyte* %2301		; <ubyte>:3640 [#uses=1]
	seteq ubyte %3640, 0		; <bool>:1546 [#uses=1]
	br bool %1546, label %1547, label %1546

; <label>:1547		; preds = %1545, %1546
	add uint %999, 4294967285		; <uint>:1020 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1020		; <ubyte*>:2302 [#uses=1]
	load ubyte* %2302		; <ubyte>:3641 [#uses=1]
	seteq ubyte %3641, 0		; <bool>:1547 [#uses=1]
	br bool %1547, label %1549, label %1548

; <label>:1548		; preds = %1547, %1548
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1018		; <ubyte*>:2303 [#uses=2]
	load ubyte* %2303		; <ubyte>:3642 [#uses=1]
	add ubyte %3642, 1		; <ubyte>:3643 [#uses=1]
	store ubyte %3643, ubyte* %2303
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1020		; <ubyte*>:2304 [#uses=2]
	load ubyte* %2304		; <ubyte>:3644 [#uses=2]
	add ubyte %3644, 255		; <ubyte>:3645 [#uses=1]
	store ubyte %3645, ubyte* %2304
	seteq ubyte %3644, 1		; <bool>:1548 [#uses=1]
	br bool %1548, label %1549, label %1548

; <label>:1549		; preds = %1547, %1548
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %999		; <ubyte*>:2305 [#uses=2]
	load ubyte* %2305		; <ubyte>:3646 [#uses=2]
	add ubyte %3646, 1		; <ubyte>:3647 [#uses=1]
	store ubyte %3647, ubyte* %2305
	seteq ubyte %3646, 255		; <bool>:1549 [#uses=1]
	br bool %1549, label %1551, label %1550

; <label>:1550		; preds = %1549, %1573
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %999		; <ubyte*>:2306 [#uses=2]
	load ubyte* %2306		; <ubyte>:3648 [#uses=1]
	add ubyte %3648, 1		; <ubyte>:3649 [#uses=1]
	store ubyte %3649, ubyte* %2306
	add uint %999, 4294967292		; <uint>:1021 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1021		; <ubyte*>:2307 [#uses=1]
	load ubyte* %2307		; <ubyte>:3650 [#uses=1]
	seteq ubyte %3650, 0		; <bool>:1550 [#uses=1]
	br bool %1550, label %1553, label %1552

; <label>:1551		; preds = %1549, %1573
	add uint %999, 4294967292		; <uint>:1022 [#uses=10]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1022		; <ubyte*>:2308 [#uses=1]
	load ubyte* %2308		; <ubyte>:3651 [#uses=1]
	seteq ubyte %3651, 0		; <bool>:1551 [#uses=1]
	br bool %1551, label %1575, label %1574

; <label>:1552		; preds = %1550, %1552
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1021		; <ubyte*>:2309 [#uses=2]
	load ubyte* %2309		; <ubyte>:3652 [#uses=1]
	add ubyte %3652, 255		; <ubyte>:3653 [#uses=1]
	store ubyte %3653, ubyte* %2309
	add uint %999, 4294967293		; <uint>:1023 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1023		; <ubyte*>:2310 [#uses=2]
	load ubyte* %2310		; <ubyte>:3654 [#uses=1]
	add ubyte %3654, 1		; <ubyte>:3655 [#uses=1]
	store ubyte %3655, ubyte* %2310
	add uint %999, 1		; <uint>:1024 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1024		; <ubyte*>:2311 [#uses=2]
	load ubyte* %2311		; <ubyte>:3656 [#uses=1]
	add ubyte %3656, 1		; <ubyte>:3657 [#uses=1]
	store ubyte %3657, ubyte* %2311
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1021		; <ubyte*>:2312 [#uses=1]
	load ubyte* %2312		; <ubyte>:3658 [#uses=1]
	seteq ubyte %3658, 0		; <bool>:1552 [#uses=1]
	br bool %1552, label %1553, label %1552

; <label>:1553		; preds = %1550, %1552
	add uint %999, 4294967293		; <uint>:1025 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1025		; <ubyte*>:2313 [#uses=1]
	load ubyte* %2313		; <ubyte>:3659 [#uses=1]
	seteq ubyte %3659, 0		; <bool>:1553 [#uses=1]
	br bool %1553, label %1555, label %1554

; <label>:1554		; preds = %1553, %1554
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1021		; <ubyte*>:2314 [#uses=2]
	load ubyte* %2314		; <ubyte>:3660 [#uses=1]
	add ubyte %3660, 1		; <ubyte>:3661 [#uses=1]
	store ubyte %3661, ubyte* %2314
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1025		; <ubyte*>:2315 [#uses=2]
	load ubyte* %2315		; <ubyte>:3662 [#uses=2]
	add ubyte %3662, 255		; <ubyte>:3663 [#uses=1]
	store ubyte %3663, ubyte* %2315
	seteq ubyte %3662, 1		; <bool>:1554 [#uses=1]
	br bool %1554, label %1555, label %1554

; <label>:1555		; preds = %1553, %1554
	add uint %999, 1		; <uint>:1026 [#uses=13]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1026		; <ubyte*>:2316 [#uses=1]
	load ubyte* %2316		; <ubyte>:3664 [#uses=1]
	seteq ubyte %3664, 0		; <bool>:1555 [#uses=1]
	br bool %1555, label %1557, label %1556

; <label>:1556		; preds = %1555, %1559
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1026		; <ubyte*>:2317 [#uses=1]
	load ubyte* %2317		; <ubyte>:3665 [#uses=1]
	seteq ubyte %3665, 0		; <bool>:1556 [#uses=1]
	br bool %1556, label %1559, label %1558

; <label>:1557		; preds = %1555, %1559
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1017		; <ubyte*>:2318 [#uses=1]
	load ubyte* %2318		; <ubyte>:3666 [#uses=1]
	seteq ubyte %3666, 0		; <bool>:1557 [#uses=1]
	br bool %1557, label %1561, label %1560

; <label>:1558		; preds = %1556, %1558
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1026		; <ubyte*>:2319 [#uses=2]
	load ubyte* %2319		; <ubyte>:3667 [#uses=2]
	add ubyte %3667, 255		; <ubyte>:3668 [#uses=1]
	store ubyte %3668, ubyte* %2319
	seteq ubyte %3667, 1		; <bool>:1558 [#uses=1]
	br bool %1558, label %1559, label %1558

; <label>:1559		; preds = %1556, %1558
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %999		; <ubyte*>:2320 [#uses=2]
	load ubyte* %2320		; <ubyte>:3669 [#uses=1]
	add ubyte %3669, 255		; <ubyte>:3670 [#uses=1]
	store ubyte %3670, ubyte* %2320
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1026		; <ubyte*>:2321 [#uses=1]
	load ubyte* %2321		; <ubyte>:3671 [#uses=1]
	seteq ubyte %3671, 0		; <bool>:1559 [#uses=1]
	br bool %1559, label %1557, label %1556

; <label>:1560		; preds = %1557, %1560
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1017		; <ubyte*>:2322 [#uses=2]
	load ubyte* %2322		; <ubyte>:3672 [#uses=1]
	add ubyte %3672, 255		; <ubyte>:3673 [#uses=1]
	store ubyte %3673, ubyte* %2322
	add uint %999, 4294967295		; <uint>:1027 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1027		; <ubyte*>:2323 [#uses=2]
	load ubyte* %2323		; <ubyte>:3674 [#uses=1]
	add ubyte %3674, 1		; <ubyte>:3675 [#uses=1]
	store ubyte %3675, ubyte* %2323
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1026		; <ubyte*>:2324 [#uses=2]
	load ubyte* %2324		; <ubyte>:3676 [#uses=1]
	add ubyte %3676, 1		; <ubyte>:3677 [#uses=1]
	store ubyte %3677, ubyte* %2324
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1017		; <ubyte*>:2325 [#uses=1]
	load ubyte* %2325		; <ubyte>:3678 [#uses=1]
	seteq ubyte %3678, 0		; <bool>:1560 [#uses=1]
	br bool %1560, label %1561, label %1560

; <label>:1561		; preds = %1557, %1560
	add uint %999, 4294967295		; <uint>:1028 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1028		; <ubyte*>:2326 [#uses=1]
	load ubyte* %2326		; <ubyte>:3679 [#uses=1]
	seteq ubyte %3679, 0		; <bool>:1561 [#uses=1]
	br bool %1561, label %1563, label %1562

; <label>:1562		; preds = %1561, %1562
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1017		; <ubyte*>:2327 [#uses=2]
	load ubyte* %2327		; <ubyte>:3680 [#uses=1]
	add ubyte %3680, 1		; <ubyte>:3681 [#uses=1]
	store ubyte %3681, ubyte* %2327
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1028		; <ubyte*>:2328 [#uses=2]
	load ubyte* %2328		; <ubyte>:3682 [#uses=2]
	add ubyte %3682, 255		; <ubyte>:3683 [#uses=1]
	store ubyte %3683, ubyte* %2328
	seteq ubyte %3682, 1		; <bool>:1562 [#uses=1]
	br bool %1562, label %1563, label %1562

; <label>:1563		; preds = %1561, %1562
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1026		; <ubyte*>:2329 [#uses=1]
	load ubyte* %2329		; <ubyte>:3684 [#uses=1]
	seteq ubyte %3684, 0		; <bool>:1563 [#uses=1]
	br bool %1563, label %1565, label %1564

; <label>:1564		; preds = %1563, %1567
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1026		; <ubyte*>:2330 [#uses=1]
	load ubyte* %2330		; <ubyte>:3685 [#uses=1]
	seteq ubyte %3685, 0		; <bool>:1564 [#uses=1]
	br bool %1564, label %1567, label %1566

; <label>:1565		; preds = %1563, %1567
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1026		; <ubyte*>:2331 [#uses=2]
	load ubyte* %2331		; <ubyte>:3686 [#uses=1]
	add ubyte %3686, 1		; <ubyte>:3687 [#uses=1]
	store ubyte %3687, ubyte* %2331
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %999		; <ubyte*>:2332 [#uses=1]
	load ubyte* %2332		; <ubyte>:3688 [#uses=1]
	seteq ubyte %3688, 0		; <bool>:1565 [#uses=1]
	br bool %1565, label %1569, label %1568

; <label>:1566		; preds = %1564, %1566
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1026		; <ubyte*>:2333 [#uses=2]
	load ubyte* %2333		; <ubyte>:3689 [#uses=2]
	add ubyte %3689, 255		; <ubyte>:3690 [#uses=1]
	store ubyte %3690, ubyte* %2333
	seteq ubyte %3689, 1		; <bool>:1566 [#uses=1]
	br bool %1566, label %1567, label %1566

; <label>:1567		; preds = %1564, %1566
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %999		; <ubyte*>:2334 [#uses=2]
	load ubyte* %2334		; <ubyte>:3691 [#uses=1]
	add ubyte %3691, 255		; <ubyte>:3692 [#uses=1]
	store ubyte %3692, ubyte* %2334
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1026		; <ubyte*>:2335 [#uses=1]
	load ubyte* %2335		; <ubyte>:3693 [#uses=1]
	seteq ubyte %3693, 0		; <bool>:1567 [#uses=1]
	br bool %1567, label %1565, label %1564

; <label>:1568		; preds = %1565, %1571
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %999		; <ubyte*>:2336 [#uses=1]
	load ubyte* %2336		; <ubyte>:3694 [#uses=1]
	seteq ubyte %3694, 0		; <bool>:1568 [#uses=1]
	br bool %1568, label %1571, label %1570

; <label>:1569		; preds = %1565, %1571
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1026		; <ubyte*>:2337 [#uses=1]
	load ubyte* %2337		; <ubyte>:3695 [#uses=1]
	seteq ubyte %3695, 0		; <bool>:1569 [#uses=1]
	br bool %1569, label %1573, label %1572

; <label>:1570		; preds = %1568, %1570
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %999		; <ubyte*>:2338 [#uses=2]
	load ubyte* %2338		; <ubyte>:3696 [#uses=2]
	add ubyte %3696, 255		; <ubyte>:3697 [#uses=1]
	store ubyte %3697, ubyte* %2338
	seteq ubyte %3696, 1		; <bool>:1570 [#uses=1]
	br bool %1570, label %1571, label %1570

; <label>:1571		; preds = %1568, %1570
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1026		; <ubyte*>:2339 [#uses=2]
	load ubyte* %2339		; <ubyte>:3698 [#uses=1]
	add ubyte %3698, 255		; <ubyte>:3699 [#uses=1]
	store ubyte %3699, ubyte* %2339
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %999		; <ubyte*>:2340 [#uses=1]
	load ubyte* %2340		; <ubyte>:3700 [#uses=1]
	seteq ubyte %3700, 0		; <bool>:1571 [#uses=1]
	br bool %1571, label %1569, label %1568

; <label>:1572		; preds = %1569, %1572
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1021		; <ubyte*>:2341 [#uses=2]
	load ubyte* %2341		; <ubyte>:3701 [#uses=1]
	add ubyte %3701, 255		; <ubyte>:3702 [#uses=1]
	store ubyte %3702, ubyte* %2341
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1017		; <ubyte*>:2342 [#uses=2]
	load ubyte* %2342		; <ubyte>:3703 [#uses=1]
	add ubyte %3703, 255		; <ubyte>:3704 [#uses=1]
	store ubyte %3704, ubyte* %2342
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %999		; <ubyte*>:2343 [#uses=2]
	load ubyte* %2343		; <ubyte>:3705 [#uses=1]
	add ubyte %3705, 1		; <ubyte>:3706 [#uses=1]
	store ubyte %3706, ubyte* %2343
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1026		; <ubyte*>:2344 [#uses=2]
	load ubyte* %2344		; <ubyte>:3707 [#uses=2]
	add ubyte %3707, 255		; <ubyte>:3708 [#uses=1]
	store ubyte %3708, ubyte* %2344
	seteq ubyte %3707, 1		; <bool>:1572 [#uses=1]
	br bool %1572, label %1573, label %1572

; <label>:1573		; preds = %1569, %1572
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %999		; <ubyte*>:2345 [#uses=1]
	load ubyte* %2345		; <ubyte>:3709 [#uses=1]
	seteq ubyte %3709, 0		; <bool>:1573 [#uses=1]
	br bool %1573, label %1551, label %1550

; <label>:1574		; preds = %1551, %1577
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1022		; <ubyte*>:2346 [#uses=1]
	load ubyte* %2346		; <ubyte>:3710 [#uses=1]
	seteq ubyte %3710, 0		; <bool>:1574 [#uses=1]
	br bool %1574, label %1577, label %1576

; <label>:1575		; preds = %1551, %1577
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1017		; <ubyte*>:2347 [#uses=1]
	load ubyte* %2347		; <ubyte>:3711 [#uses=1]
	seteq ubyte %3711, 0		; <bool>:1575 [#uses=1]
	br bool %1575, label %1579, label %1578

; <label>:1576		; preds = %1574, %1576
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1022		; <ubyte*>:2348 [#uses=2]
	load ubyte* %2348		; <ubyte>:3712 [#uses=2]
	add ubyte %3712, 255		; <ubyte>:3713 [#uses=1]
	store ubyte %3713, ubyte* %2348
	seteq ubyte %3712, 1		; <bool>:1576 [#uses=1]
	br bool %1576, label %1577, label %1576

; <label>:1577		; preds = %1574, %1576
	add uint %999, 4294967293		; <uint>:1029 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1029		; <ubyte*>:2349 [#uses=2]
	load ubyte* %2349		; <ubyte>:3714 [#uses=1]
	add ubyte %3714, 1		; <ubyte>:3715 [#uses=1]
	store ubyte %3715, ubyte* %2349
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1022		; <ubyte*>:2350 [#uses=1]
	load ubyte* %2350		; <ubyte>:3716 [#uses=1]
	seteq ubyte %3716, 0		; <bool>:1577 [#uses=1]
	br bool %1577, label %1575, label %1574

; <label>:1578		; preds = %1575, %1581
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1017		; <ubyte*>:2351 [#uses=1]
	load ubyte* %2351		; <ubyte>:3717 [#uses=1]
	seteq ubyte %3717, 0		; <bool>:1578 [#uses=1]
	br bool %1578, label %1581, label %1580

; <label>:1579		; preds = %1575, %1581
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1022		; <ubyte*>:2352 [#uses=2]
	load ubyte* %2352		; <ubyte>:3718 [#uses=1]
	add ubyte %3718, 1		; <ubyte>:3719 [#uses=1]
	store ubyte %3719, ubyte* %2352
	add uint %999, 4294967295		; <uint>:1030 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1030		; <ubyte*>:2353 [#uses=1]
	load ubyte* %2353		; <ubyte>:3720 [#uses=1]
	seteq ubyte %3720, 0		; <bool>:1579 [#uses=1]
	br bool %1579, label %1583, label %1582

; <label>:1580		; preds = %1578, %1580
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1017		; <ubyte*>:2354 [#uses=2]
	load ubyte* %2354		; <ubyte>:3721 [#uses=2]
	add ubyte %3721, 255		; <ubyte>:3722 [#uses=1]
	store ubyte %3722, ubyte* %2354
	seteq ubyte %3721, 1		; <bool>:1580 [#uses=1]
	br bool %1580, label %1581, label %1580

; <label>:1581		; preds = %1578, %1580
	add uint %999, 4294967295		; <uint>:1031 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1031		; <ubyte*>:2355 [#uses=2]
	load ubyte* %2355		; <ubyte>:3723 [#uses=1]
	add ubyte %3723, 1		; <ubyte>:3724 [#uses=1]
	store ubyte %3724, ubyte* %2355
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1017		; <ubyte*>:2356 [#uses=1]
	load ubyte* %2356		; <ubyte>:3725 [#uses=1]
	seteq ubyte %3725, 0		; <bool>:1581 [#uses=1]
	br bool %1581, label %1579, label %1578

; <label>:1582		; preds = %1579, %1582
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1030		; <ubyte*>:2357 [#uses=2]
	load ubyte* %2357		; <ubyte>:3726 [#uses=2]
	add ubyte %3726, 255		; <ubyte>:3727 [#uses=1]
	store ubyte %3727, ubyte* %2357
	seteq ubyte %3726, 1		; <bool>:1582 [#uses=1]
	br bool %1582, label %1583, label %1582

; <label>:1583		; preds = %1579, %1582
	add uint %999, 4294967293		; <uint>:1032 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1032		; <ubyte*>:2358 [#uses=1]
	load ubyte* %2358		; <ubyte>:3728 [#uses=1]
	seteq ubyte %3728, 0		; <bool>:1583 [#uses=1]
	br bool %1583, label %1585, label %1584

; <label>:1584		; preds = %1583, %1587
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1032		; <ubyte*>:2359 [#uses=1]
	load ubyte* %2359		; <ubyte>:3729 [#uses=1]
	seteq ubyte %3729, 0		; <bool>:1584 [#uses=1]
	br bool %1584, label %1587, label %1586

; <label>:1585		; preds = %1583, %1587
	add uint %999, 4294967281		; <uint>:1033 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1033		; <ubyte*>:2360 [#uses=2]
	load ubyte* %2360		; <ubyte>:3730 [#uses=1]
	add ubyte %3730, 7		; <ubyte>:3731 [#uses=1]
	store ubyte %3731, ubyte* %2360
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1022		; <ubyte*>:2361 [#uses=1]
	load ubyte* %2361		; <ubyte>:3732 [#uses=1]
	seteq ubyte %3732, 0		; <bool>:1585 [#uses=1]
	br bool %1585, label %1589, label %1588

; <label>:1586		; preds = %1584, %1586
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1032		; <ubyte*>:2362 [#uses=2]
	load ubyte* %2362		; <ubyte>:3733 [#uses=2]
	add ubyte %3733, 255		; <ubyte>:3734 [#uses=1]
	store ubyte %3734, ubyte* %2362
	seteq ubyte %3733, 1		; <bool>:1586 [#uses=1]
	br bool %1586, label %1587, label %1586

; <label>:1587		; preds = %1584, %1586
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1022		; <ubyte*>:2363 [#uses=2]
	load ubyte* %2363		; <ubyte>:3735 [#uses=1]
	add ubyte %3735, 255		; <ubyte>:3736 [#uses=1]
	store ubyte %3736, ubyte* %2363
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1032		; <ubyte*>:2364 [#uses=1]
	load ubyte* %2364		; <ubyte>:3737 [#uses=1]
	seteq ubyte %3737, 0		; <bool>:1587 [#uses=1]
	br bool %1587, label %1585, label %1584

; <label>:1588		; preds = %1585, %1591
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1022		; <ubyte*>:2365 [#uses=1]
	load ubyte* %2365		; <ubyte>:3738 [#uses=1]
	seteq ubyte %3738, 0		; <bool>:1588 [#uses=1]
	br bool %1588, label %1591, label %1590

; <label>:1589		; preds = %1585, %1591
	add uint %999, 4294967283		; <uint>:1034 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1034		; <ubyte*>:2366 [#uses=1]
	load ubyte* %2366		; <ubyte>:3739 [#uses=1]
	seteq ubyte %3739, 0		; <bool>:1589 [#uses=1]
	br bool %1589, label %1397, label %1396

; <label>:1590		; preds = %1588, %1590
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1022		; <ubyte*>:2367 [#uses=2]
	load ubyte* %2367		; <ubyte>:3740 [#uses=2]
	add ubyte %3740, 255		; <ubyte>:3741 [#uses=1]
	store ubyte %3741, ubyte* %2367
	seteq ubyte %3740, 1		; <bool>:1590 [#uses=1]
	br bool %1590, label %1591, label %1590

; <label>:1591		; preds = %1588, %1590
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1033		; <ubyte*>:2368 [#uses=2]
	load ubyte* %2368		; <ubyte>:3742 [#uses=1]
	add ubyte %3742, 255		; <ubyte>:3743 [#uses=1]
	store ubyte %3743, ubyte* %2368
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1022		; <ubyte*>:2369 [#uses=1]
	load ubyte* %2369		; <ubyte>:3744 [#uses=1]
	seteq ubyte %3744, 0		; <bool>:1591 [#uses=1]
	br bool %1591, label %1589, label %1588

; <label>:1592		; preds = %577, %1645
	phi uint [ %393, %577 ], [ %1059, %1645 ]		; <uint>:1035 [#uses=23]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1035		; <ubyte*>:2370 [#uses=2]
	load ubyte* %2370		; <ubyte>:3745 [#uses=1]
	add ubyte %3745, 255		; <ubyte>:3746 [#uses=1]
	store ubyte %3746, ubyte* %2370
	add uint %1035, 10		; <uint>:1036 [#uses=17]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1036		; <ubyte*>:2371 [#uses=1]
	load ubyte* %2371		; <ubyte>:3747 [#uses=1]
	seteq ubyte %3747, 0		; <bool>:1592 [#uses=1]
	br bool %1592, label %1595, label %1594

; <label>:1593		; preds = %577, %1645
	phi uint [ %393, %577 ], [ %1059, %1645 ]		; <uint>:1037 [#uses=1]
	add uint %1037, 4294967295		; <uint>:1038 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1038		; <ubyte*>:2372 [#uses=1]
	load ubyte* %2372		; <ubyte>:3748 [#uses=1]
	seteq ubyte %3748, 0		; <bool>:1593 [#uses=1]
	br bool %1593, label %575, label %574

; <label>:1594		; preds = %1592, %1594
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1036		; <ubyte*>:2373 [#uses=2]
	load ubyte* %2373		; <ubyte>:3749 [#uses=2]
	add ubyte %3749, 255		; <ubyte>:3750 [#uses=1]
	store ubyte %3750, ubyte* %2373
	seteq ubyte %3749, 1		; <bool>:1594 [#uses=1]
	br bool %1594, label %1595, label %1594

; <label>:1595		; preds = %1592, %1594
	add uint %1035, 4		; <uint>:1039 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1039		; <ubyte*>:2374 [#uses=1]
	load ubyte* %2374		; <ubyte>:3751 [#uses=1]
	seteq ubyte %3751, 0		; <bool>:1595 [#uses=1]
	br bool %1595, label %1597, label %1596

; <label>:1596		; preds = %1595, %1596
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1039		; <ubyte*>:2375 [#uses=2]
	load ubyte* %2375		; <ubyte>:3752 [#uses=1]
	add ubyte %3752, 255		; <ubyte>:3753 [#uses=1]
	store ubyte %3753, ubyte* %2375
	add uint %1035, 5		; <uint>:1040 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1040		; <ubyte*>:2376 [#uses=2]
	load ubyte* %2376		; <ubyte>:3754 [#uses=1]
	add ubyte %3754, 1		; <ubyte>:3755 [#uses=1]
	store ubyte %3755, ubyte* %2376
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1036		; <ubyte*>:2377 [#uses=2]
	load ubyte* %2377		; <ubyte>:3756 [#uses=1]
	add ubyte %3756, 1		; <ubyte>:3757 [#uses=1]
	store ubyte %3757, ubyte* %2377
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1039		; <ubyte*>:2378 [#uses=1]
	load ubyte* %2378		; <ubyte>:3758 [#uses=1]
	seteq ubyte %3758, 0		; <bool>:1596 [#uses=1]
	br bool %1596, label %1597, label %1596

; <label>:1597		; preds = %1595, %1596
	add uint %1035, 5		; <uint>:1041 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1041		; <ubyte*>:2379 [#uses=1]
	load ubyte* %2379		; <ubyte>:3759 [#uses=1]
	seteq ubyte %3759, 0		; <bool>:1597 [#uses=1]
	br bool %1597, label %1599, label %1598

; <label>:1598		; preds = %1597, %1598
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1039		; <ubyte*>:2380 [#uses=2]
	load ubyte* %2380		; <ubyte>:3760 [#uses=1]
	add ubyte %3760, 1		; <ubyte>:3761 [#uses=1]
	store ubyte %3761, ubyte* %2380
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1041		; <ubyte*>:2381 [#uses=2]
	load ubyte* %2381		; <ubyte>:3762 [#uses=2]
	add ubyte %3762, 255		; <ubyte>:3763 [#uses=1]
	store ubyte %3763, ubyte* %2381
	seteq ubyte %3762, 1		; <bool>:1598 [#uses=1]
	br bool %1598, label %1599, label %1598

; <label>:1599		; preds = %1597, %1598
	add uint %1035, 12		; <uint>:1042 [#uses=12]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1042		; <ubyte*>:2382 [#uses=1]
	load ubyte* %2382		; <ubyte>:3764 [#uses=1]
	seteq ubyte %3764, 0		; <bool>:1599 [#uses=1]
	br bool %1599, label %1601, label %1600

; <label>:1600		; preds = %1599, %1600
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1042		; <ubyte*>:2383 [#uses=2]
	load ubyte* %2383		; <ubyte>:3765 [#uses=2]
	add ubyte %3765, 255		; <ubyte>:3766 [#uses=1]
	store ubyte %3766, ubyte* %2383
	seteq ubyte %3765, 1		; <bool>:1600 [#uses=1]
	br bool %1600, label %1601, label %1600

; <label>:1601		; preds = %1599, %1600
	add uint %1035, 6		; <uint>:1043 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1043		; <ubyte*>:2384 [#uses=1]
	load ubyte* %2384		; <ubyte>:3767 [#uses=1]
	seteq ubyte %3767, 0		; <bool>:1601 [#uses=1]
	br bool %1601, label %1603, label %1602

; <label>:1602		; preds = %1601, %1602
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1043		; <ubyte*>:2385 [#uses=2]
	load ubyte* %2385		; <ubyte>:3768 [#uses=1]
	add ubyte %3768, 255		; <ubyte>:3769 [#uses=1]
	store ubyte %3769, ubyte* %2385
	add uint %1035, 7		; <uint>:1044 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1044		; <ubyte*>:2386 [#uses=2]
	load ubyte* %2386		; <ubyte>:3770 [#uses=1]
	add ubyte %3770, 1		; <ubyte>:3771 [#uses=1]
	store ubyte %3771, ubyte* %2386
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1042		; <ubyte*>:2387 [#uses=2]
	load ubyte* %2387		; <ubyte>:3772 [#uses=1]
	add ubyte %3772, 1		; <ubyte>:3773 [#uses=1]
	store ubyte %3773, ubyte* %2387
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1043		; <ubyte*>:2388 [#uses=1]
	load ubyte* %2388		; <ubyte>:3774 [#uses=1]
	seteq ubyte %3774, 0		; <bool>:1602 [#uses=1]
	br bool %1602, label %1603, label %1602

; <label>:1603		; preds = %1601, %1602
	add uint %1035, 7		; <uint>:1045 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1045		; <ubyte*>:2389 [#uses=1]
	load ubyte* %2389		; <ubyte>:3775 [#uses=1]
	seteq ubyte %3775, 0		; <bool>:1603 [#uses=1]
	br bool %1603, label %1605, label %1604

; <label>:1604		; preds = %1603, %1604
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1043		; <ubyte*>:2390 [#uses=2]
	load ubyte* %2390		; <ubyte>:3776 [#uses=1]
	add ubyte %3776, 1		; <ubyte>:3777 [#uses=1]
	store ubyte %3777, ubyte* %2390
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1045		; <ubyte*>:2391 [#uses=2]
	load ubyte* %2391		; <ubyte>:3778 [#uses=2]
	add ubyte %3778, 255		; <ubyte>:3779 [#uses=1]
	store ubyte %3779, ubyte* %2391
	seteq ubyte %3778, 1		; <bool>:1604 [#uses=1]
	br bool %1604, label %1605, label %1604

; <label>:1605		; preds = %1603, %1604
	add uint %1035, 14		; <uint>:1046 [#uses=9]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1046		; <ubyte*>:2392 [#uses=2]
	load ubyte* %2392		; <ubyte>:3780 [#uses=2]
	add ubyte %3780, 1		; <ubyte>:3781 [#uses=1]
	store ubyte %3781, ubyte* %2392
	seteq ubyte %3780, 255		; <bool>:1605 [#uses=1]
	br bool %1605, label %1607, label %1606

; <label>:1606		; preds = %1605, %1629
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1046		; <ubyte*>:2393 [#uses=2]
	load ubyte* %2393		; <ubyte>:3782 [#uses=1]
	add ubyte %3782, 1		; <ubyte>:3783 [#uses=1]
	store ubyte %3783, ubyte* %2393
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1036		; <ubyte*>:2394 [#uses=1]
	load ubyte* %2394		; <ubyte>:3784 [#uses=1]
	seteq ubyte %3784, 0		; <bool>:1606 [#uses=1]
	br bool %1606, label %1609, label %1608

; <label>:1607		; preds = %1605, %1629
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1036		; <ubyte*>:2395 [#uses=1]
	load ubyte* %2395		; <ubyte>:3785 [#uses=1]
	seteq ubyte %3785, 0		; <bool>:1607 [#uses=1]
	br bool %1607, label %1631, label %1630

; <label>:1608		; preds = %1606, %1608
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1036		; <ubyte*>:2396 [#uses=2]
	load ubyte* %2396		; <ubyte>:3786 [#uses=1]
	add ubyte %3786, 255		; <ubyte>:3787 [#uses=1]
	store ubyte %3787, ubyte* %2396
	add uint %1035, 11		; <uint>:1047 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1047		; <ubyte*>:2397 [#uses=2]
	load ubyte* %2397		; <ubyte>:3788 [#uses=1]
	add ubyte %3788, 1		; <ubyte>:3789 [#uses=1]
	store ubyte %3789, ubyte* %2397
	add uint %1035, 15		; <uint>:1048 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1048		; <ubyte*>:2398 [#uses=2]
	load ubyte* %2398		; <ubyte>:3790 [#uses=1]
	add ubyte %3790, 1		; <ubyte>:3791 [#uses=1]
	store ubyte %3791, ubyte* %2398
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1036		; <ubyte*>:2399 [#uses=1]
	load ubyte* %2399		; <ubyte>:3792 [#uses=1]
	seteq ubyte %3792, 0		; <bool>:1608 [#uses=1]
	br bool %1608, label %1609, label %1608

; <label>:1609		; preds = %1606, %1608
	add uint %1035, 11		; <uint>:1049 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1049		; <ubyte*>:2400 [#uses=1]
	load ubyte* %2400		; <ubyte>:3793 [#uses=1]
	seteq ubyte %3793, 0		; <bool>:1609 [#uses=1]
	br bool %1609, label %1611, label %1610

; <label>:1610		; preds = %1609, %1610
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1036		; <ubyte*>:2401 [#uses=2]
	load ubyte* %2401		; <ubyte>:3794 [#uses=1]
	add ubyte %3794, 1		; <ubyte>:3795 [#uses=1]
	store ubyte %3795, ubyte* %2401
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1049		; <ubyte*>:2402 [#uses=2]
	load ubyte* %2402		; <ubyte>:3796 [#uses=2]
	add ubyte %3796, 255		; <ubyte>:3797 [#uses=1]
	store ubyte %3797, ubyte* %2402
	seteq ubyte %3796, 1		; <bool>:1610 [#uses=1]
	br bool %1610, label %1611, label %1610

; <label>:1611		; preds = %1609, %1610
	add uint %1035, 15		; <uint>:1050 [#uses=13]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1050		; <ubyte*>:2403 [#uses=1]
	load ubyte* %2403		; <ubyte>:3798 [#uses=1]
	seteq ubyte %3798, 0		; <bool>:1611 [#uses=1]
	br bool %1611, label %1613, label %1612

; <label>:1612		; preds = %1611, %1615
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1050		; <ubyte*>:2404 [#uses=1]
	load ubyte* %2404		; <ubyte>:3799 [#uses=1]
	seteq ubyte %3799, 0		; <bool>:1612 [#uses=1]
	br bool %1612, label %1615, label %1614

; <label>:1613		; preds = %1611, %1615
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1042		; <ubyte*>:2405 [#uses=1]
	load ubyte* %2405		; <ubyte>:3800 [#uses=1]
	seteq ubyte %3800, 0		; <bool>:1613 [#uses=1]
	br bool %1613, label %1617, label %1616

; <label>:1614		; preds = %1612, %1614
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1050		; <ubyte*>:2406 [#uses=2]
	load ubyte* %2406		; <ubyte>:3801 [#uses=2]
	add ubyte %3801, 255		; <ubyte>:3802 [#uses=1]
	store ubyte %3802, ubyte* %2406
	seteq ubyte %3801, 1		; <bool>:1614 [#uses=1]
	br bool %1614, label %1615, label %1614

; <label>:1615		; preds = %1612, %1614
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1046		; <ubyte*>:2407 [#uses=2]
	load ubyte* %2407		; <ubyte>:3803 [#uses=1]
	add ubyte %3803, 255		; <ubyte>:3804 [#uses=1]
	store ubyte %3804, ubyte* %2407
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1050		; <ubyte*>:2408 [#uses=1]
	load ubyte* %2408		; <ubyte>:3805 [#uses=1]
	seteq ubyte %3805, 0		; <bool>:1615 [#uses=1]
	br bool %1615, label %1613, label %1612

; <label>:1616		; preds = %1613, %1616
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1042		; <ubyte*>:2409 [#uses=2]
	load ubyte* %2409		; <ubyte>:3806 [#uses=1]
	add ubyte %3806, 255		; <ubyte>:3807 [#uses=1]
	store ubyte %3807, ubyte* %2409
	add uint %1035, 13		; <uint>:1051 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1051		; <ubyte*>:2410 [#uses=2]
	load ubyte* %2410		; <ubyte>:3808 [#uses=1]
	add ubyte %3808, 1		; <ubyte>:3809 [#uses=1]
	store ubyte %3809, ubyte* %2410
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1050		; <ubyte*>:2411 [#uses=2]
	load ubyte* %2411		; <ubyte>:3810 [#uses=1]
	add ubyte %3810, 1		; <ubyte>:3811 [#uses=1]
	store ubyte %3811, ubyte* %2411
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1042		; <ubyte*>:2412 [#uses=1]
	load ubyte* %2412		; <ubyte>:3812 [#uses=1]
	seteq ubyte %3812, 0		; <bool>:1616 [#uses=1]
	br bool %1616, label %1617, label %1616

; <label>:1617		; preds = %1613, %1616
	add uint %1035, 13		; <uint>:1052 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1052		; <ubyte*>:2413 [#uses=1]
	load ubyte* %2413		; <ubyte>:3813 [#uses=1]
	seteq ubyte %3813, 0		; <bool>:1617 [#uses=1]
	br bool %1617, label %1619, label %1618

; <label>:1618		; preds = %1617, %1618
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1042		; <ubyte*>:2414 [#uses=2]
	load ubyte* %2414		; <ubyte>:3814 [#uses=1]
	add ubyte %3814, 1		; <ubyte>:3815 [#uses=1]
	store ubyte %3815, ubyte* %2414
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1052		; <ubyte*>:2415 [#uses=2]
	load ubyte* %2415		; <ubyte>:3816 [#uses=2]
	add ubyte %3816, 255		; <ubyte>:3817 [#uses=1]
	store ubyte %3817, ubyte* %2415
	seteq ubyte %3816, 1		; <bool>:1618 [#uses=1]
	br bool %1618, label %1619, label %1618

; <label>:1619		; preds = %1617, %1618
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1050		; <ubyte*>:2416 [#uses=1]
	load ubyte* %2416		; <ubyte>:3818 [#uses=1]
	seteq ubyte %3818, 0		; <bool>:1619 [#uses=1]
	br bool %1619, label %1621, label %1620

; <label>:1620		; preds = %1619, %1623
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1050		; <ubyte*>:2417 [#uses=1]
	load ubyte* %2417		; <ubyte>:3819 [#uses=1]
	seteq ubyte %3819, 0		; <bool>:1620 [#uses=1]
	br bool %1620, label %1623, label %1622

; <label>:1621		; preds = %1619, %1623
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1050		; <ubyte*>:2418 [#uses=2]
	load ubyte* %2418		; <ubyte>:3820 [#uses=1]
	add ubyte %3820, 1		; <ubyte>:3821 [#uses=1]
	store ubyte %3821, ubyte* %2418
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1046		; <ubyte*>:2419 [#uses=1]
	load ubyte* %2419		; <ubyte>:3822 [#uses=1]
	seteq ubyte %3822, 0		; <bool>:1621 [#uses=1]
	br bool %1621, label %1625, label %1624

; <label>:1622		; preds = %1620, %1622
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1050		; <ubyte*>:2420 [#uses=2]
	load ubyte* %2420		; <ubyte>:3823 [#uses=2]
	add ubyte %3823, 255		; <ubyte>:3824 [#uses=1]
	store ubyte %3824, ubyte* %2420
	seteq ubyte %3823, 1		; <bool>:1622 [#uses=1]
	br bool %1622, label %1623, label %1622

; <label>:1623		; preds = %1620, %1622
	add uint %1035, 14		; <uint>:1053 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1053		; <ubyte*>:2421 [#uses=2]
	load ubyte* %2421		; <ubyte>:3825 [#uses=1]
	add ubyte %3825, 255		; <ubyte>:3826 [#uses=1]
	store ubyte %3826, ubyte* %2421
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1050		; <ubyte*>:2422 [#uses=1]
	load ubyte* %2422		; <ubyte>:3827 [#uses=1]
	seteq ubyte %3827, 0		; <bool>:1623 [#uses=1]
	br bool %1623, label %1621, label %1620

; <label>:1624		; preds = %1621, %1627
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1046		; <ubyte*>:2423 [#uses=1]
	load ubyte* %2423		; <ubyte>:3828 [#uses=1]
	seteq ubyte %3828, 0		; <bool>:1624 [#uses=1]
	br bool %1624, label %1627, label %1626

; <label>:1625		; preds = %1621, %1627
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1050		; <ubyte*>:2424 [#uses=1]
	load ubyte* %2424		; <ubyte>:3829 [#uses=1]
	seteq ubyte %3829, 0		; <bool>:1625 [#uses=1]
	br bool %1625, label %1629, label %1628

; <label>:1626		; preds = %1624, %1626
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1046		; <ubyte*>:2425 [#uses=2]
	load ubyte* %2425		; <ubyte>:3830 [#uses=2]
	add ubyte %3830, 255		; <ubyte>:3831 [#uses=1]
	store ubyte %3831, ubyte* %2425
	seteq ubyte %3830, 1		; <bool>:1626 [#uses=1]
	br bool %1626, label %1627, label %1626

; <label>:1627		; preds = %1624, %1626
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1050		; <ubyte*>:2426 [#uses=2]
	load ubyte* %2426		; <ubyte>:3832 [#uses=1]
	add ubyte %3832, 255		; <ubyte>:3833 [#uses=1]
	store ubyte %3833, ubyte* %2426
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1046		; <ubyte*>:2427 [#uses=1]
	load ubyte* %2427		; <ubyte>:3834 [#uses=1]
	seteq ubyte %3834, 0		; <bool>:1627 [#uses=1]
	br bool %1627, label %1625, label %1624

; <label>:1628		; preds = %1625, %1628
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1036		; <ubyte*>:2428 [#uses=2]
	load ubyte* %2428		; <ubyte>:3835 [#uses=1]
	add ubyte %3835, 255		; <ubyte>:3836 [#uses=1]
	store ubyte %3836, ubyte* %2428
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1042		; <ubyte*>:2429 [#uses=2]
	load ubyte* %2429		; <ubyte>:3837 [#uses=1]
	add ubyte %3837, 255		; <ubyte>:3838 [#uses=1]
	store ubyte %3838, ubyte* %2429
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1046		; <ubyte*>:2430 [#uses=2]
	load ubyte* %2430		; <ubyte>:3839 [#uses=1]
	add ubyte %3839, 1		; <ubyte>:3840 [#uses=1]
	store ubyte %3840, ubyte* %2430
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1050		; <ubyte*>:2431 [#uses=2]
	load ubyte* %2431		; <ubyte>:3841 [#uses=2]
	add ubyte %3841, 255		; <ubyte>:3842 [#uses=1]
	store ubyte %3842, ubyte* %2431
	seteq ubyte %3841, 1		; <bool>:1628 [#uses=1]
	br bool %1628, label %1629, label %1628

; <label>:1629		; preds = %1625, %1628
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1046		; <ubyte*>:2432 [#uses=1]
	load ubyte* %2432		; <ubyte>:3843 [#uses=1]
	seteq ubyte %3843, 0		; <bool>:1629 [#uses=1]
	br bool %1629, label %1607, label %1606

; <label>:1630		; preds = %1607, %1633
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1036		; <ubyte*>:2433 [#uses=1]
	load ubyte* %2433		; <ubyte>:3844 [#uses=1]
	seteq ubyte %3844, 0		; <bool>:1630 [#uses=1]
	br bool %1630, label %1633, label %1632

; <label>:1631		; preds = %1607, %1633
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1042		; <ubyte*>:2434 [#uses=1]
	load ubyte* %2434		; <ubyte>:3845 [#uses=1]
	seteq ubyte %3845, 0		; <bool>:1631 [#uses=1]
	br bool %1631, label %1635, label %1634

; <label>:1632		; preds = %1630, %1632
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1036		; <ubyte*>:2435 [#uses=2]
	load ubyte* %2435		; <ubyte>:3846 [#uses=2]
	add ubyte %3846, 255		; <ubyte>:3847 [#uses=1]
	store ubyte %3847, ubyte* %2435
	seteq ubyte %3846, 1		; <bool>:1632 [#uses=1]
	br bool %1632, label %1633, label %1632

; <label>:1633		; preds = %1630, %1632
	add uint %1035, 11		; <uint>:1054 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1054		; <ubyte*>:2436 [#uses=2]
	load ubyte* %2436		; <ubyte>:3848 [#uses=1]
	add ubyte %3848, 1		; <ubyte>:3849 [#uses=1]
	store ubyte %3849, ubyte* %2436
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1036		; <ubyte*>:2437 [#uses=1]
	load ubyte* %2437		; <ubyte>:3850 [#uses=1]
	seteq ubyte %3850, 0		; <bool>:1633 [#uses=1]
	br bool %1633, label %1631, label %1630

; <label>:1634		; preds = %1631, %1637
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1042		; <ubyte*>:2438 [#uses=1]
	load ubyte* %2438		; <ubyte>:3851 [#uses=1]
	seteq ubyte %3851, 0		; <bool>:1634 [#uses=1]
	br bool %1634, label %1637, label %1636

; <label>:1635		; preds = %1631, %1637
	add uint %1035, 11		; <uint>:1055 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1055		; <ubyte*>:2439 [#uses=1]
	load ubyte* %2439		; <ubyte>:3852 [#uses=1]
	seteq ubyte %3852, 0		; <bool>:1635 [#uses=1]
	br bool %1635, label %1639, label %1638

; <label>:1636		; preds = %1634, %1636
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1042		; <ubyte*>:2440 [#uses=2]
	load ubyte* %2440		; <ubyte>:3853 [#uses=2]
	add ubyte %3853, 255		; <ubyte>:3854 [#uses=1]
	store ubyte %3854, ubyte* %2440
	seteq ubyte %3853, 1		; <bool>:1636 [#uses=1]
	br bool %1636, label %1637, label %1636

; <label>:1637		; preds = %1634, %1636
	add uint %1035, 13		; <uint>:1056 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1056		; <ubyte*>:2441 [#uses=2]
	load ubyte* %2441		; <ubyte>:3855 [#uses=1]
	add ubyte %3855, 1		; <ubyte>:3856 [#uses=1]
	store ubyte %3856, ubyte* %2441
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1042		; <ubyte*>:2442 [#uses=1]
	load ubyte* %2442		; <ubyte>:3857 [#uses=1]
	seteq ubyte %3857, 0		; <bool>:1637 [#uses=1]
	br bool %1637, label %1635, label %1634

; <label>:1638		; preds = %1635, %1638
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1055		; <ubyte*>:2443 [#uses=2]
	load ubyte* %2443		; <ubyte>:3858 [#uses=2]
	add ubyte %3858, 255		; <ubyte>:3859 [#uses=1]
	store ubyte %3859, ubyte* %2443
	seteq ubyte %3858, 1		; <bool>:1638 [#uses=1]
	br bool %1638, label %1639, label %1638

; <label>:1639		; preds = %1635, %1638
	add uint %1035, 13		; <uint>:1057 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1057		; <ubyte*>:2444 [#uses=1]
	load ubyte* %2444		; <ubyte>:3860 [#uses=1]
	seteq ubyte %3860, 0		; <bool>:1639 [#uses=1]
	br bool %1639, label %1641, label %1640

; <label>:1640		; preds = %1639, %1643
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1057		; <ubyte*>:2445 [#uses=1]
	load ubyte* %2445		; <ubyte>:3861 [#uses=1]
	seteq ubyte %3861, 0		; <bool>:1640 [#uses=1]
	br bool %1640, label %1643, label %1642

; <label>:1641		; preds = %1639, %1643
	add uint %1035, 4294967295		; <uint>:1058 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1058		; <ubyte*>:2446 [#uses=2]
	load ubyte* %2446		; <ubyte>:3862 [#uses=1]
	add ubyte %3862, 2		; <ubyte>:3863 [#uses=1]
	store ubyte %3863, ubyte* %2446
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1036		; <ubyte*>:2447 [#uses=1]
	load ubyte* %2447		; <ubyte>:3864 [#uses=1]
	seteq ubyte %3864, 0		; <bool>:1641 [#uses=1]
	br bool %1641, label %1645, label %1644

; <label>:1642		; preds = %1640, %1642
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1057		; <ubyte*>:2448 [#uses=2]
	load ubyte* %2448		; <ubyte>:3865 [#uses=2]
	add ubyte %3865, 255		; <ubyte>:3866 [#uses=1]
	store ubyte %3866, ubyte* %2448
	seteq ubyte %3865, 1		; <bool>:1642 [#uses=1]
	br bool %1642, label %1643, label %1642

; <label>:1643		; preds = %1640, %1642
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1036		; <ubyte*>:2449 [#uses=2]
	load ubyte* %2449		; <ubyte>:3867 [#uses=1]
	add ubyte %3867, 1		; <ubyte>:3868 [#uses=1]
	store ubyte %3868, ubyte* %2449
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1057		; <ubyte*>:2450 [#uses=1]
	load ubyte* %2450		; <ubyte>:3869 [#uses=1]
	seteq ubyte %3869, 0		; <bool>:1643 [#uses=1]
	br bool %1643, label %1641, label %1640

; <label>:1644		; preds = %1641, %1647
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1036		; <ubyte*>:2451 [#uses=1]
	load ubyte* %2451		; <ubyte>:3870 [#uses=1]
	seteq ubyte %3870, 0		; <bool>:1644 [#uses=1]
	br bool %1644, label %1647, label %1646

; <label>:1645		; preds = %1641, %1647
	add uint %1035, 1		; <uint>:1059 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1059		; <ubyte*>:2452 [#uses=1]
	load ubyte* %2452		; <ubyte>:3871 [#uses=1]
	seteq ubyte %3871, 0		; <bool>:1645 [#uses=1]
	br bool %1645, label %1593, label %1592

; <label>:1646		; preds = %1644, %1646
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1036		; <ubyte*>:2453 [#uses=2]
	load ubyte* %2453		; <ubyte>:3872 [#uses=2]
	add ubyte %3872, 255		; <ubyte>:3873 [#uses=1]
	store ubyte %3873, ubyte* %2453
	seteq ubyte %3872, 1		; <bool>:1646 [#uses=1]
	br bool %1646, label %1647, label %1646

; <label>:1647		; preds = %1644, %1646
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1058		; <ubyte*>:2454 [#uses=2]
	load ubyte* %2454		; <ubyte>:3874 [#uses=1]
	add ubyte %3874, 2		; <ubyte>:3875 [#uses=1]
	store ubyte %3875, ubyte* %2454
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1036		; <ubyte*>:2455 [#uses=1]
	load ubyte* %2455		; <ubyte>:3876 [#uses=1]
	seteq ubyte %3876, 0		; <bool>:1647 [#uses=1]
	br bool %1647, label %1645, label %1644

; <label>:1648		; preds = %575, %2073
	phi uint [ %390, %575 ], [ %1301, %2073 ]		; <uint>:1060 [#uses=71]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1060		; <ubyte*>:2456 [#uses=2]
	load ubyte* %2456		; <ubyte>:3877 [#uses=1]
	add ubyte %3877, 255		; <ubyte>:3878 [#uses=1]
	store ubyte %3878, ubyte* %2456
	add uint %1060, 10		; <uint>:1061 [#uses=6]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1061		; <ubyte*>:2457 [#uses=1]
	load ubyte* %2457		; <ubyte>:3879 [#uses=1]
	seteq ubyte %3879, 0		; <bool>:1648 [#uses=1]
	br bool %1648, label %1651, label %1650

; <label>:1649		; preds = %575, %2073
	phi uint [ %390, %575 ], [ %1301, %2073 ]		; <uint>:1062 [#uses=1]
	add uint %1062, 4294967295		; <uint>:1063 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1063		; <ubyte*>:2458 [#uses=1]
	load ubyte* %2458		; <ubyte>:3880 [#uses=1]
	seteq ubyte %3880, 0		; <bool>:1649 [#uses=1]
	br bool %1649, label %573, label %572

; <label>:1650		; preds = %1648, %1650
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1061		; <ubyte*>:2459 [#uses=2]
	load ubyte* %2459		; <ubyte>:3881 [#uses=2]
	add ubyte %3881, 255		; <ubyte>:3882 [#uses=1]
	store ubyte %3882, ubyte* %2459
	seteq ubyte %3881, 1		; <bool>:1650 [#uses=1]
	br bool %1650, label %1651, label %1650

; <label>:1651		; preds = %1648, %1650
	add uint %1060, 4		; <uint>:1064 [#uses=7]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1064		; <ubyte*>:2460 [#uses=1]
	load ubyte* %2460		; <ubyte>:3883 [#uses=1]
	seteq ubyte %3883, 0		; <bool>:1651 [#uses=1]
	br bool %1651, label %1653, label %1652

; <label>:1652		; preds = %1651, %1652
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1064		; <ubyte*>:2461 [#uses=2]
	load ubyte* %2461		; <ubyte>:3884 [#uses=1]
	add ubyte %3884, 255		; <ubyte>:3885 [#uses=1]
	store ubyte %3885, ubyte* %2461
	add uint %1060, 5		; <uint>:1065 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1065		; <ubyte*>:2462 [#uses=2]
	load ubyte* %2462		; <ubyte>:3886 [#uses=1]
	add ubyte %3886, 1		; <ubyte>:3887 [#uses=1]
	store ubyte %3887, ubyte* %2462
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1061		; <ubyte*>:2463 [#uses=2]
	load ubyte* %2463		; <ubyte>:3888 [#uses=1]
	add ubyte %3888, 1		; <ubyte>:3889 [#uses=1]
	store ubyte %3889, ubyte* %2463
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1064		; <ubyte*>:2464 [#uses=1]
	load ubyte* %2464		; <ubyte>:3890 [#uses=1]
	seteq ubyte %3890, 0		; <bool>:1652 [#uses=1]
	br bool %1652, label %1653, label %1652

; <label>:1653		; preds = %1651, %1652
	add uint %1060, 5		; <uint>:1066 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1066		; <ubyte*>:2465 [#uses=1]
	load ubyte* %2465		; <ubyte>:3891 [#uses=1]
	seteq ubyte %3891, 0		; <bool>:1653 [#uses=1]
	br bool %1653, label %1655, label %1654

; <label>:1654		; preds = %1653, %1654
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1064		; <ubyte*>:2466 [#uses=2]
	load ubyte* %2466		; <ubyte>:3892 [#uses=1]
	add ubyte %3892, 1		; <ubyte>:3893 [#uses=1]
	store ubyte %3893, ubyte* %2466
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1066		; <ubyte*>:2467 [#uses=2]
	load ubyte* %2467		; <ubyte>:3894 [#uses=2]
	add ubyte %3894, 255		; <ubyte>:3895 [#uses=1]
	store ubyte %3895, ubyte* %2467
	seteq ubyte %3894, 1		; <bool>:1654 [#uses=1]
	br bool %1654, label %1655, label %1654

; <label>:1655		; preds = %1653, %1654
	add uint %1060, 12		; <uint>:1067 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1067		; <ubyte*>:2468 [#uses=2]
	load ubyte* %2468		; <ubyte>:3896 [#uses=2]
	add ubyte %3896, 1		; <ubyte>:3897 [#uses=1]
	store ubyte %3897, ubyte* %2468
	seteq ubyte %3896, 255		; <bool>:1655 [#uses=1]
	br bool %1655, label %1657, label %1656

; <label>:1656		; preds = %1655, %1656
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1061		; <ubyte*>:2469 [#uses=2]
	load ubyte* %2469		; <ubyte>:3898 [#uses=1]
	add ubyte %3898, 255		; <ubyte>:3899 [#uses=1]
	store ubyte %3899, ubyte* %2469
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1067		; <ubyte*>:2470 [#uses=2]
	load ubyte* %2470		; <ubyte>:3900 [#uses=2]
	add ubyte %3900, 255		; <ubyte>:3901 [#uses=1]
	store ubyte %3901, ubyte* %2470
	seteq ubyte %3900, 1		; <bool>:1656 [#uses=1]
	br bool %1656, label %1657, label %1656

; <label>:1657		; preds = %1655, %1656
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1064		; <ubyte*>:2471 [#uses=1]
	load ubyte* %2471		; <ubyte>:3902 [#uses=1]
	seteq ubyte %3902, 0		; <bool>:1657 [#uses=1]
	br bool %1657, label %1659, label %1658

; <label>:1658		; preds = %1657, %1658
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1064		; <ubyte*>:2472 [#uses=2]
	load ubyte* %2472		; <ubyte>:3903 [#uses=2]
	add ubyte %3903, 255		; <ubyte>:3904 [#uses=1]
	store ubyte %3904, ubyte* %2472
	seteq ubyte %3903, 1		; <bool>:1658 [#uses=1]
	br bool %1658, label %1659, label %1658

; <label>:1659		; preds = %1657, %1658
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1061		; <ubyte*>:2473 [#uses=1]
	load ubyte* %2473		; <ubyte>:3905 [#uses=1]
	seteq ubyte %3905, 0		; <bool>:1659 [#uses=1]
	br bool %1659, label %1661, label %1660

; <label>:1660		; preds = %1659, %1660
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1064		; <ubyte*>:2474 [#uses=2]
	load ubyte* %2474		; <ubyte>:3906 [#uses=1]
	add ubyte %3906, 1		; <ubyte>:3907 [#uses=1]
	store ubyte %3907, ubyte* %2474
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1061		; <ubyte*>:2475 [#uses=2]
	load ubyte* %2475		; <ubyte>:3908 [#uses=2]
	add ubyte %3908, 255		; <ubyte>:3909 [#uses=1]
	store ubyte %3909, ubyte* %2475
	seteq ubyte %3908, 1		; <bool>:1660 [#uses=1]
	br bool %1660, label %1661, label %1660

; <label>:1661		; preds = %1659, %1660
	add uint %1060, 18		; <uint>:1068 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1068		; <ubyte*>:2476 [#uses=1]
	load ubyte* %2476		; <ubyte>:3910 [#uses=1]
	seteq ubyte %3910, 0		; <bool>:1661 [#uses=1]
	br bool %1661, label %1663, label %1662

; <label>:1662		; preds = %1661, %1662
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1068		; <ubyte*>:2477 [#uses=2]
	load ubyte* %2477		; <ubyte>:3911 [#uses=2]
	add ubyte %3911, 255		; <ubyte>:3912 [#uses=1]
	store ubyte %3912, ubyte* %2477
	seteq ubyte %3911, 1		; <bool>:1662 [#uses=1]
	br bool %1662, label %1663, label %1662

; <label>:1663		; preds = %1661, %1662
	add uint %1060, 4294967201		; <uint>:1069 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1069		; <ubyte*>:2478 [#uses=1]
	load ubyte* %2478		; <ubyte>:3913 [#uses=1]
	seteq ubyte %3913, 0		; <bool>:1663 [#uses=1]
	br bool %1663, label %1665, label %1664

; <label>:1664		; preds = %1663, %1664
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1069		; <ubyte*>:2479 [#uses=2]
	load ubyte* %2479		; <ubyte>:3914 [#uses=1]
	add ubyte %3914, 255		; <ubyte>:3915 [#uses=1]
	store ubyte %3915, ubyte* %2479
	add uint %1060, 4294967202		; <uint>:1070 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1070		; <ubyte*>:2480 [#uses=2]
	load ubyte* %2480		; <ubyte>:3916 [#uses=1]
	add ubyte %3916, 1		; <ubyte>:3917 [#uses=1]
	store ubyte %3917, ubyte* %2480
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1068		; <ubyte*>:2481 [#uses=2]
	load ubyte* %2481		; <ubyte>:3918 [#uses=1]
	add ubyte %3918, 1		; <ubyte>:3919 [#uses=1]
	store ubyte %3919, ubyte* %2481
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1069		; <ubyte*>:2482 [#uses=1]
	load ubyte* %2482		; <ubyte>:3920 [#uses=1]
	seteq ubyte %3920, 0		; <bool>:1664 [#uses=1]
	br bool %1664, label %1665, label %1664

; <label>:1665		; preds = %1663, %1664
	add uint %1060, 4294967202		; <uint>:1071 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1071		; <ubyte*>:2483 [#uses=1]
	load ubyte* %2483		; <ubyte>:3921 [#uses=1]
	seteq ubyte %3921, 0		; <bool>:1665 [#uses=1]
	br bool %1665, label %1667, label %1666

; <label>:1666		; preds = %1665, %1666
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1069		; <ubyte*>:2484 [#uses=2]
	load ubyte* %2484		; <ubyte>:3922 [#uses=1]
	add ubyte %3922, 1		; <ubyte>:3923 [#uses=1]
	store ubyte %3923, ubyte* %2484
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1071		; <ubyte*>:2485 [#uses=2]
	load ubyte* %2485		; <ubyte>:3924 [#uses=2]
	add ubyte %3924, 255		; <ubyte>:3925 [#uses=1]
	store ubyte %3925, ubyte* %2485
	seteq ubyte %3924, 1		; <bool>:1666 [#uses=1]
	br bool %1666, label %1667, label %1666

; <label>:1667		; preds = %1665, %1666
	add uint %1060, 24		; <uint>:1072 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1072		; <ubyte*>:2486 [#uses=1]
	load ubyte* %2486		; <ubyte>:3926 [#uses=1]
	seteq ubyte %3926, 0		; <bool>:1667 [#uses=1]
	br bool %1667, label %1669, label %1668

; <label>:1668		; preds = %1667, %1668
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1072		; <ubyte*>:2487 [#uses=2]
	load ubyte* %2487		; <ubyte>:3927 [#uses=2]
	add ubyte %3927, 255		; <ubyte>:3928 [#uses=1]
	store ubyte %3928, ubyte* %2487
	seteq ubyte %3927, 1		; <bool>:1668 [#uses=1]
	br bool %1668, label %1669, label %1668

; <label>:1669		; preds = %1667, %1668
	add uint %1060, 4294967207		; <uint>:1073 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1073		; <ubyte*>:2488 [#uses=1]
	load ubyte* %2488		; <ubyte>:3929 [#uses=1]
	seteq ubyte %3929, 0		; <bool>:1669 [#uses=1]
	br bool %1669, label %1671, label %1670

; <label>:1670		; preds = %1669, %1670
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1073		; <ubyte*>:2489 [#uses=2]
	load ubyte* %2489		; <ubyte>:3930 [#uses=1]
	add ubyte %3930, 255		; <ubyte>:3931 [#uses=1]
	store ubyte %3931, ubyte* %2489
	add uint %1060, 4294967208		; <uint>:1074 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1074		; <ubyte*>:2490 [#uses=2]
	load ubyte* %2490		; <ubyte>:3932 [#uses=1]
	add ubyte %3932, 1		; <ubyte>:3933 [#uses=1]
	store ubyte %3933, ubyte* %2490
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1072		; <ubyte*>:2491 [#uses=2]
	load ubyte* %2491		; <ubyte>:3934 [#uses=1]
	add ubyte %3934, 1		; <ubyte>:3935 [#uses=1]
	store ubyte %3935, ubyte* %2491
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1073		; <ubyte*>:2492 [#uses=1]
	load ubyte* %2492		; <ubyte>:3936 [#uses=1]
	seteq ubyte %3936, 0		; <bool>:1670 [#uses=1]
	br bool %1670, label %1671, label %1670

; <label>:1671		; preds = %1669, %1670
	add uint %1060, 4294967208		; <uint>:1075 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1075		; <ubyte*>:2493 [#uses=1]
	load ubyte* %2493		; <ubyte>:3937 [#uses=1]
	seteq ubyte %3937, 0		; <bool>:1671 [#uses=1]
	br bool %1671, label %1673, label %1672

; <label>:1672		; preds = %1671, %1672
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1073		; <ubyte*>:2494 [#uses=2]
	load ubyte* %2494		; <ubyte>:3938 [#uses=1]
	add ubyte %3938, 1		; <ubyte>:3939 [#uses=1]
	store ubyte %3939, ubyte* %2494
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1075		; <ubyte*>:2495 [#uses=2]
	load ubyte* %2495		; <ubyte>:3940 [#uses=2]
	add ubyte %3940, 255		; <ubyte>:3941 [#uses=1]
	store ubyte %3941, ubyte* %2495
	seteq ubyte %3940, 1		; <bool>:1672 [#uses=1]
	br bool %1672, label %1673, label %1672

; <label>:1673		; preds = %1671, %1672
	add uint %1060, 30		; <uint>:1076 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1076		; <ubyte*>:2496 [#uses=1]
	load ubyte* %2496		; <ubyte>:3942 [#uses=1]
	seteq ubyte %3942, 0		; <bool>:1673 [#uses=1]
	br bool %1673, label %1675, label %1674

; <label>:1674		; preds = %1673, %1674
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1076		; <ubyte*>:2497 [#uses=2]
	load ubyte* %2497		; <ubyte>:3943 [#uses=2]
	add ubyte %3943, 255		; <ubyte>:3944 [#uses=1]
	store ubyte %3944, ubyte* %2497
	seteq ubyte %3943, 1		; <bool>:1674 [#uses=1]
	br bool %1674, label %1675, label %1674

; <label>:1675		; preds = %1673, %1674
	add uint %1060, 4294967213		; <uint>:1077 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1077		; <ubyte*>:2498 [#uses=1]
	load ubyte* %2498		; <ubyte>:3945 [#uses=1]
	seteq ubyte %3945, 0		; <bool>:1675 [#uses=1]
	br bool %1675, label %1677, label %1676

; <label>:1676		; preds = %1675, %1676
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1077		; <ubyte*>:2499 [#uses=2]
	load ubyte* %2499		; <ubyte>:3946 [#uses=1]
	add ubyte %3946, 255		; <ubyte>:3947 [#uses=1]
	store ubyte %3947, ubyte* %2499
	add uint %1060, 4294967214		; <uint>:1078 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1078		; <ubyte*>:2500 [#uses=2]
	load ubyte* %2500		; <ubyte>:3948 [#uses=1]
	add ubyte %3948, 1		; <ubyte>:3949 [#uses=1]
	store ubyte %3949, ubyte* %2500
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1076		; <ubyte*>:2501 [#uses=2]
	load ubyte* %2501		; <ubyte>:3950 [#uses=1]
	add ubyte %3950, 1		; <ubyte>:3951 [#uses=1]
	store ubyte %3951, ubyte* %2501
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1077		; <ubyte*>:2502 [#uses=1]
	load ubyte* %2502		; <ubyte>:3952 [#uses=1]
	seteq ubyte %3952, 0		; <bool>:1676 [#uses=1]
	br bool %1676, label %1677, label %1676

; <label>:1677		; preds = %1675, %1676
	add uint %1060, 4294967214		; <uint>:1079 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1079		; <ubyte*>:2503 [#uses=1]
	load ubyte* %2503		; <ubyte>:3953 [#uses=1]
	seteq ubyte %3953, 0		; <bool>:1677 [#uses=1]
	br bool %1677, label %1679, label %1678

; <label>:1678		; preds = %1677, %1678
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1077		; <ubyte*>:2504 [#uses=2]
	load ubyte* %2504		; <ubyte>:3954 [#uses=1]
	add ubyte %3954, 1		; <ubyte>:3955 [#uses=1]
	store ubyte %3955, ubyte* %2504
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1079		; <ubyte*>:2505 [#uses=2]
	load ubyte* %2505		; <ubyte>:3956 [#uses=2]
	add ubyte %3956, 255		; <ubyte>:3957 [#uses=1]
	store ubyte %3957, ubyte* %2505
	seteq ubyte %3956, 1		; <bool>:1678 [#uses=1]
	br bool %1678, label %1679, label %1678

; <label>:1679		; preds = %1677, %1678
	add uint %1060, 36		; <uint>:1080 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1080		; <ubyte*>:2506 [#uses=1]
	load ubyte* %2506		; <ubyte>:3958 [#uses=1]
	seteq ubyte %3958, 0		; <bool>:1679 [#uses=1]
	br bool %1679, label %1681, label %1680

; <label>:1680		; preds = %1679, %1680
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1080		; <ubyte*>:2507 [#uses=2]
	load ubyte* %2507		; <ubyte>:3959 [#uses=2]
	add ubyte %3959, 255		; <ubyte>:3960 [#uses=1]
	store ubyte %3960, ubyte* %2507
	seteq ubyte %3959, 1		; <bool>:1680 [#uses=1]
	br bool %1680, label %1681, label %1680

; <label>:1681		; preds = %1679, %1680
	add uint %1060, 4294967219		; <uint>:1081 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1081		; <ubyte*>:2508 [#uses=1]
	load ubyte* %2508		; <ubyte>:3961 [#uses=1]
	seteq ubyte %3961, 0		; <bool>:1681 [#uses=1]
	br bool %1681, label %1683, label %1682

; <label>:1682		; preds = %1681, %1682
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1081		; <ubyte*>:2509 [#uses=2]
	load ubyte* %2509		; <ubyte>:3962 [#uses=1]
	add ubyte %3962, 255		; <ubyte>:3963 [#uses=1]
	store ubyte %3963, ubyte* %2509
	add uint %1060, 4294967220		; <uint>:1082 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1082		; <ubyte*>:2510 [#uses=2]
	load ubyte* %2510		; <ubyte>:3964 [#uses=1]
	add ubyte %3964, 1		; <ubyte>:3965 [#uses=1]
	store ubyte %3965, ubyte* %2510
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1080		; <ubyte*>:2511 [#uses=2]
	load ubyte* %2511		; <ubyte>:3966 [#uses=1]
	add ubyte %3966, 1		; <ubyte>:3967 [#uses=1]
	store ubyte %3967, ubyte* %2511
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1081		; <ubyte*>:2512 [#uses=1]
	load ubyte* %2512		; <ubyte>:3968 [#uses=1]
	seteq ubyte %3968, 0		; <bool>:1682 [#uses=1]
	br bool %1682, label %1683, label %1682

; <label>:1683		; preds = %1681, %1682
	add uint %1060, 4294967220		; <uint>:1083 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1083		; <ubyte*>:2513 [#uses=1]
	load ubyte* %2513		; <ubyte>:3969 [#uses=1]
	seteq ubyte %3969, 0		; <bool>:1683 [#uses=1]
	br bool %1683, label %1685, label %1684

; <label>:1684		; preds = %1683, %1684
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1081		; <ubyte*>:2514 [#uses=2]
	load ubyte* %2514		; <ubyte>:3970 [#uses=1]
	add ubyte %3970, 1		; <ubyte>:3971 [#uses=1]
	store ubyte %3971, ubyte* %2514
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1083		; <ubyte*>:2515 [#uses=2]
	load ubyte* %2515		; <ubyte>:3972 [#uses=2]
	add ubyte %3972, 255		; <ubyte>:3973 [#uses=1]
	store ubyte %3973, ubyte* %2515
	seteq ubyte %3972, 1		; <bool>:1684 [#uses=1]
	br bool %1684, label %1685, label %1684

; <label>:1685		; preds = %1683, %1684
	add uint %1060, 42		; <uint>:1084 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1084		; <ubyte*>:2516 [#uses=1]
	load ubyte* %2516		; <ubyte>:3974 [#uses=1]
	seteq ubyte %3974, 0		; <bool>:1685 [#uses=1]
	br bool %1685, label %1687, label %1686

; <label>:1686		; preds = %1685, %1686
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1084		; <ubyte*>:2517 [#uses=2]
	load ubyte* %2517		; <ubyte>:3975 [#uses=2]
	add ubyte %3975, 255		; <ubyte>:3976 [#uses=1]
	store ubyte %3976, ubyte* %2517
	seteq ubyte %3975, 1		; <bool>:1686 [#uses=1]
	br bool %1686, label %1687, label %1686

; <label>:1687		; preds = %1685, %1686
	add uint %1060, 4294967225		; <uint>:1085 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1085		; <ubyte*>:2518 [#uses=1]
	load ubyte* %2518		; <ubyte>:3977 [#uses=1]
	seteq ubyte %3977, 0		; <bool>:1687 [#uses=1]
	br bool %1687, label %1689, label %1688

; <label>:1688		; preds = %1687, %1688
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1085		; <ubyte*>:2519 [#uses=2]
	load ubyte* %2519		; <ubyte>:3978 [#uses=1]
	add ubyte %3978, 255		; <ubyte>:3979 [#uses=1]
	store ubyte %3979, ubyte* %2519
	add uint %1060, 4294967226		; <uint>:1086 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1086		; <ubyte*>:2520 [#uses=2]
	load ubyte* %2520		; <ubyte>:3980 [#uses=1]
	add ubyte %3980, 1		; <ubyte>:3981 [#uses=1]
	store ubyte %3981, ubyte* %2520
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1084		; <ubyte*>:2521 [#uses=2]
	load ubyte* %2521		; <ubyte>:3982 [#uses=1]
	add ubyte %3982, 1		; <ubyte>:3983 [#uses=1]
	store ubyte %3983, ubyte* %2521
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1085		; <ubyte*>:2522 [#uses=1]
	load ubyte* %2522		; <ubyte>:3984 [#uses=1]
	seteq ubyte %3984, 0		; <bool>:1688 [#uses=1]
	br bool %1688, label %1689, label %1688

; <label>:1689		; preds = %1687, %1688
	add uint %1060, 4294967226		; <uint>:1087 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1087		; <ubyte*>:2523 [#uses=1]
	load ubyte* %2523		; <ubyte>:3985 [#uses=1]
	seteq ubyte %3985, 0		; <bool>:1689 [#uses=1]
	br bool %1689, label %1691, label %1690

; <label>:1690		; preds = %1689, %1690
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1085		; <ubyte*>:2524 [#uses=2]
	load ubyte* %2524		; <ubyte>:3986 [#uses=1]
	add ubyte %3986, 1		; <ubyte>:3987 [#uses=1]
	store ubyte %3987, ubyte* %2524
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1087		; <ubyte*>:2525 [#uses=2]
	load ubyte* %2525		; <ubyte>:3988 [#uses=2]
	add ubyte %3988, 255		; <ubyte>:3989 [#uses=1]
	store ubyte %3989, ubyte* %2525
	seteq ubyte %3988, 1		; <bool>:1690 [#uses=1]
	br bool %1690, label %1691, label %1690

; <label>:1691		; preds = %1689, %1690
	add uint %1060, 48		; <uint>:1088 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1088		; <ubyte*>:2526 [#uses=1]
	load ubyte* %2526		; <ubyte>:3990 [#uses=1]
	seteq ubyte %3990, 0		; <bool>:1691 [#uses=1]
	br bool %1691, label %1693, label %1692

; <label>:1692		; preds = %1691, %1692
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1088		; <ubyte*>:2527 [#uses=2]
	load ubyte* %2527		; <ubyte>:3991 [#uses=2]
	add ubyte %3991, 255		; <ubyte>:3992 [#uses=1]
	store ubyte %3992, ubyte* %2527
	seteq ubyte %3991, 1		; <bool>:1692 [#uses=1]
	br bool %1692, label %1693, label %1692

; <label>:1693		; preds = %1691, %1692
	add uint %1060, 4294967231		; <uint>:1089 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1089		; <ubyte*>:2528 [#uses=1]
	load ubyte* %2528		; <ubyte>:3993 [#uses=1]
	seteq ubyte %3993, 0		; <bool>:1693 [#uses=1]
	br bool %1693, label %1695, label %1694

; <label>:1694		; preds = %1693, %1694
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1089		; <ubyte*>:2529 [#uses=2]
	load ubyte* %2529		; <ubyte>:3994 [#uses=1]
	add ubyte %3994, 255		; <ubyte>:3995 [#uses=1]
	store ubyte %3995, ubyte* %2529
	add uint %1060, 4294967232		; <uint>:1090 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1090		; <ubyte*>:2530 [#uses=2]
	load ubyte* %2530		; <ubyte>:3996 [#uses=1]
	add ubyte %3996, 1		; <ubyte>:3997 [#uses=1]
	store ubyte %3997, ubyte* %2530
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1088		; <ubyte*>:2531 [#uses=2]
	load ubyte* %2531		; <ubyte>:3998 [#uses=1]
	add ubyte %3998, 1		; <ubyte>:3999 [#uses=1]
	store ubyte %3999, ubyte* %2531
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1089		; <ubyte*>:2532 [#uses=1]
	load ubyte* %2532		; <ubyte>:4000 [#uses=1]
	seteq ubyte %4000, 0		; <bool>:1694 [#uses=1]
	br bool %1694, label %1695, label %1694

; <label>:1695		; preds = %1693, %1694
	add uint %1060, 4294967232		; <uint>:1091 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1091		; <ubyte*>:2533 [#uses=1]
	load ubyte* %2533		; <ubyte>:4001 [#uses=1]
	seteq ubyte %4001, 0		; <bool>:1695 [#uses=1]
	br bool %1695, label %1697, label %1696

; <label>:1696		; preds = %1695, %1696
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1089		; <ubyte*>:2534 [#uses=2]
	load ubyte* %2534		; <ubyte>:4002 [#uses=1]
	add ubyte %4002, 1		; <ubyte>:4003 [#uses=1]
	store ubyte %4003, ubyte* %2534
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1091		; <ubyte*>:2535 [#uses=2]
	load ubyte* %2535		; <ubyte>:4004 [#uses=2]
	add ubyte %4004, 255		; <ubyte>:4005 [#uses=1]
	store ubyte %4005, ubyte* %2535
	seteq ubyte %4004, 1		; <bool>:1696 [#uses=1]
	br bool %1696, label %1697, label %1696

; <label>:1697		; preds = %1695, %1696
	add uint %1060, 54		; <uint>:1092 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1092		; <ubyte*>:2536 [#uses=1]
	load ubyte* %2536		; <ubyte>:4006 [#uses=1]
	seteq ubyte %4006, 0		; <bool>:1697 [#uses=1]
	br bool %1697, label %1699, label %1698

; <label>:1698		; preds = %1697, %1698
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1092		; <ubyte*>:2537 [#uses=2]
	load ubyte* %2537		; <ubyte>:4007 [#uses=2]
	add ubyte %4007, 255		; <ubyte>:4008 [#uses=1]
	store ubyte %4008, ubyte* %2537
	seteq ubyte %4007, 1		; <bool>:1698 [#uses=1]
	br bool %1698, label %1699, label %1698

; <label>:1699		; preds = %1697, %1698
	add uint %1060, 4294967237		; <uint>:1093 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1093		; <ubyte*>:2538 [#uses=1]
	load ubyte* %2538		; <ubyte>:4009 [#uses=1]
	seteq ubyte %4009, 0		; <bool>:1699 [#uses=1]
	br bool %1699, label %1701, label %1700

; <label>:1700		; preds = %1699, %1700
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1093		; <ubyte*>:2539 [#uses=2]
	load ubyte* %2539		; <ubyte>:4010 [#uses=1]
	add ubyte %4010, 255		; <ubyte>:4011 [#uses=1]
	store ubyte %4011, ubyte* %2539
	add uint %1060, 4294967238		; <uint>:1094 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1094		; <ubyte*>:2540 [#uses=2]
	load ubyte* %2540		; <ubyte>:4012 [#uses=1]
	add ubyte %4012, 1		; <ubyte>:4013 [#uses=1]
	store ubyte %4013, ubyte* %2540
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1092		; <ubyte*>:2541 [#uses=2]
	load ubyte* %2541		; <ubyte>:4014 [#uses=1]
	add ubyte %4014, 1		; <ubyte>:4015 [#uses=1]
	store ubyte %4015, ubyte* %2541
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1093		; <ubyte*>:2542 [#uses=1]
	load ubyte* %2542		; <ubyte>:4016 [#uses=1]
	seteq ubyte %4016, 0		; <bool>:1700 [#uses=1]
	br bool %1700, label %1701, label %1700

; <label>:1701		; preds = %1699, %1700
	add uint %1060, 4294967238		; <uint>:1095 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1095		; <ubyte*>:2543 [#uses=1]
	load ubyte* %2543		; <ubyte>:4017 [#uses=1]
	seteq ubyte %4017, 0		; <bool>:1701 [#uses=1]
	br bool %1701, label %1703, label %1702

; <label>:1702		; preds = %1701, %1702
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1093		; <ubyte*>:2544 [#uses=2]
	load ubyte* %2544		; <ubyte>:4018 [#uses=1]
	add ubyte %4018, 1		; <ubyte>:4019 [#uses=1]
	store ubyte %4019, ubyte* %2544
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1095		; <ubyte*>:2545 [#uses=2]
	load ubyte* %2545		; <ubyte>:4020 [#uses=2]
	add ubyte %4020, 255		; <ubyte>:4021 [#uses=1]
	store ubyte %4021, ubyte* %2545
	seteq ubyte %4020, 1		; <bool>:1702 [#uses=1]
	br bool %1702, label %1703, label %1702

; <label>:1703		; preds = %1701, %1702
	add uint %1060, 60		; <uint>:1096 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1096		; <ubyte*>:2546 [#uses=1]
	load ubyte* %2546		; <ubyte>:4022 [#uses=1]
	seteq ubyte %4022, 0		; <bool>:1703 [#uses=1]
	br bool %1703, label %1705, label %1704

; <label>:1704		; preds = %1703, %1704
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1096		; <ubyte*>:2547 [#uses=2]
	load ubyte* %2547		; <ubyte>:4023 [#uses=2]
	add ubyte %4023, 255		; <ubyte>:4024 [#uses=1]
	store ubyte %4024, ubyte* %2547
	seteq ubyte %4023, 1		; <bool>:1704 [#uses=1]
	br bool %1704, label %1705, label %1704

; <label>:1705		; preds = %1703, %1704
	add uint %1060, 4294967243		; <uint>:1097 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1097		; <ubyte*>:2548 [#uses=1]
	load ubyte* %2548		; <ubyte>:4025 [#uses=1]
	seteq ubyte %4025, 0		; <bool>:1705 [#uses=1]
	br bool %1705, label %1707, label %1706

; <label>:1706		; preds = %1705, %1706
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1097		; <ubyte*>:2549 [#uses=2]
	load ubyte* %2549		; <ubyte>:4026 [#uses=1]
	add ubyte %4026, 255		; <ubyte>:4027 [#uses=1]
	store ubyte %4027, ubyte* %2549
	add uint %1060, 4294967244		; <uint>:1098 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1098		; <ubyte*>:2550 [#uses=2]
	load ubyte* %2550		; <ubyte>:4028 [#uses=1]
	add ubyte %4028, 1		; <ubyte>:4029 [#uses=1]
	store ubyte %4029, ubyte* %2550
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1096		; <ubyte*>:2551 [#uses=2]
	load ubyte* %2551		; <ubyte>:4030 [#uses=1]
	add ubyte %4030, 1		; <ubyte>:4031 [#uses=1]
	store ubyte %4031, ubyte* %2551
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1097		; <ubyte*>:2552 [#uses=1]
	load ubyte* %2552		; <ubyte>:4032 [#uses=1]
	seteq ubyte %4032, 0		; <bool>:1706 [#uses=1]
	br bool %1706, label %1707, label %1706

; <label>:1707		; preds = %1705, %1706
	add uint %1060, 4294967244		; <uint>:1099 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1099		; <ubyte*>:2553 [#uses=1]
	load ubyte* %2553		; <ubyte>:4033 [#uses=1]
	seteq ubyte %4033, 0		; <bool>:1707 [#uses=1]
	br bool %1707, label %1709, label %1708

; <label>:1708		; preds = %1707, %1708
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1097		; <ubyte*>:2554 [#uses=2]
	load ubyte* %2554		; <ubyte>:4034 [#uses=1]
	add ubyte %4034, 1		; <ubyte>:4035 [#uses=1]
	store ubyte %4035, ubyte* %2554
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1099		; <ubyte*>:2555 [#uses=2]
	load ubyte* %2555		; <ubyte>:4036 [#uses=2]
	add ubyte %4036, 255		; <ubyte>:4037 [#uses=1]
	store ubyte %4037, ubyte* %2555
	seteq ubyte %4036, 1		; <bool>:1708 [#uses=1]
	br bool %1708, label %1709, label %1708

; <label>:1709		; preds = %1707, %1708
	add uint %1060, 66		; <uint>:1100 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1100		; <ubyte*>:2556 [#uses=1]
	load ubyte* %2556		; <ubyte>:4038 [#uses=1]
	seteq ubyte %4038, 0		; <bool>:1709 [#uses=1]
	br bool %1709, label %1711, label %1710

; <label>:1710		; preds = %1709, %1710
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1100		; <ubyte*>:2557 [#uses=2]
	load ubyte* %2557		; <ubyte>:4039 [#uses=2]
	add ubyte %4039, 255		; <ubyte>:4040 [#uses=1]
	store ubyte %4040, ubyte* %2557
	seteq ubyte %4039, 1		; <bool>:1710 [#uses=1]
	br bool %1710, label %1711, label %1710

; <label>:1711		; preds = %1709, %1710
	add uint %1060, 4294967249		; <uint>:1101 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1101		; <ubyte*>:2558 [#uses=1]
	load ubyte* %2558		; <ubyte>:4041 [#uses=1]
	seteq ubyte %4041, 0		; <bool>:1711 [#uses=1]
	br bool %1711, label %1713, label %1712

; <label>:1712		; preds = %1711, %1712
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1101		; <ubyte*>:2559 [#uses=2]
	load ubyte* %2559		; <ubyte>:4042 [#uses=1]
	add ubyte %4042, 255		; <ubyte>:4043 [#uses=1]
	store ubyte %4043, ubyte* %2559
	add uint %1060, 4294967250		; <uint>:1102 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1102		; <ubyte*>:2560 [#uses=2]
	load ubyte* %2560		; <ubyte>:4044 [#uses=1]
	add ubyte %4044, 1		; <ubyte>:4045 [#uses=1]
	store ubyte %4045, ubyte* %2560
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1100		; <ubyte*>:2561 [#uses=2]
	load ubyte* %2561		; <ubyte>:4046 [#uses=1]
	add ubyte %4046, 1		; <ubyte>:4047 [#uses=1]
	store ubyte %4047, ubyte* %2561
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1101		; <ubyte*>:2562 [#uses=1]
	load ubyte* %2562		; <ubyte>:4048 [#uses=1]
	seteq ubyte %4048, 0		; <bool>:1712 [#uses=1]
	br bool %1712, label %1713, label %1712

; <label>:1713		; preds = %1711, %1712
	add uint %1060, 4294967250		; <uint>:1103 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1103		; <ubyte*>:2563 [#uses=1]
	load ubyte* %2563		; <ubyte>:4049 [#uses=1]
	seteq ubyte %4049, 0		; <bool>:1713 [#uses=1]
	br bool %1713, label %1715, label %1714

; <label>:1714		; preds = %1713, %1714
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1101		; <ubyte*>:2564 [#uses=2]
	load ubyte* %2564		; <ubyte>:4050 [#uses=1]
	add ubyte %4050, 1		; <ubyte>:4051 [#uses=1]
	store ubyte %4051, ubyte* %2564
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1103		; <ubyte*>:2565 [#uses=2]
	load ubyte* %2565		; <ubyte>:4052 [#uses=2]
	add ubyte %4052, 255		; <ubyte>:4053 [#uses=1]
	store ubyte %4053, ubyte* %2565
	seteq ubyte %4052, 1		; <bool>:1714 [#uses=1]
	br bool %1714, label %1715, label %1714

; <label>:1715		; preds = %1713, %1714
	add uint %1060, 72		; <uint>:1104 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1104		; <ubyte*>:2566 [#uses=1]
	load ubyte* %2566		; <ubyte>:4054 [#uses=1]
	seteq ubyte %4054, 0		; <bool>:1715 [#uses=1]
	br bool %1715, label %1717, label %1716

; <label>:1716		; preds = %1715, %1716
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1104		; <ubyte*>:2567 [#uses=2]
	load ubyte* %2567		; <ubyte>:4055 [#uses=2]
	add ubyte %4055, 255		; <ubyte>:4056 [#uses=1]
	store ubyte %4056, ubyte* %2567
	seteq ubyte %4055, 1		; <bool>:1716 [#uses=1]
	br bool %1716, label %1717, label %1716

; <label>:1717		; preds = %1715, %1716
	add uint %1060, 4294967255		; <uint>:1105 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1105		; <ubyte*>:2568 [#uses=1]
	load ubyte* %2568		; <ubyte>:4057 [#uses=1]
	seteq ubyte %4057, 0		; <bool>:1717 [#uses=1]
	br bool %1717, label %1719, label %1718

; <label>:1718		; preds = %1717, %1718
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1105		; <ubyte*>:2569 [#uses=2]
	load ubyte* %2569		; <ubyte>:4058 [#uses=1]
	add ubyte %4058, 255		; <ubyte>:4059 [#uses=1]
	store ubyte %4059, ubyte* %2569
	add uint %1060, 4294967256		; <uint>:1106 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1106		; <ubyte*>:2570 [#uses=2]
	load ubyte* %2570		; <ubyte>:4060 [#uses=1]
	add ubyte %4060, 1		; <ubyte>:4061 [#uses=1]
	store ubyte %4061, ubyte* %2570
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1104		; <ubyte*>:2571 [#uses=2]
	load ubyte* %2571		; <ubyte>:4062 [#uses=1]
	add ubyte %4062, 1		; <ubyte>:4063 [#uses=1]
	store ubyte %4063, ubyte* %2571
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1105		; <ubyte*>:2572 [#uses=1]
	load ubyte* %2572		; <ubyte>:4064 [#uses=1]
	seteq ubyte %4064, 0		; <bool>:1718 [#uses=1]
	br bool %1718, label %1719, label %1718

; <label>:1719		; preds = %1717, %1718
	add uint %1060, 4294967256		; <uint>:1107 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1107		; <ubyte*>:2573 [#uses=1]
	load ubyte* %2573		; <ubyte>:4065 [#uses=1]
	seteq ubyte %4065, 0		; <bool>:1719 [#uses=1]
	br bool %1719, label %1721, label %1720

; <label>:1720		; preds = %1719, %1720
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1105		; <ubyte*>:2574 [#uses=2]
	load ubyte* %2574		; <ubyte>:4066 [#uses=1]
	add ubyte %4066, 1		; <ubyte>:4067 [#uses=1]
	store ubyte %4067, ubyte* %2574
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1107		; <ubyte*>:2575 [#uses=2]
	load ubyte* %2575		; <ubyte>:4068 [#uses=2]
	add ubyte %4068, 255		; <ubyte>:4069 [#uses=1]
	store ubyte %4069, ubyte* %2575
	seteq ubyte %4068, 1		; <bool>:1720 [#uses=1]
	br bool %1720, label %1721, label %1720

; <label>:1721		; preds = %1719, %1720
	add uint %1060, 78		; <uint>:1108 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1108		; <ubyte*>:2576 [#uses=1]
	load ubyte* %2576		; <ubyte>:4070 [#uses=1]
	seteq ubyte %4070, 0		; <bool>:1721 [#uses=1]
	br bool %1721, label %1723, label %1722

; <label>:1722		; preds = %1721, %1722
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1108		; <ubyte*>:2577 [#uses=2]
	load ubyte* %2577		; <ubyte>:4071 [#uses=2]
	add ubyte %4071, 255		; <ubyte>:4072 [#uses=1]
	store ubyte %4072, ubyte* %2577
	seteq ubyte %4071, 1		; <bool>:1722 [#uses=1]
	br bool %1722, label %1723, label %1722

; <label>:1723		; preds = %1721, %1722
	add uint %1060, 4294967261		; <uint>:1109 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1109		; <ubyte*>:2578 [#uses=1]
	load ubyte* %2578		; <ubyte>:4073 [#uses=1]
	seteq ubyte %4073, 0		; <bool>:1723 [#uses=1]
	br bool %1723, label %1725, label %1724

; <label>:1724		; preds = %1723, %1724
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1109		; <ubyte*>:2579 [#uses=2]
	load ubyte* %2579		; <ubyte>:4074 [#uses=1]
	add ubyte %4074, 255		; <ubyte>:4075 [#uses=1]
	store ubyte %4075, ubyte* %2579
	add uint %1060, 4294967262		; <uint>:1110 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1110		; <ubyte*>:2580 [#uses=2]
	load ubyte* %2580		; <ubyte>:4076 [#uses=1]
	add ubyte %4076, 1		; <ubyte>:4077 [#uses=1]
	store ubyte %4077, ubyte* %2580
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1108		; <ubyte*>:2581 [#uses=2]
	load ubyte* %2581		; <ubyte>:4078 [#uses=1]
	add ubyte %4078, 1		; <ubyte>:4079 [#uses=1]
	store ubyte %4079, ubyte* %2581
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1109		; <ubyte*>:2582 [#uses=1]
	load ubyte* %2582		; <ubyte>:4080 [#uses=1]
	seteq ubyte %4080, 0		; <bool>:1724 [#uses=1]
	br bool %1724, label %1725, label %1724

; <label>:1725		; preds = %1723, %1724
	add uint %1060, 4294967262		; <uint>:1111 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1111		; <ubyte*>:2583 [#uses=1]
	load ubyte* %2583		; <ubyte>:4081 [#uses=1]
	seteq ubyte %4081, 0		; <bool>:1725 [#uses=1]
	br bool %1725, label %1727, label %1726

; <label>:1726		; preds = %1725, %1726
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1109		; <ubyte*>:2584 [#uses=2]
	load ubyte* %2584		; <ubyte>:4082 [#uses=1]
	add ubyte %4082, 1		; <ubyte>:4083 [#uses=1]
	store ubyte %4083, ubyte* %2584
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1111		; <ubyte*>:2585 [#uses=2]
	load ubyte* %2585		; <ubyte>:4084 [#uses=2]
	add ubyte %4084, 255		; <ubyte>:4085 [#uses=1]
	store ubyte %4085, ubyte* %2585
	seteq ubyte %4084, 1		; <bool>:1726 [#uses=1]
	br bool %1726, label %1727, label %1726

; <label>:1727		; preds = %1725, %1726
	add uint %1060, 84		; <uint>:1112 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1112		; <ubyte*>:2586 [#uses=1]
	load ubyte* %2586		; <ubyte>:4086 [#uses=1]
	seteq ubyte %4086, 0		; <bool>:1727 [#uses=1]
	br bool %1727, label %1729, label %1728

; <label>:1728		; preds = %1727, %1728
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1112		; <ubyte*>:2587 [#uses=2]
	load ubyte* %2587		; <ubyte>:4087 [#uses=2]
	add ubyte %4087, 255		; <ubyte>:4088 [#uses=1]
	store ubyte %4088, ubyte* %2587
	seteq ubyte %4087, 1		; <bool>:1728 [#uses=1]
	br bool %1728, label %1729, label %1728

; <label>:1729		; preds = %1727, %1728
	add uint %1060, 4294967267		; <uint>:1113 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1113		; <ubyte*>:2588 [#uses=1]
	load ubyte* %2588		; <ubyte>:4089 [#uses=1]
	seteq ubyte %4089, 0		; <bool>:1729 [#uses=1]
	br bool %1729, label %1731, label %1730

; <label>:1730		; preds = %1729, %1730
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1113		; <ubyte*>:2589 [#uses=2]
	load ubyte* %2589		; <ubyte>:4090 [#uses=1]
	add ubyte %4090, 255		; <ubyte>:4091 [#uses=1]
	store ubyte %4091, ubyte* %2589
	add uint %1060, 4294967268		; <uint>:1114 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1114		; <ubyte*>:2590 [#uses=2]
	load ubyte* %2590		; <ubyte>:4092 [#uses=1]
	add ubyte %4092, 1		; <ubyte>:4093 [#uses=1]
	store ubyte %4093, ubyte* %2590
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1112		; <ubyte*>:2591 [#uses=2]
	load ubyte* %2591		; <ubyte>:4094 [#uses=1]
	add ubyte %4094, 1		; <ubyte>:4095 [#uses=1]
	store ubyte %4095, ubyte* %2591
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1113		; <ubyte*>:2592 [#uses=1]
	load ubyte* %2592		; <ubyte>:4096 [#uses=1]
	seteq ubyte %4096, 0		; <bool>:1730 [#uses=1]
	br bool %1730, label %1731, label %1730

; <label>:1731		; preds = %1729, %1730
	add uint %1060, 4294967268		; <uint>:1115 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1115		; <ubyte*>:2593 [#uses=1]
	load ubyte* %2593		; <ubyte>:4097 [#uses=1]
	seteq ubyte %4097, 0		; <bool>:1731 [#uses=1]
	br bool %1731, label %1733, label %1732

; <label>:1732		; preds = %1731, %1732
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1113		; <ubyte*>:2594 [#uses=2]
	load ubyte* %2594		; <ubyte>:4098 [#uses=1]
	add ubyte %4098, 1		; <ubyte>:4099 [#uses=1]
	store ubyte %4099, ubyte* %2594
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1115		; <ubyte*>:2595 [#uses=2]
	load ubyte* %2595		; <ubyte>:4100 [#uses=2]
	add ubyte %4100, 255		; <ubyte>:4101 [#uses=1]
	store ubyte %4101, ubyte* %2595
	seteq ubyte %4100, 1		; <bool>:1732 [#uses=1]
	br bool %1732, label %1733, label %1732

; <label>:1733		; preds = %1731, %1732
	add uint %1060, 90		; <uint>:1116 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1116		; <ubyte*>:2596 [#uses=1]
	load ubyte* %2596		; <ubyte>:4102 [#uses=1]
	seteq ubyte %4102, 0		; <bool>:1733 [#uses=1]
	br bool %1733, label %1735, label %1734

; <label>:1734		; preds = %1733, %1734
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1116		; <ubyte*>:2597 [#uses=2]
	load ubyte* %2597		; <ubyte>:4103 [#uses=2]
	add ubyte %4103, 255		; <ubyte>:4104 [#uses=1]
	store ubyte %4104, ubyte* %2597
	seteq ubyte %4103, 1		; <bool>:1734 [#uses=1]
	br bool %1734, label %1735, label %1734

; <label>:1735		; preds = %1733, %1734
	add uint %1060, 4294967273		; <uint>:1117 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1117		; <ubyte*>:2598 [#uses=1]
	load ubyte* %2598		; <ubyte>:4105 [#uses=1]
	seteq ubyte %4105, 0		; <bool>:1735 [#uses=1]
	br bool %1735, label %1737, label %1736

; <label>:1736		; preds = %1735, %1736
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1117		; <ubyte*>:2599 [#uses=2]
	load ubyte* %2599		; <ubyte>:4106 [#uses=1]
	add ubyte %4106, 255		; <ubyte>:4107 [#uses=1]
	store ubyte %4107, ubyte* %2599
	add uint %1060, 4294967274		; <uint>:1118 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1118		; <ubyte*>:2600 [#uses=2]
	load ubyte* %2600		; <ubyte>:4108 [#uses=1]
	add ubyte %4108, 1		; <ubyte>:4109 [#uses=1]
	store ubyte %4109, ubyte* %2600
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1116		; <ubyte*>:2601 [#uses=2]
	load ubyte* %2601		; <ubyte>:4110 [#uses=1]
	add ubyte %4110, 1		; <ubyte>:4111 [#uses=1]
	store ubyte %4111, ubyte* %2601
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1117		; <ubyte*>:2602 [#uses=1]
	load ubyte* %2602		; <ubyte>:4112 [#uses=1]
	seteq ubyte %4112, 0		; <bool>:1736 [#uses=1]
	br bool %1736, label %1737, label %1736

; <label>:1737		; preds = %1735, %1736
	add uint %1060, 4294967274		; <uint>:1119 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1119		; <ubyte*>:2603 [#uses=1]
	load ubyte* %2603		; <ubyte>:4113 [#uses=1]
	seteq ubyte %4113, 0		; <bool>:1737 [#uses=1]
	br bool %1737, label %1739, label %1738

; <label>:1738		; preds = %1737, %1738
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1117		; <ubyte*>:2604 [#uses=2]
	load ubyte* %2604		; <ubyte>:4114 [#uses=1]
	add ubyte %4114, 1		; <ubyte>:4115 [#uses=1]
	store ubyte %4115, ubyte* %2604
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1119		; <ubyte*>:2605 [#uses=2]
	load ubyte* %2605		; <ubyte>:4116 [#uses=2]
	add ubyte %4116, 255		; <ubyte>:4117 [#uses=1]
	store ubyte %4117, ubyte* %2605
	seteq ubyte %4116, 1		; <bool>:1738 [#uses=1]
	br bool %1738, label %1739, label %1738

; <label>:1739		; preds = %1737, %1738
	add uint %1060, 96		; <uint>:1120 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1120		; <ubyte*>:2606 [#uses=1]
	load ubyte* %2606		; <ubyte>:4118 [#uses=1]
	seteq ubyte %4118, 0		; <bool>:1739 [#uses=1]
	br bool %1739, label %1741, label %1740

; <label>:1740		; preds = %1739, %1740
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1120		; <ubyte*>:2607 [#uses=2]
	load ubyte* %2607		; <ubyte>:4119 [#uses=2]
	add ubyte %4119, 255		; <ubyte>:4120 [#uses=1]
	store ubyte %4120, ubyte* %2607
	seteq ubyte %4119, 1		; <bool>:1740 [#uses=1]
	br bool %1740, label %1741, label %1740

; <label>:1741		; preds = %1739, %1740
	add uint %1060, 4294967279		; <uint>:1121 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1121		; <ubyte*>:2608 [#uses=1]
	load ubyte* %2608		; <ubyte>:4121 [#uses=1]
	seteq ubyte %4121, 0		; <bool>:1741 [#uses=1]
	br bool %1741, label %1743, label %1742

; <label>:1742		; preds = %1741, %1742
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1121		; <ubyte*>:2609 [#uses=2]
	load ubyte* %2609		; <ubyte>:4122 [#uses=1]
	add ubyte %4122, 255		; <ubyte>:4123 [#uses=1]
	store ubyte %4123, ubyte* %2609
	add uint %1060, 4294967280		; <uint>:1122 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1122		; <ubyte*>:2610 [#uses=2]
	load ubyte* %2610		; <ubyte>:4124 [#uses=1]
	add ubyte %4124, 1		; <ubyte>:4125 [#uses=1]
	store ubyte %4125, ubyte* %2610
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1120		; <ubyte*>:2611 [#uses=2]
	load ubyte* %2611		; <ubyte>:4126 [#uses=1]
	add ubyte %4126, 1		; <ubyte>:4127 [#uses=1]
	store ubyte %4127, ubyte* %2611
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1121		; <ubyte*>:2612 [#uses=1]
	load ubyte* %2612		; <ubyte>:4128 [#uses=1]
	seteq ubyte %4128, 0		; <bool>:1742 [#uses=1]
	br bool %1742, label %1743, label %1742

; <label>:1743		; preds = %1741, %1742
	add uint %1060, 4294967280		; <uint>:1123 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1123		; <ubyte*>:2613 [#uses=1]
	load ubyte* %2613		; <ubyte>:4129 [#uses=1]
	seteq ubyte %4129, 0		; <bool>:1743 [#uses=1]
	br bool %1743, label %1745, label %1744

; <label>:1744		; preds = %1743, %1744
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1121		; <ubyte*>:2614 [#uses=2]
	load ubyte* %2614		; <ubyte>:4130 [#uses=1]
	add ubyte %4130, 1		; <ubyte>:4131 [#uses=1]
	store ubyte %4131, ubyte* %2614
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1123		; <ubyte*>:2615 [#uses=2]
	load ubyte* %2615		; <ubyte>:4132 [#uses=2]
	add ubyte %4132, 255		; <ubyte>:4133 [#uses=1]
	store ubyte %4133, ubyte* %2615
	seteq ubyte %4132, 1		; <bool>:1744 [#uses=1]
	br bool %1744, label %1745, label %1744

; <label>:1745		; preds = %1743, %1744
	add uint %1060, 102		; <uint>:1124 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1124		; <ubyte*>:2616 [#uses=1]
	load ubyte* %2616		; <ubyte>:4134 [#uses=1]
	seteq ubyte %4134, 0		; <bool>:1745 [#uses=1]
	br bool %1745, label %1747, label %1746

; <label>:1746		; preds = %1745, %1746
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1124		; <ubyte*>:2617 [#uses=2]
	load ubyte* %2617		; <ubyte>:4135 [#uses=2]
	add ubyte %4135, 255		; <ubyte>:4136 [#uses=1]
	store ubyte %4136, ubyte* %2617
	seteq ubyte %4135, 1		; <bool>:1746 [#uses=1]
	br bool %1746, label %1747, label %1746

; <label>:1747		; preds = %1745, %1746
	add uint %1060, 4294967285		; <uint>:1125 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1125		; <ubyte*>:2618 [#uses=1]
	load ubyte* %2618		; <ubyte>:4137 [#uses=1]
	seteq ubyte %4137, 0		; <bool>:1747 [#uses=1]
	br bool %1747, label %1749, label %1748

; <label>:1748		; preds = %1747, %1748
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1125		; <ubyte*>:2619 [#uses=2]
	load ubyte* %2619		; <ubyte>:4138 [#uses=1]
	add ubyte %4138, 255		; <ubyte>:4139 [#uses=1]
	store ubyte %4139, ubyte* %2619
	add uint %1060, 4294967286		; <uint>:1126 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1126		; <ubyte*>:2620 [#uses=2]
	load ubyte* %2620		; <ubyte>:4140 [#uses=1]
	add ubyte %4140, 1		; <ubyte>:4141 [#uses=1]
	store ubyte %4141, ubyte* %2620
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1124		; <ubyte*>:2621 [#uses=2]
	load ubyte* %2621		; <ubyte>:4142 [#uses=1]
	add ubyte %4142, 1		; <ubyte>:4143 [#uses=1]
	store ubyte %4143, ubyte* %2621
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1125		; <ubyte*>:2622 [#uses=1]
	load ubyte* %2622		; <ubyte>:4144 [#uses=1]
	seteq ubyte %4144, 0		; <bool>:1748 [#uses=1]
	br bool %1748, label %1749, label %1748

; <label>:1749		; preds = %1747, %1748
	add uint %1060, 4294967286		; <uint>:1127 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1127		; <ubyte*>:2623 [#uses=1]
	load ubyte* %2623		; <ubyte>:4145 [#uses=1]
	seteq ubyte %4145, 0		; <bool>:1749 [#uses=1]
	br bool %1749, label %1751, label %1750

; <label>:1750		; preds = %1749, %1750
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1125		; <ubyte*>:2624 [#uses=2]
	load ubyte* %2624		; <ubyte>:4146 [#uses=1]
	add ubyte %4146, 1		; <ubyte>:4147 [#uses=1]
	store ubyte %4147, ubyte* %2624
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1127		; <ubyte*>:2625 [#uses=2]
	load ubyte* %2625		; <ubyte>:4148 [#uses=2]
	add ubyte %4148, 255		; <ubyte>:4149 [#uses=1]
	store ubyte %4149, ubyte* %2625
	seteq ubyte %4148, 1		; <bool>:1750 [#uses=1]
	br bool %1750, label %1751, label %1750

; <label>:1751		; preds = %1749, %1750
	add uint %1060, 106		; <uint>:1128 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1128		; <ubyte*>:2626 [#uses=1]
	load ubyte* %2626		; <ubyte>:4150 [#uses=1]
	seteq ubyte %4150, 0		; <bool>:1751 [#uses=1]
	br bool %1751, label %1753, label %1752

; <label>:1752		; preds = %1751, %1752
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1128		; <ubyte*>:2627 [#uses=2]
	load ubyte* %2627		; <ubyte>:4151 [#uses=2]
	add ubyte %4151, 255		; <ubyte>:4152 [#uses=1]
	store ubyte %4152, ubyte* %2627
	seteq ubyte %4151, 1		; <bool>:1752 [#uses=1]
	br bool %1752, label %1753, label %1752

; <label>:1753		; preds = %1751, %1752
	add uint %1060, 4294967191		; <uint>:1129 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1129		; <ubyte*>:2628 [#uses=1]
	load ubyte* %2628		; <ubyte>:4153 [#uses=1]
	seteq ubyte %4153, 0		; <bool>:1753 [#uses=1]
	br bool %1753, label %1755, label %1754

; <label>:1754		; preds = %1753, %1754
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1129		; <ubyte*>:2629 [#uses=2]
	load ubyte* %2629		; <ubyte>:4154 [#uses=1]
	add ubyte %4154, 255		; <ubyte>:4155 [#uses=1]
	store ubyte %4155, ubyte* %2629
	add uint %1060, 4294967192		; <uint>:1130 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1130		; <ubyte*>:2630 [#uses=2]
	load ubyte* %2630		; <ubyte>:4156 [#uses=1]
	add ubyte %4156, 1		; <ubyte>:4157 [#uses=1]
	store ubyte %4157, ubyte* %2630
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1128		; <ubyte*>:2631 [#uses=2]
	load ubyte* %2631		; <ubyte>:4158 [#uses=1]
	add ubyte %4158, 1		; <ubyte>:4159 [#uses=1]
	store ubyte %4159, ubyte* %2631
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1129		; <ubyte*>:2632 [#uses=1]
	load ubyte* %2632		; <ubyte>:4160 [#uses=1]
	seteq ubyte %4160, 0		; <bool>:1754 [#uses=1]
	br bool %1754, label %1755, label %1754

; <label>:1755		; preds = %1753, %1754
	add uint %1060, 4294967192		; <uint>:1131 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1131		; <ubyte*>:2633 [#uses=1]
	load ubyte* %2633		; <ubyte>:4161 [#uses=1]
	seteq ubyte %4161, 0		; <bool>:1755 [#uses=1]
	br bool %1755, label %1757, label %1756

; <label>:1756		; preds = %1755, %1756
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1129		; <ubyte*>:2634 [#uses=2]
	load ubyte* %2634		; <ubyte>:4162 [#uses=1]
	add ubyte %4162, 1		; <ubyte>:4163 [#uses=1]
	store ubyte %4163, ubyte* %2634
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1131		; <ubyte*>:2635 [#uses=2]
	load ubyte* %2635		; <ubyte>:4164 [#uses=2]
	add ubyte %4164, 255		; <ubyte>:4165 [#uses=1]
	store ubyte %4165, ubyte* %2635
	seteq ubyte %4164, 1		; <bool>:1756 [#uses=1]
	br bool %1756, label %1757, label %1756

; <label>:1757		; preds = %1755, %1756
	add uint %1060, 20		; <uint>:1132 [#uses=6]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1132		; <ubyte*>:2636 [#uses=1]
	load ubyte* %2636		; <ubyte>:4166 [#uses=1]
	seteq ubyte %4166, 0		; <bool>:1757 [#uses=1]
	br bool %1757, label %1759, label %1758

; <label>:1758		; preds = %1757, %1758
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1132		; <ubyte*>:2637 [#uses=2]
	load ubyte* %2637		; <ubyte>:4167 [#uses=2]
	add ubyte %4167, 255		; <ubyte>:4168 [#uses=1]
	store ubyte %4168, ubyte* %2637
	seteq ubyte %4167, 1		; <bool>:1758 [#uses=1]
	br bool %1758, label %1759, label %1758

; <label>:1759		; preds = %1757, %1758
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1128		; <ubyte*>:2638 [#uses=1]
	load ubyte* %2638		; <ubyte>:4169 [#uses=1]
	seteq ubyte %4169, 0		; <bool>:1759 [#uses=1]
	br bool %1759, label %1761, label %1760

; <label>:1760		; preds = %1759, %1760
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1132		; <ubyte*>:2639 [#uses=2]
	load ubyte* %2639		; <ubyte>:4170 [#uses=1]
	add ubyte %4170, 1		; <ubyte>:4171 [#uses=1]
	store ubyte %4171, ubyte* %2639
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1128		; <ubyte*>:2640 [#uses=2]
	load ubyte* %2640		; <ubyte>:4172 [#uses=2]
	add ubyte %4172, 255		; <ubyte>:4173 [#uses=1]
	store ubyte %4173, ubyte* %2640
	seteq ubyte %4172, 1		; <bool>:1760 [#uses=1]
	br bool %1760, label %1761, label %1760

; <label>:1761		; preds = %1759, %1760
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1132		; <ubyte*>:2641 [#uses=1]
	load ubyte* %2641		; <ubyte>:4174 [#uses=1]
	seteq ubyte %4174, 0		; <bool>:1761 [#uses=1]
	br bool %1761, label %1763, label %1762

; <label>:1762		; preds = %1761, %1765
	phi uint [ %1132, %1761 ], [ %1137, %1765 ]		; <uint>:1133 [#uses=6]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1133		; <ubyte*>:2642 [#uses=1]
	load ubyte* %2642		; <ubyte>:4175 [#uses=1]
	seteq ubyte %4175, 0		; <bool>:1762 [#uses=1]
	br bool %1762, label %1765, label %1764

; <label>:1763		; preds = %1761, %1765
	phi uint [ %1132, %1761 ], [ %1137, %1765 ]		; <uint>:1134 [#uses=7]
	add uint %1134, 4294967292		; <uint>:1135 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1135		; <ubyte*>:2643 [#uses=1]
	load ubyte* %2643		; <ubyte>:4176 [#uses=1]
	seteq ubyte %4176, 0		; <bool>:1763 [#uses=1]
	br bool %1763, label %1767, label %1766

; <label>:1764		; preds = %1762, %1764
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1133		; <ubyte*>:2644 [#uses=2]
	load ubyte* %2644		; <ubyte>:4177 [#uses=1]
	add ubyte %4177, 255		; <ubyte>:4178 [#uses=1]
	store ubyte %4178, ubyte* %2644
	add uint %1133, 6		; <uint>:1136 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1136		; <ubyte*>:2645 [#uses=2]
	load ubyte* %2645		; <ubyte>:4179 [#uses=1]
	add ubyte %4179, 1		; <ubyte>:4180 [#uses=1]
	store ubyte %4180, ubyte* %2645
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1133		; <ubyte*>:2646 [#uses=1]
	load ubyte* %2646		; <ubyte>:4181 [#uses=1]
	seteq ubyte %4181, 0		; <bool>:1764 [#uses=1]
	br bool %1764, label %1765, label %1764

; <label>:1765		; preds = %1762, %1764
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1133		; <ubyte*>:2647 [#uses=2]
	load ubyte* %2647		; <ubyte>:4182 [#uses=1]
	add ubyte %4182, 1		; <ubyte>:4183 [#uses=1]
	store ubyte %4183, ubyte* %2647
	add uint %1133, 6		; <uint>:1137 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1137		; <ubyte*>:2648 [#uses=2]
	load ubyte* %2648		; <ubyte>:4184 [#uses=2]
	add ubyte %4184, 255		; <ubyte>:4185 [#uses=1]
	store ubyte %4185, ubyte* %2648
	seteq ubyte %4184, 1		; <bool>:1765 [#uses=1]
	br bool %1765, label %1763, label %1762

; <label>:1766		; preds = %1763, %1766
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1135		; <ubyte*>:2649 [#uses=2]
	load ubyte* %2649		; <ubyte>:4186 [#uses=2]
	add ubyte %4186, 255		; <ubyte>:4187 [#uses=1]
	store ubyte %4187, ubyte* %2649
	seteq ubyte %4186, 1		; <bool>:1766 [#uses=1]
	br bool %1766, label %1767, label %1766

; <label>:1767		; preds = %1763, %1766
	add uint %1134, 4294967294		; <uint>:1138 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1138		; <ubyte*>:2650 [#uses=1]
	load ubyte* %2650		; <ubyte>:4188 [#uses=1]
	seteq ubyte %4188, 0		; <bool>:1767 [#uses=1]
	br bool %1767, label %1769, label %1768

; <label>:1768		; preds = %1767, %1768
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1135		; <ubyte*>:2651 [#uses=2]
	load ubyte* %2651		; <ubyte>:4189 [#uses=1]
	add ubyte %4189, 1		; <ubyte>:4190 [#uses=1]
	store ubyte %4190, ubyte* %2651
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1138		; <ubyte*>:2652 [#uses=2]
	load ubyte* %2652		; <ubyte>:4191 [#uses=1]
	add ubyte %4191, 255		; <ubyte>:4192 [#uses=1]
	store ubyte %4192, ubyte* %2652
	add uint %1134, 4294967295		; <uint>:1139 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1139		; <ubyte*>:2653 [#uses=2]
	load ubyte* %2653		; <ubyte>:4193 [#uses=1]
	add ubyte %4193, 1		; <ubyte>:4194 [#uses=1]
	store ubyte %4194, ubyte* %2653
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1138		; <ubyte*>:2654 [#uses=1]
	load ubyte* %2654		; <ubyte>:4195 [#uses=1]
	seteq ubyte %4195, 0		; <bool>:1768 [#uses=1]
	br bool %1768, label %1769, label %1768

; <label>:1769		; preds = %1767, %1768
	add uint %1134, 4294967295		; <uint>:1140 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1140		; <ubyte*>:2655 [#uses=1]
	load ubyte* %2655		; <ubyte>:4196 [#uses=1]
	seteq ubyte %4196, 0		; <bool>:1769 [#uses=1]
	br bool %1769, label %1771, label %1770

; <label>:1770		; preds = %1769, %1770
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1138		; <ubyte*>:2656 [#uses=2]
	load ubyte* %2656		; <ubyte>:4197 [#uses=1]
	add ubyte %4197, 1		; <ubyte>:4198 [#uses=1]
	store ubyte %4198, ubyte* %2656
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1140		; <ubyte*>:2657 [#uses=2]
	load ubyte* %2657		; <ubyte>:4199 [#uses=2]
	add ubyte %4199, 255		; <ubyte>:4200 [#uses=1]
	store ubyte %4200, ubyte* %2657
	seteq ubyte %4199, 1		; <bool>:1770 [#uses=1]
	br bool %1770, label %1771, label %1770

; <label>:1771		; preds = %1769, %1770
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1134		; <ubyte*>:2658 [#uses=2]
	load ubyte* %2658		; <ubyte>:4201 [#uses=2]
	add ubyte %4201, 1		; <ubyte>:4202 [#uses=1]
	store ubyte %4202, ubyte* %2658
	seteq ubyte %4201, 255		; <bool>:1771 [#uses=1]
	br bool %1771, label %1773, label %1772

; <label>:1772		; preds = %1771, %1777
	phi uint [ %1134, %1771 ], [ %1146, %1777 ]		; <uint>:1141 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1141		; <ubyte*>:2659 [#uses=2]
	load ubyte* %2659		; <ubyte>:4203 [#uses=1]
	add ubyte %4203, 255		; <ubyte>:4204 [#uses=1]
	store ubyte %4204, ubyte* %2659
	add uint %1141, 4294967286		; <uint>:1142 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1142		; <ubyte*>:2660 [#uses=1]
	load ubyte* %2660		; <ubyte>:4205 [#uses=1]
	seteq ubyte %4205, 0		; <bool>:1772 [#uses=1]
	br bool %1772, label %1775, label %1774

; <label>:1773		; preds = %1771, %1777
	phi uint [ %1134, %1771 ], [ %1146, %1777 ]		; <uint>:1143 [#uses=67]
	add uint %1143, 4		; <uint>:1144 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1144		; <ubyte*>:2661 [#uses=1]
	load ubyte* %2661		; <ubyte>:4206 [#uses=1]
	seteq ubyte %4206, 0		; <bool>:1773 [#uses=1]
	br bool %1773, label %1779, label %1778

; <label>:1774		; preds = %1772, %1774
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1142		; <ubyte*>:2662 [#uses=2]
	load ubyte* %2662		; <ubyte>:4207 [#uses=2]
	add ubyte %4207, 255		; <ubyte>:4208 [#uses=1]
	store ubyte %4208, ubyte* %2662
	seteq ubyte %4207, 1		; <bool>:1774 [#uses=1]
	br bool %1774, label %1775, label %1774

; <label>:1775		; preds = %1772, %1774
	add uint %1141, 4294967292		; <uint>:1145 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1145		; <ubyte*>:2663 [#uses=1]
	load ubyte* %2663		; <ubyte>:4209 [#uses=1]
	seteq ubyte %4209, 0		; <bool>:1775 [#uses=1]
	br bool %1775, label %1777, label %1776

; <label>:1776		; preds = %1775, %1776
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1142		; <ubyte*>:2664 [#uses=2]
	load ubyte* %2664		; <ubyte>:4210 [#uses=1]
	add ubyte %4210, 1		; <ubyte>:4211 [#uses=1]
	store ubyte %4211, ubyte* %2664
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1145		; <ubyte*>:2665 [#uses=2]
	load ubyte* %2665		; <ubyte>:4212 [#uses=2]
	add ubyte %4212, 255		; <ubyte>:4213 [#uses=1]
	store ubyte %4213, ubyte* %2665
	seteq ubyte %4212, 1		; <bool>:1776 [#uses=1]
	br bool %1776, label %1777, label %1776

; <label>:1777		; preds = %1775, %1776
	add uint %1141, 4294967290		; <uint>:1146 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1146		; <ubyte*>:2666 [#uses=1]
	load ubyte* %2666		; <ubyte>:4214 [#uses=1]
	seteq ubyte %4214, 0		; <bool>:1777 [#uses=1]
	br bool %1777, label %1773, label %1772

; <label>:1778		; preds = %1773, %1778
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1144		; <ubyte*>:2667 [#uses=2]
	load ubyte* %2667		; <ubyte>:4215 [#uses=2]
	add ubyte %4215, 255		; <ubyte>:4216 [#uses=1]
	store ubyte %4216, ubyte* %2667
	seteq ubyte %4215, 1		; <bool>:1778 [#uses=1]
	br bool %1778, label %1779, label %1778

; <label>:1779		; preds = %1773, %1778
	add uint %1143, 10		; <uint>:1147 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1147		; <ubyte*>:2668 [#uses=1]
	load ubyte* %2668		; <ubyte>:4217 [#uses=1]
	seteq ubyte %4217, 0		; <bool>:1779 [#uses=1]
	br bool %1779, label %1781, label %1780

; <label>:1780		; preds = %1779, %1780
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1147		; <ubyte*>:2669 [#uses=2]
	load ubyte* %2669		; <ubyte>:4218 [#uses=2]
	add ubyte %4218, 255		; <ubyte>:4219 [#uses=1]
	store ubyte %4219, ubyte* %2669
	seteq ubyte %4218, 1		; <bool>:1780 [#uses=1]
	br bool %1780, label %1781, label %1780

; <label>:1781		; preds = %1779, %1780
	add uint %1143, 16		; <uint>:1148 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1148		; <ubyte*>:2670 [#uses=1]
	load ubyte* %2670		; <ubyte>:4220 [#uses=1]
	seteq ubyte %4220, 0		; <bool>:1781 [#uses=1]
	br bool %1781, label %1783, label %1782

; <label>:1782		; preds = %1781, %1782
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1148		; <ubyte*>:2671 [#uses=2]
	load ubyte* %2671		; <ubyte>:4221 [#uses=2]
	add ubyte %4221, 255		; <ubyte>:4222 [#uses=1]
	store ubyte %4222, ubyte* %2671
	seteq ubyte %4221, 1		; <bool>:1782 [#uses=1]
	br bool %1782, label %1783, label %1782

; <label>:1783		; preds = %1781, %1782
	add uint %1143, 22		; <uint>:1149 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1149		; <ubyte*>:2672 [#uses=1]
	load ubyte* %2672		; <ubyte>:4223 [#uses=1]
	seteq ubyte %4223, 0		; <bool>:1783 [#uses=1]
	br bool %1783, label %1785, label %1784

; <label>:1784		; preds = %1783, %1784
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1149		; <ubyte*>:2673 [#uses=2]
	load ubyte* %2673		; <ubyte>:4224 [#uses=2]
	add ubyte %4224, 255		; <ubyte>:4225 [#uses=1]
	store ubyte %4225, ubyte* %2673
	seteq ubyte %4224, 1		; <bool>:1784 [#uses=1]
	br bool %1784, label %1785, label %1784

; <label>:1785		; preds = %1783, %1784
	add uint %1143, 28		; <uint>:1150 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1150		; <ubyte*>:2674 [#uses=1]
	load ubyte* %2674		; <ubyte>:4226 [#uses=1]
	seteq ubyte %4226, 0		; <bool>:1785 [#uses=1]
	br bool %1785, label %1787, label %1786

; <label>:1786		; preds = %1785, %1786
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1150		; <ubyte*>:2675 [#uses=2]
	load ubyte* %2675		; <ubyte>:4227 [#uses=2]
	add ubyte %4227, 255		; <ubyte>:4228 [#uses=1]
	store ubyte %4228, ubyte* %2675
	seteq ubyte %4227, 1		; <bool>:1786 [#uses=1]
	br bool %1786, label %1787, label %1786

; <label>:1787		; preds = %1785, %1786
	add uint %1143, 34		; <uint>:1151 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1151		; <ubyte*>:2676 [#uses=1]
	load ubyte* %2676		; <ubyte>:4229 [#uses=1]
	seteq ubyte %4229, 0		; <bool>:1787 [#uses=1]
	br bool %1787, label %1789, label %1788

; <label>:1788		; preds = %1787, %1788
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1151		; <ubyte*>:2677 [#uses=2]
	load ubyte* %2677		; <ubyte>:4230 [#uses=2]
	add ubyte %4230, 255		; <ubyte>:4231 [#uses=1]
	store ubyte %4231, ubyte* %2677
	seteq ubyte %4230, 1		; <bool>:1788 [#uses=1]
	br bool %1788, label %1789, label %1788

; <label>:1789		; preds = %1787, %1788
	add uint %1143, 40		; <uint>:1152 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1152		; <ubyte*>:2678 [#uses=1]
	load ubyte* %2678		; <ubyte>:4232 [#uses=1]
	seteq ubyte %4232, 0		; <bool>:1789 [#uses=1]
	br bool %1789, label %1791, label %1790

; <label>:1790		; preds = %1789, %1790
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1152		; <ubyte*>:2679 [#uses=2]
	load ubyte* %2679		; <ubyte>:4233 [#uses=2]
	add ubyte %4233, 255		; <ubyte>:4234 [#uses=1]
	store ubyte %4234, ubyte* %2679
	seteq ubyte %4233, 1		; <bool>:1790 [#uses=1]
	br bool %1790, label %1791, label %1790

; <label>:1791		; preds = %1789, %1790
	add uint %1143, 46		; <uint>:1153 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1153		; <ubyte*>:2680 [#uses=1]
	load ubyte* %2680		; <ubyte>:4235 [#uses=1]
	seteq ubyte %4235, 0		; <bool>:1791 [#uses=1]
	br bool %1791, label %1793, label %1792

; <label>:1792		; preds = %1791, %1792
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1153		; <ubyte*>:2681 [#uses=2]
	load ubyte* %2681		; <ubyte>:4236 [#uses=2]
	add ubyte %4236, 255		; <ubyte>:4237 [#uses=1]
	store ubyte %4237, ubyte* %2681
	seteq ubyte %4236, 1		; <bool>:1792 [#uses=1]
	br bool %1792, label %1793, label %1792

; <label>:1793		; preds = %1791, %1792
	add uint %1143, 52		; <uint>:1154 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1154		; <ubyte*>:2682 [#uses=1]
	load ubyte* %2682		; <ubyte>:4238 [#uses=1]
	seteq ubyte %4238, 0		; <bool>:1793 [#uses=1]
	br bool %1793, label %1795, label %1794

; <label>:1794		; preds = %1793, %1794
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1154		; <ubyte*>:2683 [#uses=2]
	load ubyte* %2683		; <ubyte>:4239 [#uses=2]
	add ubyte %4239, 255		; <ubyte>:4240 [#uses=1]
	store ubyte %4240, ubyte* %2683
	seteq ubyte %4239, 1		; <bool>:1794 [#uses=1]
	br bool %1794, label %1795, label %1794

; <label>:1795		; preds = %1793, %1794
	add uint %1143, 58		; <uint>:1155 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1155		; <ubyte*>:2684 [#uses=1]
	load ubyte* %2684		; <ubyte>:4241 [#uses=1]
	seteq ubyte %4241, 0		; <bool>:1795 [#uses=1]
	br bool %1795, label %1797, label %1796

; <label>:1796		; preds = %1795, %1796
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1155		; <ubyte*>:2685 [#uses=2]
	load ubyte* %2685		; <ubyte>:4242 [#uses=2]
	add ubyte %4242, 255		; <ubyte>:4243 [#uses=1]
	store ubyte %4243, ubyte* %2685
	seteq ubyte %4242, 1		; <bool>:1796 [#uses=1]
	br bool %1796, label %1797, label %1796

; <label>:1797		; preds = %1795, %1796
	add uint %1143, 64		; <uint>:1156 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1156		; <ubyte*>:2686 [#uses=1]
	load ubyte* %2686		; <ubyte>:4244 [#uses=1]
	seteq ubyte %4244, 0		; <bool>:1797 [#uses=1]
	br bool %1797, label %1799, label %1798

; <label>:1798		; preds = %1797, %1798
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1156		; <ubyte*>:2687 [#uses=2]
	load ubyte* %2687		; <ubyte>:4245 [#uses=2]
	add ubyte %4245, 255		; <ubyte>:4246 [#uses=1]
	store ubyte %4246, ubyte* %2687
	seteq ubyte %4245, 1		; <bool>:1798 [#uses=1]
	br bool %1798, label %1799, label %1798

; <label>:1799		; preds = %1797, %1798
	add uint %1143, 70		; <uint>:1157 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1157		; <ubyte*>:2688 [#uses=1]
	load ubyte* %2688		; <ubyte>:4247 [#uses=1]
	seteq ubyte %4247, 0		; <bool>:1799 [#uses=1]
	br bool %1799, label %1801, label %1800

; <label>:1800		; preds = %1799, %1800
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1157		; <ubyte*>:2689 [#uses=2]
	load ubyte* %2689		; <ubyte>:4248 [#uses=2]
	add ubyte %4248, 255		; <ubyte>:4249 [#uses=1]
	store ubyte %4249, ubyte* %2689
	seteq ubyte %4248, 1		; <bool>:1800 [#uses=1]
	br bool %1800, label %1801, label %1800

; <label>:1801		; preds = %1799, %1800
	add uint %1143, 76		; <uint>:1158 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1158		; <ubyte*>:2690 [#uses=1]
	load ubyte* %2690		; <ubyte>:4250 [#uses=1]
	seteq ubyte %4250, 0		; <bool>:1801 [#uses=1]
	br bool %1801, label %1803, label %1802

; <label>:1802		; preds = %1801, %1802
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1158		; <ubyte*>:2691 [#uses=2]
	load ubyte* %2691		; <ubyte>:4251 [#uses=2]
	add ubyte %4251, 255		; <ubyte>:4252 [#uses=1]
	store ubyte %4252, ubyte* %2691
	seteq ubyte %4251, 1		; <bool>:1802 [#uses=1]
	br bool %1802, label %1803, label %1802

; <label>:1803		; preds = %1801, %1802
	add uint %1143, 82		; <uint>:1159 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1159		; <ubyte*>:2692 [#uses=1]
	load ubyte* %2692		; <ubyte>:4253 [#uses=1]
	seteq ubyte %4253, 0		; <bool>:1803 [#uses=1]
	br bool %1803, label %1805, label %1804

; <label>:1804		; preds = %1803, %1804
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1159		; <ubyte*>:2693 [#uses=2]
	load ubyte* %2693		; <ubyte>:4254 [#uses=2]
	add ubyte %4254, 255		; <ubyte>:4255 [#uses=1]
	store ubyte %4255, ubyte* %2693
	seteq ubyte %4254, 1		; <bool>:1804 [#uses=1]
	br bool %1804, label %1805, label %1804

; <label>:1805		; preds = %1803, %1804
	add uint %1143, 88		; <uint>:1160 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1160		; <ubyte*>:2694 [#uses=1]
	load ubyte* %2694		; <ubyte>:4256 [#uses=1]
	seteq ubyte %4256, 0		; <bool>:1805 [#uses=1]
	br bool %1805, label %1807, label %1806

; <label>:1806		; preds = %1805, %1806
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1160		; <ubyte*>:2695 [#uses=2]
	load ubyte* %2695		; <ubyte>:4257 [#uses=2]
	add ubyte %4257, 255		; <ubyte>:4258 [#uses=1]
	store ubyte %4258, ubyte* %2695
	seteq ubyte %4257, 1		; <bool>:1806 [#uses=1]
	br bool %1806, label %1807, label %1806

; <label>:1807		; preds = %1805, %1806
	add uint %1143, 4294967290		; <uint>:1161 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1161		; <ubyte*>:2696 [#uses=1]
	load ubyte* %2696		; <ubyte>:4259 [#uses=1]
	seteq ubyte %4259, 0		; <bool>:1807 [#uses=1]
	br bool %1807, label %1809, label %1808

; <label>:1808		; preds = %1807, %1808
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1161		; <ubyte*>:2697 [#uses=2]
	load ubyte* %2697		; <ubyte>:4260 [#uses=2]
	add ubyte %4260, 255		; <ubyte>:4261 [#uses=1]
	store ubyte %4261, ubyte* %2697
	seteq ubyte %4260, 1		; <bool>:1808 [#uses=1]
	br bool %1808, label %1809, label %1808

; <label>:1809		; preds = %1807, %1808
	add uint %1143, 4294967292		; <uint>:1162 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1162		; <ubyte*>:2698 [#uses=1]
	load ubyte* %2698		; <ubyte>:4262 [#uses=1]
	seteq ubyte %4262, 0		; <bool>:1809 [#uses=1]
	br bool %1809, label %1811, label %1810

; <label>:1810		; preds = %1809, %1810
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1161		; <ubyte*>:2699 [#uses=2]
	load ubyte* %2699		; <ubyte>:4263 [#uses=1]
	add ubyte %4263, 1		; <ubyte>:4264 [#uses=1]
	store ubyte %4264, ubyte* %2699
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1162		; <ubyte*>:2700 [#uses=2]
	load ubyte* %2700		; <ubyte>:4265 [#uses=2]
	add ubyte %4265, 255		; <ubyte>:4266 [#uses=1]
	store ubyte %4266, ubyte* %2700
	seteq ubyte %4265, 1		; <bool>:1810 [#uses=1]
	br bool %1810, label %1811, label %1810

; <label>:1811		; preds = %1809, %1810
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1144		; <ubyte*>:2701 [#uses=1]
	load ubyte* %2701		; <ubyte>:4267 [#uses=1]
	seteq ubyte %4267, 0		; <bool>:1811 [#uses=1]
	br bool %1811, label %1813, label %1812

; <label>:1812		; preds = %1811, %1812
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1144		; <ubyte*>:2702 [#uses=2]
	load ubyte* %2702		; <ubyte>:4268 [#uses=2]
	add ubyte %4268, 255		; <ubyte>:4269 [#uses=1]
	store ubyte %4269, ubyte* %2702
	seteq ubyte %4268, 1		; <bool>:1812 [#uses=1]
	br bool %1812, label %1813, label %1812

; <label>:1813		; preds = %1811, %1812
	add uint %1143, 4294967187		; <uint>:1163 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1163		; <ubyte*>:2703 [#uses=1]
	load ubyte* %2703		; <ubyte>:4270 [#uses=1]
	seteq ubyte %4270, 0		; <bool>:1813 [#uses=1]
	br bool %1813, label %1815, label %1814

; <label>:1814		; preds = %1813, %1814
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1163		; <ubyte*>:2704 [#uses=2]
	load ubyte* %2704		; <ubyte>:4271 [#uses=1]
	add ubyte %4271, 255		; <ubyte>:4272 [#uses=1]
	store ubyte %4272, ubyte* %2704
	add uint %1143, 4294967188		; <uint>:1164 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1164		; <ubyte*>:2705 [#uses=2]
	load ubyte* %2705		; <ubyte>:4273 [#uses=1]
	add ubyte %4273, 1		; <ubyte>:4274 [#uses=1]
	store ubyte %4274, ubyte* %2705
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1144		; <ubyte*>:2706 [#uses=2]
	load ubyte* %2706		; <ubyte>:4275 [#uses=1]
	add ubyte %4275, 1		; <ubyte>:4276 [#uses=1]
	store ubyte %4276, ubyte* %2706
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1163		; <ubyte*>:2707 [#uses=1]
	load ubyte* %2707		; <ubyte>:4277 [#uses=1]
	seteq ubyte %4277, 0		; <bool>:1814 [#uses=1]
	br bool %1814, label %1815, label %1814

; <label>:1815		; preds = %1813, %1814
	add uint %1143, 4294967188		; <uint>:1165 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1165		; <ubyte*>:2708 [#uses=1]
	load ubyte* %2708		; <ubyte>:4278 [#uses=1]
	seteq ubyte %4278, 0		; <bool>:1815 [#uses=1]
	br bool %1815, label %1817, label %1816

; <label>:1816		; preds = %1815, %1816
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1163		; <ubyte*>:2709 [#uses=2]
	load ubyte* %2709		; <ubyte>:4279 [#uses=1]
	add ubyte %4279, 1		; <ubyte>:4280 [#uses=1]
	store ubyte %4280, ubyte* %2709
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1165		; <ubyte*>:2710 [#uses=2]
	load ubyte* %2710		; <ubyte>:4281 [#uses=2]
	add ubyte %4281, 255		; <ubyte>:4282 [#uses=1]
	store ubyte %4282, ubyte* %2710
	seteq ubyte %4281, 1		; <bool>:1816 [#uses=1]
	br bool %1816, label %1817, label %1816

; <label>:1817		; preds = %1815, %1816
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1147		; <ubyte*>:2711 [#uses=1]
	load ubyte* %2711		; <ubyte>:4283 [#uses=1]
	seteq ubyte %4283, 0		; <bool>:1817 [#uses=1]
	br bool %1817, label %1819, label %1818

; <label>:1818		; preds = %1817, %1818
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1147		; <ubyte*>:2712 [#uses=2]
	load ubyte* %2712		; <ubyte>:4284 [#uses=2]
	add ubyte %4284, 255		; <ubyte>:4285 [#uses=1]
	store ubyte %4285, ubyte* %2712
	seteq ubyte %4284, 1		; <bool>:1818 [#uses=1]
	br bool %1818, label %1819, label %1818

; <label>:1819		; preds = %1817, %1818
	add uint %1143, 4294967193		; <uint>:1166 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1166		; <ubyte*>:2713 [#uses=1]
	load ubyte* %2713		; <ubyte>:4286 [#uses=1]
	seteq ubyte %4286, 0		; <bool>:1819 [#uses=1]
	br bool %1819, label %1821, label %1820

; <label>:1820		; preds = %1819, %1820
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1166		; <ubyte*>:2714 [#uses=2]
	load ubyte* %2714		; <ubyte>:4287 [#uses=1]
	add ubyte %4287, 255		; <ubyte>:4288 [#uses=1]
	store ubyte %4288, ubyte* %2714
	add uint %1143, 4294967194		; <uint>:1167 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1167		; <ubyte*>:2715 [#uses=2]
	load ubyte* %2715		; <ubyte>:4289 [#uses=1]
	add ubyte %4289, 1		; <ubyte>:4290 [#uses=1]
	store ubyte %4290, ubyte* %2715
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1147		; <ubyte*>:2716 [#uses=2]
	load ubyte* %2716		; <ubyte>:4291 [#uses=1]
	add ubyte %4291, 1		; <ubyte>:4292 [#uses=1]
	store ubyte %4292, ubyte* %2716
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1166		; <ubyte*>:2717 [#uses=1]
	load ubyte* %2717		; <ubyte>:4293 [#uses=1]
	seteq ubyte %4293, 0		; <bool>:1820 [#uses=1]
	br bool %1820, label %1821, label %1820

; <label>:1821		; preds = %1819, %1820
	add uint %1143, 4294967194		; <uint>:1168 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1168		; <ubyte*>:2718 [#uses=1]
	load ubyte* %2718		; <ubyte>:4294 [#uses=1]
	seteq ubyte %4294, 0		; <bool>:1821 [#uses=1]
	br bool %1821, label %1823, label %1822

; <label>:1822		; preds = %1821, %1822
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1166		; <ubyte*>:2719 [#uses=2]
	load ubyte* %2719		; <ubyte>:4295 [#uses=1]
	add ubyte %4295, 1		; <ubyte>:4296 [#uses=1]
	store ubyte %4296, ubyte* %2719
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1168		; <ubyte*>:2720 [#uses=2]
	load ubyte* %2720		; <ubyte>:4297 [#uses=2]
	add ubyte %4297, 255		; <ubyte>:4298 [#uses=1]
	store ubyte %4298, ubyte* %2720
	seteq ubyte %4297, 1		; <bool>:1822 [#uses=1]
	br bool %1822, label %1823, label %1822

; <label>:1823		; preds = %1821, %1822
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1148		; <ubyte*>:2721 [#uses=1]
	load ubyte* %2721		; <ubyte>:4299 [#uses=1]
	seteq ubyte %4299, 0		; <bool>:1823 [#uses=1]
	br bool %1823, label %1825, label %1824

; <label>:1824		; preds = %1823, %1824
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1148		; <ubyte*>:2722 [#uses=2]
	load ubyte* %2722		; <ubyte>:4300 [#uses=2]
	add ubyte %4300, 255		; <ubyte>:4301 [#uses=1]
	store ubyte %4301, ubyte* %2722
	seteq ubyte %4300, 1		; <bool>:1824 [#uses=1]
	br bool %1824, label %1825, label %1824

; <label>:1825		; preds = %1823, %1824
	add uint %1143, 4294967199		; <uint>:1169 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1169		; <ubyte*>:2723 [#uses=1]
	load ubyte* %2723		; <ubyte>:4302 [#uses=1]
	seteq ubyte %4302, 0		; <bool>:1825 [#uses=1]
	br bool %1825, label %1827, label %1826

; <label>:1826		; preds = %1825, %1826
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1169		; <ubyte*>:2724 [#uses=2]
	load ubyte* %2724		; <ubyte>:4303 [#uses=1]
	add ubyte %4303, 255		; <ubyte>:4304 [#uses=1]
	store ubyte %4304, ubyte* %2724
	add uint %1143, 4294967200		; <uint>:1170 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1170		; <ubyte*>:2725 [#uses=2]
	load ubyte* %2725		; <ubyte>:4305 [#uses=1]
	add ubyte %4305, 1		; <ubyte>:4306 [#uses=1]
	store ubyte %4306, ubyte* %2725
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1148		; <ubyte*>:2726 [#uses=2]
	load ubyte* %2726		; <ubyte>:4307 [#uses=1]
	add ubyte %4307, 1		; <ubyte>:4308 [#uses=1]
	store ubyte %4308, ubyte* %2726
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1169		; <ubyte*>:2727 [#uses=1]
	load ubyte* %2727		; <ubyte>:4309 [#uses=1]
	seteq ubyte %4309, 0		; <bool>:1826 [#uses=1]
	br bool %1826, label %1827, label %1826

; <label>:1827		; preds = %1825, %1826
	add uint %1143, 4294967200		; <uint>:1171 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1171		; <ubyte*>:2728 [#uses=1]
	load ubyte* %2728		; <ubyte>:4310 [#uses=1]
	seteq ubyte %4310, 0		; <bool>:1827 [#uses=1]
	br bool %1827, label %1829, label %1828

; <label>:1828		; preds = %1827, %1828
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1169		; <ubyte*>:2729 [#uses=2]
	load ubyte* %2729		; <ubyte>:4311 [#uses=1]
	add ubyte %4311, 1		; <ubyte>:4312 [#uses=1]
	store ubyte %4312, ubyte* %2729
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1171		; <ubyte*>:2730 [#uses=2]
	load ubyte* %2730		; <ubyte>:4313 [#uses=2]
	add ubyte %4313, 255		; <ubyte>:4314 [#uses=1]
	store ubyte %4314, ubyte* %2730
	seteq ubyte %4313, 1		; <bool>:1828 [#uses=1]
	br bool %1828, label %1829, label %1828

; <label>:1829		; preds = %1827, %1828
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1149		; <ubyte*>:2731 [#uses=1]
	load ubyte* %2731		; <ubyte>:4315 [#uses=1]
	seteq ubyte %4315, 0		; <bool>:1829 [#uses=1]
	br bool %1829, label %1831, label %1830

; <label>:1830		; preds = %1829, %1830
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1149		; <ubyte*>:2732 [#uses=2]
	load ubyte* %2732		; <ubyte>:4316 [#uses=2]
	add ubyte %4316, 255		; <ubyte>:4317 [#uses=1]
	store ubyte %4317, ubyte* %2732
	seteq ubyte %4316, 1		; <bool>:1830 [#uses=1]
	br bool %1830, label %1831, label %1830

; <label>:1831		; preds = %1829, %1830
	add uint %1143, 4294967205		; <uint>:1172 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1172		; <ubyte*>:2733 [#uses=1]
	load ubyte* %2733		; <ubyte>:4318 [#uses=1]
	seteq ubyte %4318, 0		; <bool>:1831 [#uses=1]
	br bool %1831, label %1833, label %1832

; <label>:1832		; preds = %1831, %1832
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1172		; <ubyte*>:2734 [#uses=2]
	load ubyte* %2734		; <ubyte>:4319 [#uses=1]
	add ubyte %4319, 255		; <ubyte>:4320 [#uses=1]
	store ubyte %4320, ubyte* %2734
	add uint %1143, 4294967206		; <uint>:1173 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1173		; <ubyte*>:2735 [#uses=2]
	load ubyte* %2735		; <ubyte>:4321 [#uses=1]
	add ubyte %4321, 1		; <ubyte>:4322 [#uses=1]
	store ubyte %4322, ubyte* %2735
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1149		; <ubyte*>:2736 [#uses=2]
	load ubyte* %2736		; <ubyte>:4323 [#uses=1]
	add ubyte %4323, 1		; <ubyte>:4324 [#uses=1]
	store ubyte %4324, ubyte* %2736
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1172		; <ubyte*>:2737 [#uses=1]
	load ubyte* %2737		; <ubyte>:4325 [#uses=1]
	seteq ubyte %4325, 0		; <bool>:1832 [#uses=1]
	br bool %1832, label %1833, label %1832

; <label>:1833		; preds = %1831, %1832
	add uint %1143, 4294967206		; <uint>:1174 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1174		; <ubyte*>:2738 [#uses=1]
	load ubyte* %2738		; <ubyte>:4326 [#uses=1]
	seteq ubyte %4326, 0		; <bool>:1833 [#uses=1]
	br bool %1833, label %1835, label %1834

; <label>:1834		; preds = %1833, %1834
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1172		; <ubyte*>:2739 [#uses=2]
	load ubyte* %2739		; <ubyte>:4327 [#uses=1]
	add ubyte %4327, 1		; <ubyte>:4328 [#uses=1]
	store ubyte %4328, ubyte* %2739
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1174		; <ubyte*>:2740 [#uses=2]
	load ubyte* %2740		; <ubyte>:4329 [#uses=2]
	add ubyte %4329, 255		; <ubyte>:4330 [#uses=1]
	store ubyte %4330, ubyte* %2740
	seteq ubyte %4329, 1		; <bool>:1834 [#uses=1]
	br bool %1834, label %1835, label %1834

; <label>:1835		; preds = %1833, %1834
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1150		; <ubyte*>:2741 [#uses=1]
	load ubyte* %2741		; <ubyte>:4331 [#uses=1]
	seteq ubyte %4331, 0		; <bool>:1835 [#uses=1]
	br bool %1835, label %1837, label %1836

; <label>:1836		; preds = %1835, %1836
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1150		; <ubyte*>:2742 [#uses=2]
	load ubyte* %2742		; <ubyte>:4332 [#uses=2]
	add ubyte %4332, 255		; <ubyte>:4333 [#uses=1]
	store ubyte %4333, ubyte* %2742
	seteq ubyte %4332, 1		; <bool>:1836 [#uses=1]
	br bool %1836, label %1837, label %1836

; <label>:1837		; preds = %1835, %1836
	add uint %1143, 4294967211		; <uint>:1175 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1175		; <ubyte*>:2743 [#uses=1]
	load ubyte* %2743		; <ubyte>:4334 [#uses=1]
	seteq ubyte %4334, 0		; <bool>:1837 [#uses=1]
	br bool %1837, label %1839, label %1838

; <label>:1838		; preds = %1837, %1838
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1175		; <ubyte*>:2744 [#uses=2]
	load ubyte* %2744		; <ubyte>:4335 [#uses=1]
	add ubyte %4335, 255		; <ubyte>:4336 [#uses=1]
	store ubyte %4336, ubyte* %2744
	add uint %1143, 4294967212		; <uint>:1176 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1176		; <ubyte*>:2745 [#uses=2]
	load ubyte* %2745		; <ubyte>:4337 [#uses=1]
	add ubyte %4337, 1		; <ubyte>:4338 [#uses=1]
	store ubyte %4338, ubyte* %2745
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1150		; <ubyte*>:2746 [#uses=2]
	load ubyte* %2746		; <ubyte>:4339 [#uses=1]
	add ubyte %4339, 1		; <ubyte>:4340 [#uses=1]
	store ubyte %4340, ubyte* %2746
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1175		; <ubyte*>:2747 [#uses=1]
	load ubyte* %2747		; <ubyte>:4341 [#uses=1]
	seteq ubyte %4341, 0		; <bool>:1838 [#uses=1]
	br bool %1838, label %1839, label %1838

; <label>:1839		; preds = %1837, %1838
	add uint %1143, 4294967212		; <uint>:1177 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1177		; <ubyte*>:2748 [#uses=1]
	load ubyte* %2748		; <ubyte>:4342 [#uses=1]
	seteq ubyte %4342, 0		; <bool>:1839 [#uses=1]
	br bool %1839, label %1841, label %1840

; <label>:1840		; preds = %1839, %1840
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1175		; <ubyte*>:2749 [#uses=2]
	load ubyte* %2749		; <ubyte>:4343 [#uses=1]
	add ubyte %4343, 1		; <ubyte>:4344 [#uses=1]
	store ubyte %4344, ubyte* %2749
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1177		; <ubyte*>:2750 [#uses=2]
	load ubyte* %2750		; <ubyte>:4345 [#uses=2]
	add ubyte %4345, 255		; <ubyte>:4346 [#uses=1]
	store ubyte %4346, ubyte* %2750
	seteq ubyte %4345, 1		; <bool>:1840 [#uses=1]
	br bool %1840, label %1841, label %1840

; <label>:1841		; preds = %1839, %1840
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1151		; <ubyte*>:2751 [#uses=1]
	load ubyte* %2751		; <ubyte>:4347 [#uses=1]
	seteq ubyte %4347, 0		; <bool>:1841 [#uses=1]
	br bool %1841, label %1843, label %1842

; <label>:1842		; preds = %1841, %1842
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1151		; <ubyte*>:2752 [#uses=2]
	load ubyte* %2752		; <ubyte>:4348 [#uses=2]
	add ubyte %4348, 255		; <ubyte>:4349 [#uses=1]
	store ubyte %4349, ubyte* %2752
	seteq ubyte %4348, 1		; <bool>:1842 [#uses=1]
	br bool %1842, label %1843, label %1842

; <label>:1843		; preds = %1841, %1842
	add uint %1143, 4294967217		; <uint>:1178 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1178		; <ubyte*>:2753 [#uses=1]
	load ubyte* %2753		; <ubyte>:4350 [#uses=1]
	seteq ubyte %4350, 0		; <bool>:1843 [#uses=1]
	br bool %1843, label %1845, label %1844

; <label>:1844		; preds = %1843, %1844
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1178		; <ubyte*>:2754 [#uses=2]
	load ubyte* %2754		; <ubyte>:4351 [#uses=1]
	add ubyte %4351, 255		; <ubyte>:4352 [#uses=1]
	store ubyte %4352, ubyte* %2754
	add uint %1143, 4294967218		; <uint>:1179 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1179		; <ubyte*>:2755 [#uses=2]
	load ubyte* %2755		; <ubyte>:4353 [#uses=1]
	add ubyte %4353, 1		; <ubyte>:4354 [#uses=1]
	store ubyte %4354, ubyte* %2755
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1151		; <ubyte*>:2756 [#uses=2]
	load ubyte* %2756		; <ubyte>:4355 [#uses=1]
	add ubyte %4355, 1		; <ubyte>:4356 [#uses=1]
	store ubyte %4356, ubyte* %2756
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1178		; <ubyte*>:2757 [#uses=1]
	load ubyte* %2757		; <ubyte>:4357 [#uses=1]
	seteq ubyte %4357, 0		; <bool>:1844 [#uses=1]
	br bool %1844, label %1845, label %1844

; <label>:1845		; preds = %1843, %1844
	add uint %1143, 4294967218		; <uint>:1180 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1180		; <ubyte*>:2758 [#uses=1]
	load ubyte* %2758		; <ubyte>:4358 [#uses=1]
	seteq ubyte %4358, 0		; <bool>:1845 [#uses=1]
	br bool %1845, label %1847, label %1846

; <label>:1846		; preds = %1845, %1846
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1178		; <ubyte*>:2759 [#uses=2]
	load ubyte* %2759		; <ubyte>:4359 [#uses=1]
	add ubyte %4359, 1		; <ubyte>:4360 [#uses=1]
	store ubyte %4360, ubyte* %2759
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1180		; <ubyte*>:2760 [#uses=2]
	load ubyte* %2760		; <ubyte>:4361 [#uses=2]
	add ubyte %4361, 255		; <ubyte>:4362 [#uses=1]
	store ubyte %4362, ubyte* %2760
	seteq ubyte %4361, 1		; <bool>:1846 [#uses=1]
	br bool %1846, label %1847, label %1846

; <label>:1847		; preds = %1845, %1846
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1152		; <ubyte*>:2761 [#uses=1]
	load ubyte* %2761		; <ubyte>:4363 [#uses=1]
	seteq ubyte %4363, 0		; <bool>:1847 [#uses=1]
	br bool %1847, label %1849, label %1848

; <label>:1848		; preds = %1847, %1848
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1152		; <ubyte*>:2762 [#uses=2]
	load ubyte* %2762		; <ubyte>:4364 [#uses=2]
	add ubyte %4364, 255		; <ubyte>:4365 [#uses=1]
	store ubyte %4365, ubyte* %2762
	seteq ubyte %4364, 1		; <bool>:1848 [#uses=1]
	br bool %1848, label %1849, label %1848

; <label>:1849		; preds = %1847, %1848
	add uint %1143, 4294967223		; <uint>:1181 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1181		; <ubyte*>:2763 [#uses=1]
	load ubyte* %2763		; <ubyte>:4366 [#uses=1]
	seteq ubyte %4366, 0		; <bool>:1849 [#uses=1]
	br bool %1849, label %1851, label %1850

; <label>:1850		; preds = %1849, %1850
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1181		; <ubyte*>:2764 [#uses=2]
	load ubyte* %2764		; <ubyte>:4367 [#uses=1]
	add ubyte %4367, 255		; <ubyte>:4368 [#uses=1]
	store ubyte %4368, ubyte* %2764
	add uint %1143, 4294967224		; <uint>:1182 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1182		; <ubyte*>:2765 [#uses=2]
	load ubyte* %2765		; <ubyte>:4369 [#uses=1]
	add ubyte %4369, 1		; <ubyte>:4370 [#uses=1]
	store ubyte %4370, ubyte* %2765
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1152		; <ubyte*>:2766 [#uses=2]
	load ubyte* %2766		; <ubyte>:4371 [#uses=1]
	add ubyte %4371, 1		; <ubyte>:4372 [#uses=1]
	store ubyte %4372, ubyte* %2766
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1181		; <ubyte*>:2767 [#uses=1]
	load ubyte* %2767		; <ubyte>:4373 [#uses=1]
	seteq ubyte %4373, 0		; <bool>:1850 [#uses=1]
	br bool %1850, label %1851, label %1850

; <label>:1851		; preds = %1849, %1850
	add uint %1143, 4294967224		; <uint>:1183 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1183		; <ubyte*>:2768 [#uses=1]
	load ubyte* %2768		; <ubyte>:4374 [#uses=1]
	seteq ubyte %4374, 0		; <bool>:1851 [#uses=1]
	br bool %1851, label %1853, label %1852

; <label>:1852		; preds = %1851, %1852
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1181		; <ubyte*>:2769 [#uses=2]
	load ubyte* %2769		; <ubyte>:4375 [#uses=1]
	add ubyte %4375, 1		; <ubyte>:4376 [#uses=1]
	store ubyte %4376, ubyte* %2769
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1183		; <ubyte*>:2770 [#uses=2]
	load ubyte* %2770		; <ubyte>:4377 [#uses=2]
	add ubyte %4377, 255		; <ubyte>:4378 [#uses=1]
	store ubyte %4378, ubyte* %2770
	seteq ubyte %4377, 1		; <bool>:1852 [#uses=1]
	br bool %1852, label %1853, label %1852

; <label>:1853		; preds = %1851, %1852
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1153		; <ubyte*>:2771 [#uses=1]
	load ubyte* %2771		; <ubyte>:4379 [#uses=1]
	seteq ubyte %4379, 0		; <bool>:1853 [#uses=1]
	br bool %1853, label %1855, label %1854

; <label>:1854		; preds = %1853, %1854
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1153		; <ubyte*>:2772 [#uses=2]
	load ubyte* %2772		; <ubyte>:4380 [#uses=2]
	add ubyte %4380, 255		; <ubyte>:4381 [#uses=1]
	store ubyte %4381, ubyte* %2772
	seteq ubyte %4380, 1		; <bool>:1854 [#uses=1]
	br bool %1854, label %1855, label %1854

; <label>:1855		; preds = %1853, %1854
	add uint %1143, 4294967229		; <uint>:1184 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1184		; <ubyte*>:2773 [#uses=1]
	load ubyte* %2773		; <ubyte>:4382 [#uses=1]
	seteq ubyte %4382, 0		; <bool>:1855 [#uses=1]
	br bool %1855, label %1857, label %1856

; <label>:1856		; preds = %1855, %1856
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1184		; <ubyte*>:2774 [#uses=2]
	load ubyte* %2774		; <ubyte>:4383 [#uses=1]
	add ubyte %4383, 255		; <ubyte>:4384 [#uses=1]
	store ubyte %4384, ubyte* %2774
	add uint %1143, 4294967230		; <uint>:1185 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1185		; <ubyte*>:2775 [#uses=2]
	load ubyte* %2775		; <ubyte>:4385 [#uses=1]
	add ubyte %4385, 1		; <ubyte>:4386 [#uses=1]
	store ubyte %4386, ubyte* %2775
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1153		; <ubyte*>:2776 [#uses=2]
	load ubyte* %2776		; <ubyte>:4387 [#uses=1]
	add ubyte %4387, 1		; <ubyte>:4388 [#uses=1]
	store ubyte %4388, ubyte* %2776
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1184		; <ubyte*>:2777 [#uses=1]
	load ubyte* %2777		; <ubyte>:4389 [#uses=1]
	seteq ubyte %4389, 0		; <bool>:1856 [#uses=1]
	br bool %1856, label %1857, label %1856

; <label>:1857		; preds = %1855, %1856
	add uint %1143, 4294967230		; <uint>:1186 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1186		; <ubyte*>:2778 [#uses=1]
	load ubyte* %2778		; <ubyte>:4390 [#uses=1]
	seteq ubyte %4390, 0		; <bool>:1857 [#uses=1]
	br bool %1857, label %1859, label %1858

; <label>:1858		; preds = %1857, %1858
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1184		; <ubyte*>:2779 [#uses=2]
	load ubyte* %2779		; <ubyte>:4391 [#uses=1]
	add ubyte %4391, 1		; <ubyte>:4392 [#uses=1]
	store ubyte %4392, ubyte* %2779
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1186		; <ubyte*>:2780 [#uses=2]
	load ubyte* %2780		; <ubyte>:4393 [#uses=2]
	add ubyte %4393, 255		; <ubyte>:4394 [#uses=1]
	store ubyte %4394, ubyte* %2780
	seteq ubyte %4393, 1		; <bool>:1858 [#uses=1]
	br bool %1858, label %1859, label %1858

; <label>:1859		; preds = %1857, %1858
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1154		; <ubyte*>:2781 [#uses=1]
	load ubyte* %2781		; <ubyte>:4395 [#uses=1]
	seteq ubyte %4395, 0		; <bool>:1859 [#uses=1]
	br bool %1859, label %1861, label %1860

; <label>:1860		; preds = %1859, %1860
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1154		; <ubyte*>:2782 [#uses=2]
	load ubyte* %2782		; <ubyte>:4396 [#uses=2]
	add ubyte %4396, 255		; <ubyte>:4397 [#uses=1]
	store ubyte %4397, ubyte* %2782
	seteq ubyte %4396, 1		; <bool>:1860 [#uses=1]
	br bool %1860, label %1861, label %1860

; <label>:1861		; preds = %1859, %1860
	add uint %1143, 4294967235		; <uint>:1187 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1187		; <ubyte*>:2783 [#uses=1]
	load ubyte* %2783		; <ubyte>:4398 [#uses=1]
	seteq ubyte %4398, 0		; <bool>:1861 [#uses=1]
	br bool %1861, label %1863, label %1862

; <label>:1862		; preds = %1861, %1862
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1187		; <ubyte*>:2784 [#uses=2]
	load ubyte* %2784		; <ubyte>:4399 [#uses=1]
	add ubyte %4399, 255		; <ubyte>:4400 [#uses=1]
	store ubyte %4400, ubyte* %2784
	add uint %1143, 4294967236		; <uint>:1188 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1188		; <ubyte*>:2785 [#uses=2]
	load ubyte* %2785		; <ubyte>:4401 [#uses=1]
	add ubyte %4401, 1		; <ubyte>:4402 [#uses=1]
	store ubyte %4402, ubyte* %2785
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1154		; <ubyte*>:2786 [#uses=2]
	load ubyte* %2786		; <ubyte>:4403 [#uses=1]
	add ubyte %4403, 1		; <ubyte>:4404 [#uses=1]
	store ubyte %4404, ubyte* %2786
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1187		; <ubyte*>:2787 [#uses=1]
	load ubyte* %2787		; <ubyte>:4405 [#uses=1]
	seteq ubyte %4405, 0		; <bool>:1862 [#uses=1]
	br bool %1862, label %1863, label %1862

; <label>:1863		; preds = %1861, %1862
	add uint %1143, 4294967236		; <uint>:1189 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1189		; <ubyte*>:2788 [#uses=1]
	load ubyte* %2788		; <ubyte>:4406 [#uses=1]
	seteq ubyte %4406, 0		; <bool>:1863 [#uses=1]
	br bool %1863, label %1865, label %1864

; <label>:1864		; preds = %1863, %1864
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1187		; <ubyte*>:2789 [#uses=2]
	load ubyte* %2789		; <ubyte>:4407 [#uses=1]
	add ubyte %4407, 1		; <ubyte>:4408 [#uses=1]
	store ubyte %4408, ubyte* %2789
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1189		; <ubyte*>:2790 [#uses=2]
	load ubyte* %2790		; <ubyte>:4409 [#uses=2]
	add ubyte %4409, 255		; <ubyte>:4410 [#uses=1]
	store ubyte %4410, ubyte* %2790
	seteq ubyte %4409, 1		; <bool>:1864 [#uses=1]
	br bool %1864, label %1865, label %1864

; <label>:1865		; preds = %1863, %1864
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1155		; <ubyte*>:2791 [#uses=1]
	load ubyte* %2791		; <ubyte>:4411 [#uses=1]
	seteq ubyte %4411, 0		; <bool>:1865 [#uses=1]
	br bool %1865, label %1867, label %1866

; <label>:1866		; preds = %1865, %1866
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1155		; <ubyte*>:2792 [#uses=2]
	load ubyte* %2792		; <ubyte>:4412 [#uses=2]
	add ubyte %4412, 255		; <ubyte>:4413 [#uses=1]
	store ubyte %4413, ubyte* %2792
	seteq ubyte %4412, 1		; <bool>:1866 [#uses=1]
	br bool %1866, label %1867, label %1866

; <label>:1867		; preds = %1865, %1866
	add uint %1143, 4294967241		; <uint>:1190 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1190		; <ubyte*>:2793 [#uses=1]
	load ubyte* %2793		; <ubyte>:4414 [#uses=1]
	seteq ubyte %4414, 0		; <bool>:1867 [#uses=1]
	br bool %1867, label %1869, label %1868

; <label>:1868		; preds = %1867, %1868
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1190		; <ubyte*>:2794 [#uses=2]
	load ubyte* %2794		; <ubyte>:4415 [#uses=1]
	add ubyte %4415, 255		; <ubyte>:4416 [#uses=1]
	store ubyte %4416, ubyte* %2794
	add uint %1143, 4294967242		; <uint>:1191 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1191		; <ubyte*>:2795 [#uses=2]
	load ubyte* %2795		; <ubyte>:4417 [#uses=1]
	add ubyte %4417, 1		; <ubyte>:4418 [#uses=1]
	store ubyte %4418, ubyte* %2795
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1155		; <ubyte*>:2796 [#uses=2]
	load ubyte* %2796		; <ubyte>:4419 [#uses=1]
	add ubyte %4419, 1		; <ubyte>:4420 [#uses=1]
	store ubyte %4420, ubyte* %2796
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1190		; <ubyte*>:2797 [#uses=1]
	load ubyte* %2797		; <ubyte>:4421 [#uses=1]
	seteq ubyte %4421, 0		; <bool>:1868 [#uses=1]
	br bool %1868, label %1869, label %1868

; <label>:1869		; preds = %1867, %1868
	add uint %1143, 4294967242		; <uint>:1192 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1192		; <ubyte*>:2798 [#uses=1]
	load ubyte* %2798		; <ubyte>:4422 [#uses=1]
	seteq ubyte %4422, 0		; <bool>:1869 [#uses=1]
	br bool %1869, label %1871, label %1870

; <label>:1870		; preds = %1869, %1870
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1190		; <ubyte*>:2799 [#uses=2]
	load ubyte* %2799		; <ubyte>:4423 [#uses=1]
	add ubyte %4423, 1		; <ubyte>:4424 [#uses=1]
	store ubyte %4424, ubyte* %2799
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1192		; <ubyte*>:2800 [#uses=2]
	load ubyte* %2800		; <ubyte>:4425 [#uses=2]
	add ubyte %4425, 255		; <ubyte>:4426 [#uses=1]
	store ubyte %4426, ubyte* %2800
	seteq ubyte %4425, 1		; <bool>:1870 [#uses=1]
	br bool %1870, label %1871, label %1870

; <label>:1871		; preds = %1869, %1870
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1156		; <ubyte*>:2801 [#uses=1]
	load ubyte* %2801		; <ubyte>:4427 [#uses=1]
	seteq ubyte %4427, 0		; <bool>:1871 [#uses=1]
	br bool %1871, label %1873, label %1872

; <label>:1872		; preds = %1871, %1872
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1156		; <ubyte*>:2802 [#uses=2]
	load ubyte* %2802		; <ubyte>:4428 [#uses=2]
	add ubyte %4428, 255		; <ubyte>:4429 [#uses=1]
	store ubyte %4429, ubyte* %2802
	seteq ubyte %4428, 1		; <bool>:1872 [#uses=1]
	br bool %1872, label %1873, label %1872

; <label>:1873		; preds = %1871, %1872
	add uint %1143, 4294967247		; <uint>:1193 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1193		; <ubyte*>:2803 [#uses=1]
	load ubyte* %2803		; <ubyte>:4430 [#uses=1]
	seteq ubyte %4430, 0		; <bool>:1873 [#uses=1]
	br bool %1873, label %1875, label %1874

; <label>:1874		; preds = %1873, %1874
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1193		; <ubyte*>:2804 [#uses=2]
	load ubyte* %2804		; <ubyte>:4431 [#uses=1]
	add ubyte %4431, 255		; <ubyte>:4432 [#uses=1]
	store ubyte %4432, ubyte* %2804
	add uint %1143, 4294967248		; <uint>:1194 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1194		; <ubyte*>:2805 [#uses=2]
	load ubyte* %2805		; <ubyte>:4433 [#uses=1]
	add ubyte %4433, 1		; <ubyte>:4434 [#uses=1]
	store ubyte %4434, ubyte* %2805
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1156		; <ubyte*>:2806 [#uses=2]
	load ubyte* %2806		; <ubyte>:4435 [#uses=1]
	add ubyte %4435, 1		; <ubyte>:4436 [#uses=1]
	store ubyte %4436, ubyte* %2806
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1193		; <ubyte*>:2807 [#uses=1]
	load ubyte* %2807		; <ubyte>:4437 [#uses=1]
	seteq ubyte %4437, 0		; <bool>:1874 [#uses=1]
	br bool %1874, label %1875, label %1874

; <label>:1875		; preds = %1873, %1874
	add uint %1143, 4294967248		; <uint>:1195 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1195		; <ubyte*>:2808 [#uses=1]
	load ubyte* %2808		; <ubyte>:4438 [#uses=1]
	seteq ubyte %4438, 0		; <bool>:1875 [#uses=1]
	br bool %1875, label %1877, label %1876

; <label>:1876		; preds = %1875, %1876
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1193		; <ubyte*>:2809 [#uses=2]
	load ubyte* %2809		; <ubyte>:4439 [#uses=1]
	add ubyte %4439, 1		; <ubyte>:4440 [#uses=1]
	store ubyte %4440, ubyte* %2809
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1195		; <ubyte*>:2810 [#uses=2]
	load ubyte* %2810		; <ubyte>:4441 [#uses=2]
	add ubyte %4441, 255		; <ubyte>:4442 [#uses=1]
	store ubyte %4442, ubyte* %2810
	seteq ubyte %4441, 1		; <bool>:1876 [#uses=1]
	br bool %1876, label %1877, label %1876

; <label>:1877		; preds = %1875, %1876
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1157		; <ubyte*>:2811 [#uses=1]
	load ubyte* %2811		; <ubyte>:4443 [#uses=1]
	seteq ubyte %4443, 0		; <bool>:1877 [#uses=1]
	br bool %1877, label %1879, label %1878

; <label>:1878		; preds = %1877, %1878
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1157		; <ubyte*>:2812 [#uses=2]
	load ubyte* %2812		; <ubyte>:4444 [#uses=2]
	add ubyte %4444, 255		; <ubyte>:4445 [#uses=1]
	store ubyte %4445, ubyte* %2812
	seteq ubyte %4444, 1		; <bool>:1878 [#uses=1]
	br bool %1878, label %1879, label %1878

; <label>:1879		; preds = %1877, %1878
	add uint %1143, 4294967253		; <uint>:1196 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1196		; <ubyte*>:2813 [#uses=1]
	load ubyte* %2813		; <ubyte>:4446 [#uses=1]
	seteq ubyte %4446, 0		; <bool>:1879 [#uses=1]
	br bool %1879, label %1881, label %1880

; <label>:1880		; preds = %1879, %1880
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1196		; <ubyte*>:2814 [#uses=2]
	load ubyte* %2814		; <ubyte>:4447 [#uses=1]
	add ubyte %4447, 255		; <ubyte>:4448 [#uses=1]
	store ubyte %4448, ubyte* %2814
	add uint %1143, 4294967254		; <uint>:1197 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1197		; <ubyte*>:2815 [#uses=2]
	load ubyte* %2815		; <ubyte>:4449 [#uses=1]
	add ubyte %4449, 1		; <ubyte>:4450 [#uses=1]
	store ubyte %4450, ubyte* %2815
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1157		; <ubyte*>:2816 [#uses=2]
	load ubyte* %2816		; <ubyte>:4451 [#uses=1]
	add ubyte %4451, 1		; <ubyte>:4452 [#uses=1]
	store ubyte %4452, ubyte* %2816
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1196		; <ubyte*>:2817 [#uses=1]
	load ubyte* %2817		; <ubyte>:4453 [#uses=1]
	seteq ubyte %4453, 0		; <bool>:1880 [#uses=1]
	br bool %1880, label %1881, label %1880

; <label>:1881		; preds = %1879, %1880
	add uint %1143, 4294967254		; <uint>:1198 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1198		; <ubyte*>:2818 [#uses=1]
	load ubyte* %2818		; <ubyte>:4454 [#uses=1]
	seteq ubyte %4454, 0		; <bool>:1881 [#uses=1]
	br bool %1881, label %1883, label %1882

; <label>:1882		; preds = %1881, %1882
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1196		; <ubyte*>:2819 [#uses=2]
	load ubyte* %2819		; <ubyte>:4455 [#uses=1]
	add ubyte %4455, 1		; <ubyte>:4456 [#uses=1]
	store ubyte %4456, ubyte* %2819
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1198		; <ubyte*>:2820 [#uses=2]
	load ubyte* %2820		; <ubyte>:4457 [#uses=2]
	add ubyte %4457, 255		; <ubyte>:4458 [#uses=1]
	store ubyte %4458, ubyte* %2820
	seteq ubyte %4457, 1		; <bool>:1882 [#uses=1]
	br bool %1882, label %1883, label %1882

; <label>:1883		; preds = %1881, %1882
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1158		; <ubyte*>:2821 [#uses=1]
	load ubyte* %2821		; <ubyte>:4459 [#uses=1]
	seteq ubyte %4459, 0		; <bool>:1883 [#uses=1]
	br bool %1883, label %1885, label %1884

; <label>:1884		; preds = %1883, %1884
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1158		; <ubyte*>:2822 [#uses=2]
	load ubyte* %2822		; <ubyte>:4460 [#uses=2]
	add ubyte %4460, 255		; <ubyte>:4461 [#uses=1]
	store ubyte %4461, ubyte* %2822
	seteq ubyte %4460, 1		; <bool>:1884 [#uses=1]
	br bool %1884, label %1885, label %1884

; <label>:1885		; preds = %1883, %1884
	add uint %1143, 4294967259		; <uint>:1199 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1199		; <ubyte*>:2823 [#uses=1]
	load ubyte* %2823		; <ubyte>:4462 [#uses=1]
	seteq ubyte %4462, 0		; <bool>:1885 [#uses=1]
	br bool %1885, label %1887, label %1886

; <label>:1886		; preds = %1885, %1886
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1199		; <ubyte*>:2824 [#uses=2]
	load ubyte* %2824		; <ubyte>:4463 [#uses=1]
	add ubyte %4463, 255		; <ubyte>:4464 [#uses=1]
	store ubyte %4464, ubyte* %2824
	add uint %1143, 4294967260		; <uint>:1200 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1200		; <ubyte*>:2825 [#uses=2]
	load ubyte* %2825		; <ubyte>:4465 [#uses=1]
	add ubyte %4465, 1		; <ubyte>:4466 [#uses=1]
	store ubyte %4466, ubyte* %2825
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1158		; <ubyte*>:2826 [#uses=2]
	load ubyte* %2826		; <ubyte>:4467 [#uses=1]
	add ubyte %4467, 1		; <ubyte>:4468 [#uses=1]
	store ubyte %4468, ubyte* %2826
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1199		; <ubyte*>:2827 [#uses=1]
	load ubyte* %2827		; <ubyte>:4469 [#uses=1]
	seteq ubyte %4469, 0		; <bool>:1886 [#uses=1]
	br bool %1886, label %1887, label %1886

; <label>:1887		; preds = %1885, %1886
	add uint %1143, 4294967260		; <uint>:1201 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1201		; <ubyte*>:2828 [#uses=1]
	load ubyte* %2828		; <ubyte>:4470 [#uses=1]
	seteq ubyte %4470, 0		; <bool>:1887 [#uses=1]
	br bool %1887, label %1889, label %1888

; <label>:1888		; preds = %1887, %1888
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1199		; <ubyte*>:2829 [#uses=2]
	load ubyte* %2829		; <ubyte>:4471 [#uses=1]
	add ubyte %4471, 1		; <ubyte>:4472 [#uses=1]
	store ubyte %4472, ubyte* %2829
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1201		; <ubyte*>:2830 [#uses=2]
	load ubyte* %2830		; <ubyte>:4473 [#uses=2]
	add ubyte %4473, 255		; <ubyte>:4474 [#uses=1]
	store ubyte %4474, ubyte* %2830
	seteq ubyte %4473, 1		; <bool>:1888 [#uses=1]
	br bool %1888, label %1889, label %1888

; <label>:1889		; preds = %1887, %1888
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1159		; <ubyte*>:2831 [#uses=1]
	load ubyte* %2831		; <ubyte>:4475 [#uses=1]
	seteq ubyte %4475, 0		; <bool>:1889 [#uses=1]
	br bool %1889, label %1891, label %1890

; <label>:1890		; preds = %1889, %1890
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1159		; <ubyte*>:2832 [#uses=2]
	load ubyte* %2832		; <ubyte>:4476 [#uses=2]
	add ubyte %4476, 255		; <ubyte>:4477 [#uses=1]
	store ubyte %4477, ubyte* %2832
	seteq ubyte %4476, 1		; <bool>:1890 [#uses=1]
	br bool %1890, label %1891, label %1890

; <label>:1891		; preds = %1889, %1890
	add uint %1143, 4294967265		; <uint>:1202 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1202		; <ubyte*>:2833 [#uses=1]
	load ubyte* %2833		; <ubyte>:4478 [#uses=1]
	seteq ubyte %4478, 0		; <bool>:1891 [#uses=1]
	br bool %1891, label %1893, label %1892

; <label>:1892		; preds = %1891, %1892
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1202		; <ubyte*>:2834 [#uses=2]
	load ubyte* %2834		; <ubyte>:4479 [#uses=1]
	add ubyte %4479, 255		; <ubyte>:4480 [#uses=1]
	store ubyte %4480, ubyte* %2834
	add uint %1143, 4294967266		; <uint>:1203 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1203		; <ubyte*>:2835 [#uses=2]
	load ubyte* %2835		; <ubyte>:4481 [#uses=1]
	add ubyte %4481, 1		; <ubyte>:4482 [#uses=1]
	store ubyte %4482, ubyte* %2835
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1159		; <ubyte*>:2836 [#uses=2]
	load ubyte* %2836		; <ubyte>:4483 [#uses=1]
	add ubyte %4483, 1		; <ubyte>:4484 [#uses=1]
	store ubyte %4484, ubyte* %2836
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1202		; <ubyte*>:2837 [#uses=1]
	load ubyte* %2837		; <ubyte>:4485 [#uses=1]
	seteq ubyte %4485, 0		; <bool>:1892 [#uses=1]
	br bool %1892, label %1893, label %1892

; <label>:1893		; preds = %1891, %1892
	add uint %1143, 4294967266		; <uint>:1204 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1204		; <ubyte*>:2838 [#uses=1]
	load ubyte* %2838		; <ubyte>:4486 [#uses=1]
	seteq ubyte %4486, 0		; <bool>:1893 [#uses=1]
	br bool %1893, label %1895, label %1894

; <label>:1894		; preds = %1893, %1894
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1202		; <ubyte*>:2839 [#uses=2]
	load ubyte* %2839		; <ubyte>:4487 [#uses=1]
	add ubyte %4487, 1		; <ubyte>:4488 [#uses=1]
	store ubyte %4488, ubyte* %2839
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1204		; <ubyte*>:2840 [#uses=2]
	load ubyte* %2840		; <ubyte>:4489 [#uses=2]
	add ubyte %4489, 255		; <ubyte>:4490 [#uses=1]
	store ubyte %4490, ubyte* %2840
	seteq ubyte %4489, 1		; <bool>:1894 [#uses=1]
	br bool %1894, label %1895, label %1894

; <label>:1895		; preds = %1893, %1894
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1160		; <ubyte*>:2841 [#uses=1]
	load ubyte* %2841		; <ubyte>:4491 [#uses=1]
	seteq ubyte %4491, 0		; <bool>:1895 [#uses=1]
	br bool %1895, label %1897, label %1896

; <label>:1896		; preds = %1895, %1896
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1160		; <ubyte*>:2842 [#uses=2]
	load ubyte* %2842		; <ubyte>:4492 [#uses=2]
	add ubyte %4492, 255		; <ubyte>:4493 [#uses=1]
	store ubyte %4493, ubyte* %2842
	seteq ubyte %4492, 1		; <bool>:1896 [#uses=1]
	br bool %1896, label %1897, label %1896

; <label>:1897		; preds = %1895, %1896
	add uint %1143, 4294967271		; <uint>:1205 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1205		; <ubyte*>:2843 [#uses=1]
	load ubyte* %2843		; <ubyte>:4494 [#uses=1]
	seteq ubyte %4494, 0		; <bool>:1897 [#uses=1]
	br bool %1897, label %1899, label %1898

; <label>:1898		; preds = %1897, %1898
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1205		; <ubyte*>:2844 [#uses=2]
	load ubyte* %2844		; <ubyte>:4495 [#uses=1]
	add ubyte %4495, 255		; <ubyte>:4496 [#uses=1]
	store ubyte %4496, ubyte* %2844
	add uint %1143, 4294967272		; <uint>:1206 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1206		; <ubyte*>:2845 [#uses=2]
	load ubyte* %2845		; <ubyte>:4497 [#uses=1]
	add ubyte %4497, 1		; <ubyte>:4498 [#uses=1]
	store ubyte %4498, ubyte* %2845
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1160		; <ubyte*>:2846 [#uses=2]
	load ubyte* %2846		; <ubyte>:4499 [#uses=1]
	add ubyte %4499, 1		; <ubyte>:4500 [#uses=1]
	store ubyte %4500, ubyte* %2846
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1205		; <ubyte*>:2847 [#uses=1]
	load ubyte* %2847		; <ubyte>:4501 [#uses=1]
	seteq ubyte %4501, 0		; <bool>:1898 [#uses=1]
	br bool %1898, label %1899, label %1898

; <label>:1899		; preds = %1897, %1898
	add uint %1143, 4294967272		; <uint>:1207 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1207		; <ubyte*>:2848 [#uses=1]
	load ubyte* %2848		; <ubyte>:4502 [#uses=1]
	seteq ubyte %4502, 0		; <bool>:1899 [#uses=1]
	br bool %1899, label %1901, label %1900

; <label>:1900		; preds = %1899, %1900
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1205		; <ubyte*>:2849 [#uses=2]
	load ubyte* %2849		; <ubyte>:4503 [#uses=1]
	add ubyte %4503, 1		; <ubyte>:4504 [#uses=1]
	store ubyte %4504, ubyte* %2849
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1207		; <ubyte*>:2850 [#uses=2]
	load ubyte* %2850		; <ubyte>:4505 [#uses=2]
	add ubyte %4505, 255		; <ubyte>:4506 [#uses=1]
	store ubyte %4506, ubyte* %2850
	seteq ubyte %4505, 1		; <bool>:1900 [#uses=1]
	br bool %1900, label %1901, label %1900

; <label>:1901		; preds = %1899, %1900
	add uint %1143, 92		; <uint>:1208 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1208		; <ubyte*>:2851 [#uses=1]
	load ubyte* %2851		; <ubyte>:4507 [#uses=1]
	seteq ubyte %4507, 0		; <bool>:1901 [#uses=1]
	br bool %1901, label %1903, label %1902

; <label>:1902		; preds = %1901, %1902
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1208		; <ubyte*>:2852 [#uses=2]
	load ubyte* %2852		; <ubyte>:4508 [#uses=2]
	add ubyte %4508, 255		; <ubyte>:4509 [#uses=1]
	store ubyte %4509, ubyte* %2852
	seteq ubyte %4508, 1		; <bool>:1902 [#uses=1]
	br bool %1902, label %1903, label %1902

; <label>:1903		; preds = %1901, %1902
	add uint %1143, 4294967286		; <uint>:1209 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1209		; <ubyte*>:2853 [#uses=1]
	load ubyte* %2853		; <ubyte>:4510 [#uses=1]
	seteq ubyte %4510, 0		; <bool>:1903 [#uses=1]
	br bool %1903, label %1905, label %1904

; <label>:1904		; preds = %1903, %1904
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1209		; <ubyte*>:2854 [#uses=2]
	load ubyte* %2854		; <ubyte>:4511 [#uses=1]
	add ubyte %4511, 255		; <ubyte>:4512 [#uses=1]
	store ubyte %4512, ubyte* %2854
	add uint %1143, 4294967287		; <uint>:1210 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1210		; <ubyte*>:2855 [#uses=2]
	load ubyte* %2855		; <ubyte>:4513 [#uses=1]
	add ubyte %4513, 1		; <ubyte>:4514 [#uses=1]
	store ubyte %4514, ubyte* %2855
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1208		; <ubyte*>:2856 [#uses=2]
	load ubyte* %2856		; <ubyte>:4515 [#uses=1]
	add ubyte %4515, 1		; <ubyte>:4516 [#uses=1]
	store ubyte %4516, ubyte* %2856
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1209		; <ubyte*>:2857 [#uses=1]
	load ubyte* %2857		; <ubyte>:4517 [#uses=1]
	seteq ubyte %4517, 0		; <bool>:1904 [#uses=1]
	br bool %1904, label %1905, label %1904

; <label>:1905		; preds = %1903, %1904
	add uint %1143, 4294967287		; <uint>:1211 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1211		; <ubyte*>:2858 [#uses=1]
	load ubyte* %2858		; <ubyte>:4518 [#uses=1]
	seteq ubyte %4518, 0		; <bool>:1905 [#uses=1]
	br bool %1905, label %1907, label %1906

; <label>:1906		; preds = %1905, %1906
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1209		; <ubyte*>:2859 [#uses=2]
	load ubyte* %2859		; <ubyte>:4519 [#uses=1]
	add ubyte %4519, 1		; <ubyte>:4520 [#uses=1]
	store ubyte %4520, ubyte* %2859
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1211		; <ubyte*>:2860 [#uses=2]
	load ubyte* %2860		; <ubyte>:4521 [#uses=2]
	add ubyte %4521, 255		; <ubyte>:4522 [#uses=1]
	store ubyte %4522, ubyte* %2860
	seteq ubyte %4521, 1		; <bool>:1906 [#uses=1]
	br bool %1906, label %1907, label %1906

; <label>:1907		; preds = %1905, %1906
	add uint %1143, 6		; <uint>:1212 [#uses=6]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1212		; <ubyte*>:2861 [#uses=1]
	load ubyte* %2861		; <ubyte>:4523 [#uses=1]
	seteq ubyte %4523, 0		; <bool>:1907 [#uses=1]
	br bool %1907, label %1909, label %1908

; <label>:1908		; preds = %1907, %1908
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1212		; <ubyte*>:2862 [#uses=2]
	load ubyte* %2862		; <ubyte>:4524 [#uses=2]
	add ubyte %4524, 255		; <ubyte>:4525 [#uses=1]
	store ubyte %4525, ubyte* %2862
	seteq ubyte %4524, 1		; <bool>:1908 [#uses=1]
	br bool %1908, label %1909, label %1908

; <label>:1909		; preds = %1907, %1908
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1208		; <ubyte*>:2863 [#uses=1]
	load ubyte* %2863		; <ubyte>:4526 [#uses=1]
	seteq ubyte %4526, 0		; <bool>:1909 [#uses=1]
	br bool %1909, label %1911, label %1910

; <label>:1910		; preds = %1909, %1910
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1212		; <ubyte*>:2864 [#uses=2]
	load ubyte* %2864		; <ubyte>:4527 [#uses=1]
	add ubyte %4527, 1		; <ubyte>:4528 [#uses=1]
	store ubyte %4528, ubyte* %2864
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1208		; <ubyte*>:2865 [#uses=2]
	load ubyte* %2865		; <ubyte>:4529 [#uses=2]
	add ubyte %4529, 255		; <ubyte>:4530 [#uses=1]
	store ubyte %4530, ubyte* %2865
	seteq ubyte %4529, 1		; <bool>:1910 [#uses=1]
	br bool %1910, label %1911, label %1910

; <label>:1911		; preds = %1909, %1910
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1212		; <ubyte*>:2866 [#uses=1]
	load ubyte* %2866		; <ubyte>:4531 [#uses=1]
	seteq ubyte %4531, 0		; <bool>:1911 [#uses=1]
	br bool %1911, label %1913, label %1912

; <label>:1912		; preds = %1911, %1915
	phi uint [ %1212, %1911 ], [ %1217, %1915 ]		; <uint>:1213 [#uses=6]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1213		; <ubyte*>:2867 [#uses=1]
	load ubyte* %2867		; <ubyte>:4532 [#uses=1]
	seteq ubyte %4532, 0		; <bool>:1912 [#uses=1]
	br bool %1912, label %1915, label %1914

; <label>:1913		; preds = %1911, %1915
	phi uint [ %1212, %1911 ], [ %1217, %1915 ]		; <uint>:1214 [#uses=7]
	add uint %1214, 4294967292		; <uint>:1215 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1215		; <ubyte*>:2868 [#uses=1]
	load ubyte* %2868		; <ubyte>:4533 [#uses=1]
	seteq ubyte %4533, 0		; <bool>:1913 [#uses=1]
	br bool %1913, label %1917, label %1916

; <label>:1914		; preds = %1912, %1914
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1213		; <ubyte*>:2869 [#uses=2]
	load ubyte* %2869		; <ubyte>:4534 [#uses=1]
	add ubyte %4534, 255		; <ubyte>:4535 [#uses=1]
	store ubyte %4535, ubyte* %2869
	add uint %1213, 6		; <uint>:1216 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1216		; <ubyte*>:2870 [#uses=2]
	load ubyte* %2870		; <ubyte>:4536 [#uses=1]
	add ubyte %4536, 1		; <ubyte>:4537 [#uses=1]
	store ubyte %4537, ubyte* %2870
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1213		; <ubyte*>:2871 [#uses=1]
	load ubyte* %2871		; <ubyte>:4538 [#uses=1]
	seteq ubyte %4538, 0		; <bool>:1914 [#uses=1]
	br bool %1914, label %1915, label %1914

; <label>:1915		; preds = %1912, %1914
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1213		; <ubyte*>:2872 [#uses=2]
	load ubyte* %2872		; <ubyte>:4539 [#uses=1]
	add ubyte %4539, 1		; <ubyte>:4540 [#uses=1]
	store ubyte %4540, ubyte* %2872
	add uint %1213, 6		; <uint>:1217 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1217		; <ubyte*>:2873 [#uses=2]
	load ubyte* %2873		; <ubyte>:4541 [#uses=2]
	add ubyte %4541, 255		; <ubyte>:4542 [#uses=1]
	store ubyte %4542, ubyte* %2873
	seteq ubyte %4541, 1		; <bool>:1915 [#uses=1]
	br bool %1915, label %1913, label %1912

; <label>:1916		; preds = %1913, %1916
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1215		; <ubyte*>:2874 [#uses=2]
	load ubyte* %2874		; <ubyte>:4543 [#uses=2]
	add ubyte %4543, 255		; <ubyte>:4544 [#uses=1]
	store ubyte %4544, ubyte* %2874
	seteq ubyte %4543, 1		; <bool>:1916 [#uses=1]
	br bool %1916, label %1917, label %1916

; <label>:1917		; preds = %1913, %1916
	add uint %1214, 4294967294		; <uint>:1218 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1218		; <ubyte*>:2875 [#uses=1]
	load ubyte* %2875		; <ubyte>:4545 [#uses=1]
	seteq ubyte %4545, 0		; <bool>:1917 [#uses=1]
	br bool %1917, label %1919, label %1918

; <label>:1918		; preds = %1917, %1918
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1215		; <ubyte*>:2876 [#uses=2]
	load ubyte* %2876		; <ubyte>:4546 [#uses=1]
	add ubyte %4546, 1		; <ubyte>:4547 [#uses=1]
	store ubyte %4547, ubyte* %2876
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1218		; <ubyte*>:2877 [#uses=2]
	load ubyte* %2877		; <ubyte>:4548 [#uses=1]
	add ubyte %4548, 255		; <ubyte>:4549 [#uses=1]
	store ubyte %4549, ubyte* %2877
	add uint %1214, 4294967295		; <uint>:1219 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1219		; <ubyte*>:2878 [#uses=2]
	load ubyte* %2878		; <ubyte>:4550 [#uses=1]
	add ubyte %4550, 1		; <ubyte>:4551 [#uses=1]
	store ubyte %4551, ubyte* %2878
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1218		; <ubyte*>:2879 [#uses=1]
	load ubyte* %2879		; <ubyte>:4552 [#uses=1]
	seteq ubyte %4552, 0		; <bool>:1918 [#uses=1]
	br bool %1918, label %1919, label %1918

; <label>:1919		; preds = %1917, %1918
	add uint %1214, 4294967295		; <uint>:1220 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1220		; <ubyte*>:2880 [#uses=1]
	load ubyte* %2880		; <ubyte>:4553 [#uses=1]
	seteq ubyte %4553, 0		; <bool>:1919 [#uses=1]
	br bool %1919, label %1921, label %1920

; <label>:1920		; preds = %1919, %1920
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1218		; <ubyte*>:2881 [#uses=2]
	load ubyte* %2881		; <ubyte>:4554 [#uses=1]
	add ubyte %4554, 1		; <ubyte>:4555 [#uses=1]
	store ubyte %4555, ubyte* %2881
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1220		; <ubyte*>:2882 [#uses=2]
	load ubyte* %2882		; <ubyte>:4556 [#uses=2]
	add ubyte %4556, 255		; <ubyte>:4557 [#uses=1]
	store ubyte %4557, ubyte* %2882
	seteq ubyte %4556, 1		; <bool>:1920 [#uses=1]
	br bool %1920, label %1921, label %1920

; <label>:1921		; preds = %1919, %1920
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1214		; <ubyte*>:2883 [#uses=2]
	load ubyte* %2883		; <ubyte>:4558 [#uses=2]
	add ubyte %4558, 1		; <ubyte>:4559 [#uses=1]
	store ubyte %4559, ubyte* %2883
	seteq ubyte %4558, 255		; <bool>:1921 [#uses=1]
	br bool %1921, label %1923, label %1922

; <label>:1922		; preds = %1921, %1927
	phi uint [ %1214, %1921 ], [ %1226, %1927 ]		; <uint>:1221 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1221		; <ubyte*>:2884 [#uses=2]
	load ubyte* %2884		; <ubyte>:4560 [#uses=1]
	add ubyte %4560, 255		; <ubyte>:4561 [#uses=1]
	store ubyte %4561, ubyte* %2884
	add uint %1221, 4294967286		; <uint>:1222 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1222		; <ubyte*>:2885 [#uses=1]
	load ubyte* %2885		; <ubyte>:4562 [#uses=1]
	seteq ubyte %4562, 0		; <bool>:1922 [#uses=1]
	br bool %1922, label %1925, label %1924

; <label>:1923		; preds = %1921, %1927
	phi uint [ %1214, %1921 ], [ %1226, %1927 ]		; <uint>:1223 [#uses=22]
	add uint %1223, 4		; <uint>:1224 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1224		; <ubyte*>:2886 [#uses=1]
	load ubyte* %2886		; <ubyte>:4563 [#uses=1]
	seteq ubyte %4563, 0		; <bool>:1923 [#uses=1]
	br bool %1923, label %1929, label %1928

; <label>:1924		; preds = %1922, %1924
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1222		; <ubyte*>:2887 [#uses=2]
	load ubyte* %2887		; <ubyte>:4564 [#uses=2]
	add ubyte %4564, 255		; <ubyte>:4565 [#uses=1]
	store ubyte %4565, ubyte* %2887
	seteq ubyte %4564, 1		; <bool>:1924 [#uses=1]
	br bool %1924, label %1925, label %1924

; <label>:1925		; preds = %1922, %1924
	add uint %1221, 4294967292		; <uint>:1225 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1225		; <ubyte*>:2888 [#uses=1]
	load ubyte* %2888		; <ubyte>:4566 [#uses=1]
	seteq ubyte %4566, 0		; <bool>:1925 [#uses=1]
	br bool %1925, label %1927, label %1926

; <label>:1926		; preds = %1925, %1926
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1222		; <ubyte*>:2889 [#uses=2]
	load ubyte* %2889		; <ubyte>:4567 [#uses=1]
	add ubyte %4567, 1		; <ubyte>:4568 [#uses=1]
	store ubyte %4568, ubyte* %2889
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1225		; <ubyte*>:2890 [#uses=2]
	load ubyte* %2890		; <ubyte>:4569 [#uses=2]
	add ubyte %4569, 255		; <ubyte>:4570 [#uses=1]
	store ubyte %4570, ubyte* %2890
	seteq ubyte %4569, 1		; <bool>:1926 [#uses=1]
	br bool %1926, label %1927, label %1926

; <label>:1927		; preds = %1925, %1926
	add uint %1221, 4294967290		; <uint>:1226 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1226		; <ubyte*>:2891 [#uses=1]
	load ubyte* %2891		; <ubyte>:4571 [#uses=1]
	seteq ubyte %4571, 0		; <bool>:1927 [#uses=1]
	br bool %1927, label %1923, label %1922

; <label>:1928		; preds = %1923, %1928
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1224		; <ubyte*>:2892 [#uses=2]
	load ubyte* %2892		; <ubyte>:4572 [#uses=2]
	add ubyte %4572, 255		; <ubyte>:4573 [#uses=1]
	store ubyte %4573, ubyte* %2892
	seteq ubyte %4572, 1		; <bool>:1928 [#uses=1]
	br bool %1928, label %1929, label %1928

; <label>:1929		; preds = %1923, %1928
	add uint %1223, 10		; <uint>:1227 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1227		; <ubyte*>:2893 [#uses=1]
	load ubyte* %2893		; <ubyte>:4574 [#uses=1]
	seteq ubyte %4574, 0		; <bool>:1929 [#uses=1]
	br bool %1929, label %1931, label %1930

; <label>:1930		; preds = %1929, %1930
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1227		; <ubyte*>:2894 [#uses=2]
	load ubyte* %2894		; <ubyte>:4575 [#uses=2]
	add ubyte %4575, 255		; <ubyte>:4576 [#uses=1]
	store ubyte %4576, ubyte* %2894
	seteq ubyte %4575, 1		; <bool>:1930 [#uses=1]
	br bool %1930, label %1931, label %1930

; <label>:1931		; preds = %1929, %1930
	add uint %1223, 16		; <uint>:1228 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1228		; <ubyte*>:2895 [#uses=1]
	load ubyte* %2895		; <ubyte>:4577 [#uses=1]
	seteq ubyte %4577, 0		; <bool>:1931 [#uses=1]
	br bool %1931, label %1933, label %1932

; <label>:1932		; preds = %1931, %1932
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1228		; <ubyte*>:2896 [#uses=2]
	load ubyte* %2896		; <ubyte>:4578 [#uses=2]
	add ubyte %4578, 255		; <ubyte>:4579 [#uses=1]
	store ubyte %4579, ubyte* %2896
	seteq ubyte %4578, 1		; <bool>:1932 [#uses=1]
	br bool %1932, label %1933, label %1932

; <label>:1933		; preds = %1931, %1932
	add uint %1223, 22		; <uint>:1229 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1229		; <ubyte*>:2897 [#uses=1]
	load ubyte* %2897		; <ubyte>:4580 [#uses=1]
	seteq ubyte %4580, 0		; <bool>:1933 [#uses=1]
	br bool %1933, label %1935, label %1934

; <label>:1934		; preds = %1933, %1934
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1229		; <ubyte*>:2898 [#uses=2]
	load ubyte* %2898		; <ubyte>:4581 [#uses=2]
	add ubyte %4581, 255		; <ubyte>:4582 [#uses=1]
	store ubyte %4582, ubyte* %2898
	seteq ubyte %4581, 1		; <bool>:1934 [#uses=1]
	br bool %1934, label %1935, label %1934

; <label>:1935		; preds = %1933, %1934
	add uint %1223, 28		; <uint>:1230 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1230		; <ubyte*>:2899 [#uses=1]
	load ubyte* %2899		; <ubyte>:4583 [#uses=1]
	seteq ubyte %4583, 0		; <bool>:1935 [#uses=1]
	br bool %1935, label %1937, label %1936

; <label>:1936		; preds = %1935, %1936
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1230		; <ubyte*>:2900 [#uses=2]
	load ubyte* %2900		; <ubyte>:4584 [#uses=2]
	add ubyte %4584, 255		; <ubyte>:4585 [#uses=1]
	store ubyte %4585, ubyte* %2900
	seteq ubyte %4584, 1		; <bool>:1936 [#uses=1]
	br bool %1936, label %1937, label %1936

; <label>:1937		; preds = %1935, %1936
	add uint %1223, 34		; <uint>:1231 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1231		; <ubyte*>:2901 [#uses=1]
	load ubyte* %2901		; <ubyte>:4586 [#uses=1]
	seteq ubyte %4586, 0		; <bool>:1937 [#uses=1]
	br bool %1937, label %1939, label %1938

; <label>:1938		; preds = %1937, %1938
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1231		; <ubyte*>:2902 [#uses=2]
	load ubyte* %2902		; <ubyte>:4587 [#uses=2]
	add ubyte %4587, 255		; <ubyte>:4588 [#uses=1]
	store ubyte %4588, ubyte* %2902
	seteq ubyte %4587, 1		; <bool>:1938 [#uses=1]
	br bool %1938, label %1939, label %1938

; <label>:1939		; preds = %1937, %1938
	add uint %1223, 40		; <uint>:1232 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1232		; <ubyte*>:2903 [#uses=1]
	load ubyte* %2903		; <ubyte>:4589 [#uses=1]
	seteq ubyte %4589, 0		; <bool>:1939 [#uses=1]
	br bool %1939, label %1941, label %1940

; <label>:1940		; preds = %1939, %1940
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1232		; <ubyte*>:2904 [#uses=2]
	load ubyte* %2904		; <ubyte>:4590 [#uses=2]
	add ubyte %4590, 255		; <ubyte>:4591 [#uses=1]
	store ubyte %4591, ubyte* %2904
	seteq ubyte %4590, 1		; <bool>:1940 [#uses=1]
	br bool %1940, label %1941, label %1940

; <label>:1941		; preds = %1939, %1940
	add uint %1223, 46		; <uint>:1233 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1233		; <ubyte*>:2905 [#uses=1]
	load ubyte* %2905		; <ubyte>:4592 [#uses=1]
	seteq ubyte %4592, 0		; <bool>:1941 [#uses=1]
	br bool %1941, label %1943, label %1942

; <label>:1942		; preds = %1941, %1942
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1233		; <ubyte*>:2906 [#uses=2]
	load ubyte* %2906		; <ubyte>:4593 [#uses=2]
	add ubyte %4593, 255		; <ubyte>:4594 [#uses=1]
	store ubyte %4594, ubyte* %2906
	seteq ubyte %4593, 1		; <bool>:1942 [#uses=1]
	br bool %1942, label %1943, label %1942

; <label>:1943		; preds = %1941, %1942
	add uint %1223, 52		; <uint>:1234 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1234		; <ubyte*>:2907 [#uses=1]
	load ubyte* %2907		; <ubyte>:4595 [#uses=1]
	seteq ubyte %4595, 0		; <bool>:1943 [#uses=1]
	br bool %1943, label %1945, label %1944

; <label>:1944		; preds = %1943, %1944
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1234		; <ubyte*>:2908 [#uses=2]
	load ubyte* %2908		; <ubyte>:4596 [#uses=2]
	add ubyte %4596, 255		; <ubyte>:4597 [#uses=1]
	store ubyte %4597, ubyte* %2908
	seteq ubyte %4596, 1		; <bool>:1944 [#uses=1]
	br bool %1944, label %1945, label %1944

; <label>:1945		; preds = %1943, %1944
	add uint %1223, 58		; <uint>:1235 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1235		; <ubyte*>:2909 [#uses=1]
	load ubyte* %2909		; <ubyte>:4598 [#uses=1]
	seteq ubyte %4598, 0		; <bool>:1945 [#uses=1]
	br bool %1945, label %1947, label %1946

; <label>:1946		; preds = %1945, %1946
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1235		; <ubyte*>:2910 [#uses=2]
	load ubyte* %2910		; <ubyte>:4599 [#uses=2]
	add ubyte %4599, 255		; <ubyte>:4600 [#uses=1]
	store ubyte %4600, ubyte* %2910
	seteq ubyte %4599, 1		; <bool>:1946 [#uses=1]
	br bool %1946, label %1947, label %1946

; <label>:1947		; preds = %1945, %1946
	add uint %1223, 64		; <uint>:1236 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1236		; <ubyte*>:2911 [#uses=1]
	load ubyte* %2911		; <ubyte>:4601 [#uses=1]
	seteq ubyte %4601, 0		; <bool>:1947 [#uses=1]
	br bool %1947, label %1949, label %1948

; <label>:1948		; preds = %1947, %1948
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1236		; <ubyte*>:2912 [#uses=2]
	load ubyte* %2912		; <ubyte>:4602 [#uses=2]
	add ubyte %4602, 255		; <ubyte>:4603 [#uses=1]
	store ubyte %4603, ubyte* %2912
	seteq ubyte %4602, 1		; <bool>:1948 [#uses=1]
	br bool %1948, label %1949, label %1948

; <label>:1949		; preds = %1947, %1948
	add uint %1223, 70		; <uint>:1237 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1237		; <ubyte*>:2913 [#uses=1]
	load ubyte* %2913		; <ubyte>:4604 [#uses=1]
	seteq ubyte %4604, 0		; <bool>:1949 [#uses=1]
	br bool %1949, label %1951, label %1950

; <label>:1950		; preds = %1949, %1950
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1237		; <ubyte*>:2914 [#uses=2]
	load ubyte* %2914		; <ubyte>:4605 [#uses=2]
	add ubyte %4605, 255		; <ubyte>:4606 [#uses=1]
	store ubyte %4606, ubyte* %2914
	seteq ubyte %4605, 1		; <bool>:1950 [#uses=1]
	br bool %1950, label %1951, label %1950

; <label>:1951		; preds = %1949, %1950
	add uint %1223, 76		; <uint>:1238 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1238		; <ubyte*>:2915 [#uses=1]
	load ubyte* %2915		; <ubyte>:4607 [#uses=1]
	seteq ubyte %4607, 0		; <bool>:1951 [#uses=1]
	br bool %1951, label %1953, label %1952

; <label>:1952		; preds = %1951, %1952
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1238		; <ubyte*>:2916 [#uses=2]
	load ubyte* %2916		; <ubyte>:4608 [#uses=2]
	add ubyte %4608, 255		; <ubyte>:4609 [#uses=1]
	store ubyte %4609, ubyte* %2916
	seteq ubyte %4608, 1		; <bool>:1952 [#uses=1]
	br bool %1952, label %1953, label %1952

; <label>:1953		; preds = %1951, %1952
	add uint %1223, 82		; <uint>:1239 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1239		; <ubyte*>:2917 [#uses=1]
	load ubyte* %2917		; <ubyte>:4610 [#uses=1]
	seteq ubyte %4610, 0		; <bool>:1953 [#uses=1]
	br bool %1953, label %1955, label %1954

; <label>:1954		; preds = %1953, %1954
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1239		; <ubyte*>:2918 [#uses=2]
	load ubyte* %2918		; <ubyte>:4611 [#uses=2]
	add ubyte %4611, 255		; <ubyte>:4612 [#uses=1]
	store ubyte %4612, ubyte* %2918
	seteq ubyte %4611, 1		; <bool>:1954 [#uses=1]
	br bool %1954, label %1955, label %1954

; <label>:1955		; preds = %1953, %1954
	add uint %1223, 88		; <uint>:1240 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1240		; <ubyte*>:2919 [#uses=1]
	load ubyte* %2919		; <ubyte>:4613 [#uses=1]
	seteq ubyte %4613, 0		; <bool>:1955 [#uses=1]
	br bool %1955, label %1957, label %1956

; <label>:1956		; preds = %1955, %1956
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1240		; <ubyte*>:2920 [#uses=2]
	load ubyte* %2920		; <ubyte>:4614 [#uses=2]
	add ubyte %4614, 255		; <ubyte>:4615 [#uses=1]
	store ubyte %4615, ubyte* %2920
	seteq ubyte %4614, 1		; <bool>:1956 [#uses=1]
	br bool %1956, label %1957, label %1956

; <label>:1957		; preds = %1955, %1956
	add uint %1223, 4294967294		; <uint>:1241 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1241		; <ubyte*>:2921 [#uses=1]
	load ubyte* %2921		; <ubyte>:4616 [#uses=1]
	seteq ubyte %4616, 0		; <bool>:1957 [#uses=1]
	br bool %1957, label %1959, label %1958

; <label>:1958		; preds = %1957, %1958
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1241		; <ubyte*>:2922 [#uses=2]
	load ubyte* %2922		; <ubyte>:4617 [#uses=2]
	add ubyte %4617, 255		; <ubyte>:4618 [#uses=1]
	store ubyte %4618, ubyte* %2922
	seteq ubyte %4617, 1		; <bool>:1958 [#uses=1]
	br bool %1958, label %1959, label %1958

; <label>:1959		; preds = %1957, %1958
	add uint %1223, 4294967177		; <uint>:1242 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1242		; <ubyte*>:2923 [#uses=1]
	load ubyte* %2923		; <ubyte>:4619 [#uses=1]
	seteq ubyte %4619, 0		; <bool>:1959 [#uses=1]
	br bool %1959, label %1961, label %1960

; <label>:1960		; preds = %1959, %1960
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1242		; <ubyte*>:2924 [#uses=2]
	load ubyte* %2924		; <ubyte>:4620 [#uses=1]
	add ubyte %4620, 255		; <ubyte>:4621 [#uses=1]
	store ubyte %4621, ubyte* %2924
	add uint %1223, 4294967178		; <uint>:1243 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1243		; <ubyte*>:2925 [#uses=2]
	load ubyte* %2925		; <ubyte>:4622 [#uses=1]
	add ubyte %4622, 1		; <ubyte>:4623 [#uses=1]
	store ubyte %4623, ubyte* %2925
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1241		; <ubyte*>:2926 [#uses=2]
	load ubyte* %2926		; <ubyte>:4624 [#uses=1]
	add ubyte %4624, 1		; <ubyte>:4625 [#uses=1]
	store ubyte %4625, ubyte* %2926
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1242		; <ubyte*>:2927 [#uses=1]
	load ubyte* %2927		; <ubyte>:4626 [#uses=1]
	seteq ubyte %4626, 0		; <bool>:1960 [#uses=1]
	br bool %1960, label %1961, label %1960

; <label>:1961		; preds = %1959, %1960
	add uint %1223, 4294967178		; <uint>:1244 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1244		; <ubyte*>:2928 [#uses=1]
	load ubyte* %2928		; <ubyte>:4627 [#uses=1]
	seteq ubyte %4627, 0		; <bool>:1961 [#uses=1]
	br bool %1961, label %1963, label %1962

; <label>:1962		; preds = %1961, %1962
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1242		; <ubyte*>:2929 [#uses=2]
	load ubyte* %2929		; <ubyte>:4628 [#uses=1]
	add ubyte %4628, 1		; <ubyte>:4629 [#uses=1]
	store ubyte %4629, ubyte* %2929
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1244		; <ubyte*>:2930 [#uses=2]
	load ubyte* %2930		; <ubyte>:4630 [#uses=2]
	add ubyte %4630, 255		; <ubyte>:4631 [#uses=1]
	store ubyte %4631, ubyte* %2930
	seteq ubyte %4630, 1		; <bool>:1962 [#uses=1]
	br bool %1962, label %1963, label %1962

; <label>:1963		; preds = %1961, %1962
	add uint %1223, 4294967189		; <uint>:1245 [#uses=6]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1245		; <ubyte*>:2931 [#uses=1]
	load ubyte* %2931		; <ubyte>:4632 [#uses=1]
	seteq ubyte %4632, 0		; <bool>:1963 [#uses=1]
	br bool %1963, label %1965, label %1964

; <label>:1964		; preds = %1963, %1964
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1245		; <ubyte*>:2932 [#uses=2]
	load ubyte* %2932		; <ubyte>:4633 [#uses=2]
	add ubyte %4633, 255		; <ubyte>:4634 [#uses=1]
	store ubyte %4634, ubyte* %2932
	seteq ubyte %4633, 1		; <bool>:1964 [#uses=1]
	br bool %1964, label %1965, label %1964

; <label>:1965		; preds = %1963, %1964
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1241		; <ubyte*>:2933 [#uses=1]
	load ubyte* %2933		; <ubyte>:4635 [#uses=1]
	seteq ubyte %4635, 0		; <bool>:1965 [#uses=1]
	br bool %1965, label %1967, label %1966

; <label>:1966		; preds = %1965, %1966
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1245		; <ubyte*>:2934 [#uses=2]
	load ubyte* %2934		; <ubyte>:4636 [#uses=1]
	add ubyte %4636, 1		; <ubyte>:4637 [#uses=1]
	store ubyte %4637, ubyte* %2934
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1241		; <ubyte*>:2935 [#uses=2]
	load ubyte* %2935		; <ubyte>:4638 [#uses=2]
	add ubyte %4638, 255		; <ubyte>:4639 [#uses=1]
	store ubyte %4639, ubyte* %2935
	seteq ubyte %4638, 1		; <bool>:1966 [#uses=1]
	br bool %1966, label %1967, label %1966

; <label>:1967		; preds = %1965, %1966
	add uint %1223, 4294967179		; <uint>:1246 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1246		; <ubyte*>:2936 [#uses=1]
	load ubyte* %2936		; <ubyte>:4640 [#uses=1]
	seteq ubyte %4640, 0		; <bool>:1967 [#uses=1]
	br bool %1967, label %1969, label %1968

; <label>:1968		; preds = %1967, %1968
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1246		; <ubyte*>:2937 [#uses=2]
	load ubyte* %2937		; <ubyte>:4641 [#uses=2]
	add ubyte %4641, 255		; <ubyte>:4642 [#uses=1]
	store ubyte %4642, ubyte* %2937
	seteq ubyte %4641, 1		; <bool>:1968 [#uses=1]
	br bool %1968, label %1969, label %1968

; <label>:1969		; preds = %1967, %1968
	add uint %1223, 4294967292		; <uint>:1247 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1247		; <ubyte*>:2938 [#uses=1]
	load ubyte* %2938		; <ubyte>:4643 [#uses=1]
	seteq ubyte %4643, 0		; <bool>:1969 [#uses=1]
	br bool %1969, label %1971, label %1970

; <label>:1970		; preds = %1969, %1970
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1246		; <ubyte*>:2939 [#uses=2]
	load ubyte* %2939		; <ubyte>:4644 [#uses=1]
	add ubyte %4644, 1		; <ubyte>:4645 [#uses=1]
	store ubyte %4645, ubyte* %2939
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1247		; <ubyte*>:2940 [#uses=2]
	load ubyte* %2940		; <ubyte>:4646 [#uses=2]
	add ubyte %4646, 255		; <ubyte>:4647 [#uses=1]
	store ubyte %4647, ubyte* %2940
	seteq ubyte %4646, 1		; <bool>:1970 [#uses=1]
	br bool %1970, label %1971, label %1970

; <label>:1971		; preds = %1969, %1970
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1245		; <ubyte*>:2941 [#uses=1]
	load ubyte* %2941		; <ubyte>:4648 [#uses=1]
	seteq ubyte %4648, 0		; <bool>:1971 [#uses=1]
	br bool %1971, label %1973, label %1972

; <label>:1972		; preds = %1971, %1979
	phi uint [ %1245, %1971 ], [ %1254, %1979 ]		; <uint>:1248 [#uses=8]
	add uint %1248, 4294967292		; <uint>:1249 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1249		; <ubyte*>:2942 [#uses=1]
	load ubyte* %2942		; <ubyte>:4649 [#uses=1]
	seteq ubyte %4649, 0		; <bool>:1972 [#uses=1]
	br bool %1972, label %1975, label %1974

; <label>:1973		; preds = %1971, %1979
	phi uint [ %1245, %1971 ], [ %1254, %1979 ]		; <uint>:1250 [#uses=5]
	add uint %1250, 4294967294		; <uint>:1251 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1251		; <ubyte*>:2943 [#uses=1]
	load ubyte* %2943		; <ubyte>:4650 [#uses=1]
	seteq ubyte %4650, 0		; <bool>:1973 [#uses=1]
	br bool %1973, label %1981, label %1980

; <label>:1974		; preds = %1972, %1974
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1249		; <ubyte*>:2944 [#uses=2]
	load ubyte* %2944		; <ubyte>:4651 [#uses=2]
	add ubyte %4651, 255		; <ubyte>:4652 [#uses=1]
	store ubyte %4652, ubyte* %2944
	seteq ubyte %4651, 1		; <bool>:1974 [#uses=1]
	br bool %1974, label %1975, label %1974

; <label>:1975		; preds = %1972, %1974
	add uint %1248, 4294967286		; <uint>:1252 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1252		; <ubyte*>:2945 [#uses=1]
	load ubyte* %2945		; <ubyte>:4653 [#uses=1]
	seteq ubyte %4653, 0		; <bool>:1975 [#uses=1]
	br bool %1975, label %1977, label %1976

; <label>:1976		; preds = %1975, %1976
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1252		; <ubyte*>:2946 [#uses=2]
	load ubyte* %2946		; <ubyte>:4654 [#uses=1]
	add ubyte %4654, 255		; <ubyte>:4655 [#uses=1]
	store ubyte %4655, ubyte* %2946
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1249		; <ubyte*>:2947 [#uses=2]
	load ubyte* %2947		; <ubyte>:4656 [#uses=1]
	add ubyte %4656, 1		; <ubyte>:4657 [#uses=1]
	store ubyte %4657, ubyte* %2947
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1252		; <ubyte*>:2948 [#uses=1]
	load ubyte* %2948		; <ubyte>:4658 [#uses=1]
	seteq ubyte %4658, 0		; <bool>:1976 [#uses=1]
	br bool %1976, label %1977, label %1976

; <label>:1977		; preds = %1975, %1976
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1248		; <ubyte*>:2949 [#uses=1]
	load ubyte* %2949		; <ubyte>:4659 [#uses=1]
	seteq ubyte %4659, 0		; <bool>:1977 [#uses=1]
	br bool %1977, label %1979, label %1978

; <label>:1978		; preds = %1977, %1978
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1248		; <ubyte*>:2950 [#uses=2]
	load ubyte* %2950		; <ubyte>:4660 [#uses=1]
	add ubyte %4660, 255		; <ubyte>:4661 [#uses=1]
	store ubyte %4661, ubyte* %2950
	add uint %1248, 6		; <uint>:1253 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1253		; <ubyte*>:2951 [#uses=2]
	load ubyte* %2951		; <ubyte>:4662 [#uses=1]
	add ubyte %4662, 1		; <ubyte>:4663 [#uses=1]
	store ubyte %4663, ubyte* %2951
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1248		; <ubyte*>:2952 [#uses=1]
	load ubyte* %2952		; <ubyte>:4664 [#uses=1]
	seteq ubyte %4664, 0		; <bool>:1978 [#uses=1]
	br bool %1978, label %1979, label %1978

; <label>:1979		; preds = %1977, %1978
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1248		; <ubyte*>:2953 [#uses=2]
	load ubyte* %2953		; <ubyte>:4665 [#uses=1]
	add ubyte %4665, 1		; <ubyte>:4666 [#uses=1]
	store ubyte %4666, ubyte* %2953
	add uint %1248, 6		; <uint>:1254 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1254		; <ubyte*>:2954 [#uses=2]
	load ubyte* %2954		; <ubyte>:4667 [#uses=2]
	add ubyte %4667, 255		; <ubyte>:4668 [#uses=1]
	store ubyte %4668, ubyte* %2954
	seteq ubyte %4667, 1		; <bool>:1979 [#uses=1]
	br bool %1979, label %1973, label %1972

; <label>:1980		; preds = %1973, %1980
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1251		; <ubyte*>:2955 [#uses=2]
	load ubyte* %2955		; <ubyte>:4669 [#uses=2]
	add ubyte %4669, 255		; <ubyte>:4670 [#uses=1]
	store ubyte %4670, ubyte* %2955
	seteq ubyte %4669, 1		; <bool>:1980 [#uses=1]
	br bool %1980, label %1981, label %1980

; <label>:1981		; preds = %1973, %1980
	add uint %1250, 4294967286		; <uint>:1255 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1255		; <ubyte*>:2956 [#uses=1]
	load ubyte* %2956		; <ubyte>:4671 [#uses=1]
	seteq ubyte %4671, 0		; <bool>:1981 [#uses=1]
	br bool %1981, label %1983, label %1982

; <label>:1982		; preds = %1981, %1982
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1255		; <ubyte*>:2957 [#uses=2]
	load ubyte* %2957		; <ubyte>:4672 [#uses=1]
	add ubyte %4672, 255		; <ubyte>:4673 [#uses=1]
	store ubyte %4673, ubyte* %2957
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1251		; <ubyte*>:2958 [#uses=2]
	load ubyte* %2958		; <ubyte>:4674 [#uses=1]
	add ubyte %4674, 1		; <ubyte>:4675 [#uses=1]
	store ubyte %4675, ubyte* %2958
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1255		; <ubyte*>:2959 [#uses=1]
	load ubyte* %2959		; <ubyte>:4676 [#uses=1]
	seteq ubyte %4676, 0		; <bool>:1982 [#uses=1]
	br bool %1982, label %1983, label %1982

; <label>:1983		; preds = %1981, %1982
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1250		; <ubyte*>:2960 [#uses=2]
	load ubyte* %2960		; <ubyte>:4677 [#uses=2]
	add ubyte %4677, 1		; <ubyte>:4678 [#uses=1]
	store ubyte %4678, ubyte* %2960
	seteq ubyte %4677, 255		; <bool>:1983 [#uses=1]
	br bool %1983, label %1985, label %1984

; <label>:1984		; preds = %1983, %1984
	phi uint [ %1250, %1983 ], [ %1257, %1984 ]		; <uint>:1256 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1256		; <ubyte*>:2961 [#uses=2]
	load ubyte* %2961		; <ubyte>:4679 [#uses=1]
	add ubyte %4679, 255		; <ubyte>:4680 [#uses=1]
	store ubyte %4680, ubyte* %2961
	add uint %1256, 4294967290		; <uint>:1257 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1257		; <ubyte*>:2962 [#uses=1]
	load ubyte* %2962		; <ubyte>:4681 [#uses=1]
	seteq ubyte %4681, 0		; <bool>:1984 [#uses=1]
	br bool %1984, label %1985, label %1984

; <label>:1985		; preds = %1983, %1984
	phi uint [ %1250, %1983 ], [ %1257, %1984 ]		; <uint>:1258 [#uses=10]
	add uint %1258, 109		; <uint>:1259 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1259		; <ubyte*>:2963 [#uses=1]
	load ubyte* %2963		; <ubyte>:4682 [#uses=1]
	seteq ubyte %4682, 0		; <bool>:1985 [#uses=1]
	br bool %1985, label %1987, label %1986

; <label>:1986		; preds = %1985, %1986
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1259		; <ubyte*>:2964 [#uses=2]
	load ubyte* %2964		; <ubyte>:4683 [#uses=2]
	add ubyte %4683, 255		; <ubyte>:4684 [#uses=1]
	store ubyte %4684, ubyte* %2964
	seteq ubyte %4683, 1		; <bool>:1986 [#uses=1]
	br bool %1986, label %1987, label %1986

; <label>:1987		; preds = %1985, %1986
	add uint %1258, 107		; <uint>:1260 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1260		; <ubyte*>:2965 [#uses=1]
	load ubyte* %2965		; <ubyte>:4685 [#uses=1]
	seteq ubyte %4685, 0		; <bool>:1987 [#uses=1]
	br bool %1987, label %1989, label %1988

; <label>:1988		; preds = %1987, %1988
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1260		; <ubyte*>:2966 [#uses=2]
	load ubyte* %2966		; <ubyte>:4686 [#uses=1]
	add ubyte %4686, 255		; <ubyte>:4687 [#uses=1]
	store ubyte %4687, ubyte* %2966
	add uint %1258, 108		; <uint>:1261 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1261		; <ubyte*>:2967 [#uses=2]
	load ubyte* %2967		; <ubyte>:4688 [#uses=1]
	add ubyte %4688, 1		; <ubyte>:4689 [#uses=1]
	store ubyte %4689, ubyte* %2967
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1259		; <ubyte*>:2968 [#uses=2]
	load ubyte* %2968		; <ubyte>:4690 [#uses=1]
	add ubyte %4690, 1		; <ubyte>:4691 [#uses=1]
	store ubyte %4691, ubyte* %2968
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1260		; <ubyte*>:2969 [#uses=1]
	load ubyte* %2969		; <ubyte>:4692 [#uses=1]
	seteq ubyte %4692, 0		; <bool>:1988 [#uses=1]
	br bool %1988, label %1989, label %1988

; <label>:1989		; preds = %1987, %1988
	add uint %1258, 108		; <uint>:1262 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1262		; <ubyte*>:2970 [#uses=1]
	load ubyte* %2970		; <ubyte>:4693 [#uses=1]
	seteq ubyte %4693, 0		; <bool>:1989 [#uses=1]
	br bool %1989, label %1991, label %1990

; <label>:1990		; preds = %1989, %1990
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1260		; <ubyte*>:2971 [#uses=2]
	load ubyte* %2971		; <ubyte>:4694 [#uses=1]
	add ubyte %4694, 1		; <ubyte>:4695 [#uses=1]
	store ubyte %4695, ubyte* %2971
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1262		; <ubyte*>:2972 [#uses=2]
	load ubyte* %2972		; <ubyte>:4696 [#uses=2]
	add ubyte %4696, 255		; <ubyte>:4697 [#uses=1]
	store ubyte %4697, ubyte* %2972
	seteq ubyte %4696, 1		; <bool>:1990 [#uses=1]
	br bool %1990, label %1991, label %1990

; <label>:1991		; preds = %1989, %1990
	add uint %1258, 111		; <uint>:1263 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1263		; <ubyte*>:2973 [#uses=1]
	load ubyte* %2973		; <ubyte>:4698 [#uses=1]
	seteq ubyte %4698, 0		; <bool>:1991 [#uses=1]
	br bool %1991, label %1993, label %1992

; <label>:1992		; preds = %1991, %1992
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1263		; <ubyte*>:2974 [#uses=2]
	load ubyte* %2974		; <ubyte>:4699 [#uses=2]
	add ubyte %4699, 255		; <ubyte>:4700 [#uses=1]
	store ubyte %4700, ubyte* %2974
	seteq ubyte %4699, 1		; <bool>:1992 [#uses=1]
	br bool %1992, label %1993, label %1992

; <label>:1993		; preds = %1991, %1992
	add uint %1258, 103		; <uint>:1264 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1264		; <ubyte*>:2975 [#uses=1]
	load ubyte* %2975		; <ubyte>:4701 [#uses=1]
	seteq ubyte %4701, 0		; <bool>:1993 [#uses=1]
	br bool %1993, label %1995, label %1994

; <label>:1994		; preds = %1993, %1994
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1264		; <ubyte*>:2976 [#uses=2]
	load ubyte* %2976		; <ubyte>:4702 [#uses=1]
	add ubyte %4702, 255		; <ubyte>:4703 [#uses=1]
	store ubyte %4703, ubyte* %2976
	add uint %1258, 104		; <uint>:1265 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1265		; <ubyte*>:2977 [#uses=2]
	load ubyte* %2977		; <ubyte>:4704 [#uses=1]
	add ubyte %4704, 1		; <ubyte>:4705 [#uses=1]
	store ubyte %4705, ubyte* %2977
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1263		; <ubyte*>:2978 [#uses=2]
	load ubyte* %2978		; <ubyte>:4706 [#uses=1]
	add ubyte %4706, 1		; <ubyte>:4707 [#uses=1]
	store ubyte %4707, ubyte* %2978
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1264		; <ubyte*>:2979 [#uses=1]
	load ubyte* %2979		; <ubyte>:4708 [#uses=1]
	seteq ubyte %4708, 0		; <bool>:1994 [#uses=1]
	br bool %1994, label %1995, label %1994

; <label>:1995		; preds = %1993, %1994
	add uint %1258, 104		; <uint>:1266 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1266		; <ubyte*>:2980 [#uses=1]
	load ubyte* %2980		; <ubyte>:4709 [#uses=1]
	seteq ubyte %4709, 0		; <bool>:1995 [#uses=1]
	br bool %1995, label %1997, label %1996

; <label>:1996		; preds = %1995, %1996
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1264		; <ubyte*>:2981 [#uses=2]
	load ubyte* %2981		; <ubyte>:4710 [#uses=1]
	add ubyte %4710, 1		; <ubyte>:4711 [#uses=1]
	store ubyte %4711, ubyte* %2981
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1266		; <ubyte*>:2982 [#uses=2]
	load ubyte* %2982		; <ubyte>:4712 [#uses=2]
	add ubyte %4712, 255		; <ubyte>:4713 [#uses=1]
	store ubyte %4713, ubyte* %2982
	seteq ubyte %4712, 1		; <bool>:1996 [#uses=1]
	br bool %1996, label %1997, label %1996

; <label>:1997		; preds = %1995, %1996
	add uint %1258, 6		; <uint>:1267 [#uses=6]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1267		; <ubyte*>:2983 [#uses=1]
	load ubyte* %2983		; <ubyte>:4714 [#uses=1]
	seteq ubyte %4714, 0		; <bool>:1997 [#uses=1]
	br bool %1997, label %1999, label %1998

; <label>:1998		; preds = %1997, %1998
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1267		; <ubyte*>:2984 [#uses=2]
	load ubyte* %2984		; <ubyte>:4715 [#uses=2]
	add ubyte %4715, 255		; <ubyte>:4716 [#uses=1]
	store ubyte %4716, ubyte* %2984
	seteq ubyte %4715, 1		; <bool>:1998 [#uses=1]
	br bool %1998, label %1999, label %1998

; <label>:1999		; preds = %1997, %1998
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1263		; <ubyte*>:2985 [#uses=1]
	load ubyte* %2985		; <ubyte>:4717 [#uses=1]
	seteq ubyte %4717, 0		; <bool>:1999 [#uses=1]
	br bool %1999, label %2001, label %2000

; <label>:2000		; preds = %1999, %2000
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1267		; <ubyte*>:2986 [#uses=2]
	load ubyte* %2986		; <ubyte>:4718 [#uses=1]
	add ubyte %4718, 1		; <ubyte>:4719 [#uses=1]
	store ubyte %4719, ubyte* %2986
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1263		; <ubyte*>:2987 [#uses=2]
	load ubyte* %2987		; <ubyte>:4720 [#uses=2]
	add ubyte %4720, 255		; <ubyte>:4721 [#uses=1]
	store ubyte %4721, ubyte* %2987
	seteq ubyte %4720, 1		; <bool>:2000 [#uses=1]
	br bool %2000, label %2001, label %2000

; <label>:2001		; preds = %1999, %2000
	add uint %1258, 4294967292		; <uint>:1268 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1268		; <ubyte*>:2988 [#uses=1]
	load ubyte* %2988		; <ubyte>:4722 [#uses=1]
	seteq ubyte %4722, 0		; <bool>:2001 [#uses=1]
	br bool %2001, label %2003, label %2002

; <label>:2002		; preds = %2001, %2002
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1268		; <ubyte*>:2989 [#uses=2]
	load ubyte* %2989		; <ubyte>:4723 [#uses=2]
	add ubyte %4723, 255		; <ubyte>:4724 [#uses=1]
	store ubyte %4724, ubyte* %2989
	seteq ubyte %4723, 1		; <bool>:2002 [#uses=1]
	br bool %2002, label %2003, label %2002

; <label>:2003		; preds = %2001, %2002
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1259		; <ubyte*>:2990 [#uses=1]
	load ubyte* %2990		; <ubyte>:4725 [#uses=1]
	seteq ubyte %4725, 0		; <bool>:2003 [#uses=1]
	br bool %2003, label %2005, label %2004

; <label>:2004		; preds = %2003, %2004
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1268		; <ubyte*>:2991 [#uses=2]
	load ubyte* %2991		; <ubyte>:4726 [#uses=1]
	add ubyte %4726, 1		; <ubyte>:4727 [#uses=1]
	store ubyte %4727, ubyte* %2991
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1259		; <ubyte*>:2992 [#uses=2]
	load ubyte* %2992		; <ubyte>:4728 [#uses=2]
	add ubyte %4728, 255		; <ubyte>:4729 [#uses=1]
	store ubyte %4729, ubyte* %2992
	seteq ubyte %4728, 1		; <bool>:2004 [#uses=1]
	br bool %2004, label %2005, label %2004

; <label>:2005		; preds = %2003, %2004
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1267		; <ubyte*>:2993 [#uses=1]
	load ubyte* %2993		; <ubyte>:4730 [#uses=1]
	seteq ubyte %4730, 0		; <bool>:2005 [#uses=1]
	br bool %2005, label %2007, label %2006

; <label>:2006		; preds = %2005, %2013
	phi uint [ %1267, %2005 ], [ %1275, %2013 ]		; <uint>:1269 [#uses=8]
	add uint %1269, 4294967292		; <uint>:1270 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1270		; <ubyte*>:2994 [#uses=1]
	load ubyte* %2994		; <ubyte>:4731 [#uses=1]
	seteq ubyte %4731, 0		; <bool>:2006 [#uses=1]
	br bool %2006, label %2009, label %2008

; <label>:2007		; preds = %2005, %2013
	phi uint [ %1267, %2005 ], [ %1275, %2013 ]		; <uint>:1271 [#uses=5]
	add uint %1271, 4294967294		; <uint>:1272 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1272		; <ubyte*>:2995 [#uses=1]
	load ubyte* %2995		; <ubyte>:4732 [#uses=1]
	seteq ubyte %4732, 0		; <bool>:2007 [#uses=1]
	br bool %2007, label %2015, label %2014

; <label>:2008		; preds = %2006, %2008
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1270		; <ubyte*>:2996 [#uses=2]
	load ubyte* %2996		; <ubyte>:4733 [#uses=2]
	add ubyte %4733, 255		; <ubyte>:4734 [#uses=1]
	store ubyte %4734, ubyte* %2996
	seteq ubyte %4733, 1		; <bool>:2008 [#uses=1]
	br bool %2008, label %2009, label %2008

; <label>:2009		; preds = %2006, %2008
	add uint %1269, 4294967286		; <uint>:1273 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1273		; <ubyte*>:2997 [#uses=1]
	load ubyte* %2997		; <ubyte>:4735 [#uses=1]
	seteq ubyte %4735, 0		; <bool>:2009 [#uses=1]
	br bool %2009, label %2011, label %2010

; <label>:2010		; preds = %2009, %2010
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1273		; <ubyte*>:2998 [#uses=2]
	load ubyte* %2998		; <ubyte>:4736 [#uses=1]
	add ubyte %4736, 255		; <ubyte>:4737 [#uses=1]
	store ubyte %4737, ubyte* %2998
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1270		; <ubyte*>:2999 [#uses=2]
	load ubyte* %2999		; <ubyte>:4738 [#uses=1]
	add ubyte %4738, 1		; <ubyte>:4739 [#uses=1]
	store ubyte %4739, ubyte* %2999
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1273		; <ubyte*>:3000 [#uses=1]
	load ubyte* %3000		; <ubyte>:4740 [#uses=1]
	seteq ubyte %4740, 0		; <bool>:2010 [#uses=1]
	br bool %2010, label %2011, label %2010

; <label>:2011		; preds = %2009, %2010
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1269		; <ubyte*>:3001 [#uses=1]
	load ubyte* %3001		; <ubyte>:4741 [#uses=1]
	seteq ubyte %4741, 0		; <bool>:2011 [#uses=1]
	br bool %2011, label %2013, label %2012

; <label>:2012		; preds = %2011, %2012
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1269		; <ubyte*>:3002 [#uses=2]
	load ubyte* %3002		; <ubyte>:4742 [#uses=1]
	add ubyte %4742, 255		; <ubyte>:4743 [#uses=1]
	store ubyte %4743, ubyte* %3002
	add uint %1269, 6		; <uint>:1274 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1274		; <ubyte*>:3003 [#uses=2]
	load ubyte* %3003		; <ubyte>:4744 [#uses=1]
	add ubyte %4744, 1		; <ubyte>:4745 [#uses=1]
	store ubyte %4745, ubyte* %3003
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1269		; <ubyte*>:3004 [#uses=1]
	load ubyte* %3004		; <ubyte>:4746 [#uses=1]
	seteq ubyte %4746, 0		; <bool>:2012 [#uses=1]
	br bool %2012, label %2013, label %2012

; <label>:2013		; preds = %2011, %2012
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1269		; <ubyte*>:3005 [#uses=2]
	load ubyte* %3005		; <ubyte>:4747 [#uses=1]
	add ubyte %4747, 1		; <ubyte>:4748 [#uses=1]
	store ubyte %4748, ubyte* %3005
	add uint %1269, 6		; <uint>:1275 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1275		; <ubyte*>:3006 [#uses=2]
	load ubyte* %3006		; <ubyte>:4749 [#uses=2]
	add ubyte %4749, 255		; <ubyte>:4750 [#uses=1]
	store ubyte %4750, ubyte* %3006
	seteq ubyte %4749, 1		; <bool>:2013 [#uses=1]
	br bool %2013, label %2007, label %2006

; <label>:2014		; preds = %2007, %2014
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1272		; <ubyte*>:3007 [#uses=2]
	load ubyte* %3007		; <ubyte>:4751 [#uses=2]
	add ubyte %4751, 255		; <ubyte>:4752 [#uses=1]
	store ubyte %4752, ubyte* %3007
	seteq ubyte %4751, 1		; <bool>:2014 [#uses=1]
	br bool %2014, label %2015, label %2014

; <label>:2015		; preds = %2007, %2014
	add uint %1271, 4294967286		; <uint>:1276 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1276		; <ubyte*>:3008 [#uses=1]
	load ubyte* %3008		; <ubyte>:4753 [#uses=1]
	seteq ubyte %4753, 0		; <bool>:2015 [#uses=1]
	br bool %2015, label %2017, label %2016

; <label>:2016		; preds = %2015, %2016
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1276		; <ubyte*>:3009 [#uses=2]
	load ubyte* %3009		; <ubyte>:4754 [#uses=1]
	add ubyte %4754, 255		; <ubyte>:4755 [#uses=1]
	store ubyte %4755, ubyte* %3009
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1272		; <ubyte*>:3010 [#uses=2]
	load ubyte* %3010		; <ubyte>:4756 [#uses=1]
	add ubyte %4756, 1		; <ubyte>:4757 [#uses=1]
	store ubyte %4757, ubyte* %3010
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1276		; <ubyte*>:3011 [#uses=1]
	load ubyte* %3011		; <ubyte>:4758 [#uses=1]
	seteq ubyte %4758, 0		; <bool>:2016 [#uses=1]
	br bool %2016, label %2017, label %2016

; <label>:2017		; preds = %2015, %2016
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1271		; <ubyte*>:3012 [#uses=2]
	load ubyte* %3012		; <ubyte>:4759 [#uses=2]
	add ubyte %4759, 1		; <ubyte>:4760 [#uses=1]
	store ubyte %4760, ubyte* %3012
	seteq ubyte %4759, 255		; <bool>:2017 [#uses=1]
	br bool %2017, label %2019, label %2018

; <label>:2018		; preds = %2017, %2018
	phi uint [ %1271, %2017 ], [ %1278, %2018 ]		; <uint>:1277 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1277		; <ubyte*>:3013 [#uses=2]
	load ubyte* %3013		; <ubyte>:4761 [#uses=1]
	add ubyte %4761, 255		; <ubyte>:4762 [#uses=1]
	store ubyte %4762, ubyte* %3013
	add uint %1277, 4294967290		; <uint>:1278 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1278		; <ubyte*>:3014 [#uses=1]
	load ubyte* %3014		; <ubyte>:4763 [#uses=1]
	seteq ubyte %4763, 0		; <bool>:2018 [#uses=1]
	br bool %2018, label %2019, label %2018

; <label>:2019		; preds = %2017, %2018
	phi uint [ %1271, %2017 ], [ %1278, %2018 ]		; <uint>:1279 [#uses=22]
	add uint %1279, 109		; <uint>:1280 [#uses=18]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1280		; <ubyte*>:3015 [#uses=1]
	load ubyte* %3015		; <ubyte>:4764 [#uses=1]
	seteq ubyte %4764, 0		; <bool>:2019 [#uses=1]
	br bool %2019, label %2021, label %2020

; <label>:2020		; preds = %2019, %2020
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1280		; <ubyte*>:3016 [#uses=2]
	load ubyte* %3016		; <ubyte>:4765 [#uses=2]
	add ubyte %4765, 255		; <ubyte>:4766 [#uses=1]
	store ubyte %4766, ubyte* %3016
	seteq ubyte %4765, 1		; <bool>:2020 [#uses=1]
	br bool %2020, label %2021, label %2020

; <label>:2021		; preds = %2019, %2020
	add uint %1279, 103		; <uint>:1281 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1281		; <ubyte*>:3017 [#uses=1]
	load ubyte* %3017		; <ubyte>:4767 [#uses=1]
	seteq ubyte %4767, 0		; <bool>:2021 [#uses=1]
	br bool %2021, label %2023, label %2022

; <label>:2022		; preds = %2021, %2022
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1281		; <ubyte*>:3018 [#uses=2]
	load ubyte* %3018		; <ubyte>:4768 [#uses=1]
	add ubyte %4768, 255		; <ubyte>:4769 [#uses=1]
	store ubyte %4769, ubyte* %3018
	add uint %1279, 104		; <uint>:1282 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1282		; <ubyte*>:3019 [#uses=2]
	load ubyte* %3019		; <ubyte>:4770 [#uses=1]
	add ubyte %4770, 1		; <ubyte>:4771 [#uses=1]
	store ubyte %4771, ubyte* %3019
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1280		; <ubyte*>:3020 [#uses=2]
	load ubyte* %3020		; <ubyte>:4772 [#uses=1]
	add ubyte %4772, 1		; <ubyte>:4773 [#uses=1]
	store ubyte %4773, ubyte* %3020
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1281		; <ubyte*>:3021 [#uses=1]
	load ubyte* %3021		; <ubyte>:4774 [#uses=1]
	seteq ubyte %4774, 0		; <bool>:2022 [#uses=1]
	br bool %2022, label %2023, label %2022

; <label>:2023		; preds = %2021, %2022
	add uint %1279, 104		; <uint>:1283 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1283		; <ubyte*>:3022 [#uses=1]
	load ubyte* %3022		; <ubyte>:4775 [#uses=1]
	seteq ubyte %4775, 0		; <bool>:2023 [#uses=1]
	br bool %2023, label %2025, label %2024

; <label>:2024		; preds = %2023, %2024
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1281		; <ubyte*>:3023 [#uses=2]
	load ubyte* %3023		; <ubyte>:4776 [#uses=1]
	add ubyte %4776, 1		; <ubyte>:4777 [#uses=1]
	store ubyte %4777, ubyte* %3023
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1283		; <ubyte*>:3024 [#uses=2]
	load ubyte* %3024		; <ubyte>:4778 [#uses=2]
	add ubyte %4778, 255		; <ubyte>:4779 [#uses=1]
	store ubyte %4779, ubyte* %3024
	seteq ubyte %4778, 1		; <bool>:2024 [#uses=1]
	br bool %2024, label %2025, label %2024

; <label>:2025		; preds = %2023, %2024
	add uint %1279, 111		; <uint>:1284 [#uses=15]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1284		; <ubyte*>:3025 [#uses=1]
	load ubyte* %3025		; <ubyte>:4780 [#uses=1]
	seteq ubyte %4780, 0		; <bool>:2025 [#uses=1]
	br bool %2025, label %2027, label %2026

; <label>:2026		; preds = %2025, %2026
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1284		; <ubyte*>:3026 [#uses=2]
	load ubyte* %3026		; <ubyte>:4781 [#uses=2]
	add ubyte %4781, 255		; <ubyte>:4782 [#uses=1]
	store ubyte %4782, ubyte* %3026
	seteq ubyte %4781, 1		; <bool>:2026 [#uses=1]
	br bool %2026, label %2027, label %2026

; <label>:2027		; preds = %2025, %2026
	add uint %1279, 4294967290		; <uint>:1285 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1285		; <ubyte*>:3027 [#uses=1]
	load ubyte* %3027		; <ubyte>:4783 [#uses=1]
	seteq ubyte %4783, 0		; <bool>:2027 [#uses=1]
	br bool %2027, label %2029, label %2028

; <label>:2028		; preds = %2027, %2028
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1285		; <ubyte*>:3028 [#uses=2]
	load ubyte* %3028		; <ubyte>:4784 [#uses=1]
	add ubyte %4784, 255		; <ubyte>:4785 [#uses=1]
	store ubyte %4785, ubyte* %3028
	add uint %1279, 4294967291		; <uint>:1286 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1286		; <ubyte*>:3029 [#uses=2]
	load ubyte* %3029		; <ubyte>:4786 [#uses=1]
	add ubyte %4786, 1		; <ubyte>:4787 [#uses=1]
	store ubyte %4787, ubyte* %3029
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1284		; <ubyte*>:3030 [#uses=2]
	load ubyte* %3030		; <ubyte>:4788 [#uses=1]
	add ubyte %4788, 1		; <ubyte>:4789 [#uses=1]
	store ubyte %4789, ubyte* %3030
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1285		; <ubyte*>:3031 [#uses=1]
	load ubyte* %3031		; <ubyte>:4790 [#uses=1]
	seteq ubyte %4790, 0		; <bool>:2028 [#uses=1]
	br bool %2028, label %2029, label %2028

; <label>:2029		; preds = %2027, %2028
	add uint %1279, 4294967291		; <uint>:1287 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1287		; <ubyte*>:3032 [#uses=1]
	load ubyte* %3032		; <ubyte>:4791 [#uses=1]
	seteq ubyte %4791, 0		; <bool>:2029 [#uses=1]
	br bool %2029, label %2031, label %2030

; <label>:2030		; preds = %2029, %2030
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1285		; <ubyte*>:3033 [#uses=2]
	load ubyte* %3033		; <ubyte>:4792 [#uses=1]
	add ubyte %4792, 1		; <ubyte>:4793 [#uses=1]
	store ubyte %4793, ubyte* %3033
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1287		; <ubyte*>:3034 [#uses=2]
	load ubyte* %3034		; <ubyte>:4794 [#uses=2]
	add ubyte %4794, 255		; <ubyte>:4795 [#uses=1]
	store ubyte %4795, ubyte* %3034
	seteq ubyte %4794, 1		; <bool>:2030 [#uses=1]
	br bool %2030, label %2031, label %2030

; <label>:2031		; preds = %2029, %2030
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1284		; <ubyte*>:3035 [#uses=1]
	load ubyte* %3035		; <ubyte>:4796 [#uses=1]
	seteq ubyte %4796, 0		; <bool>:2031 [#uses=1]
	br bool %2031, label %2033, label %2032

; <label>:2032		; preds = %2031, %2032
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1280		; <ubyte*>:3036 [#uses=2]
	load ubyte* %3036		; <ubyte>:4797 [#uses=1]
	add ubyte %4797, 255		; <ubyte>:4798 [#uses=1]
	store ubyte %4798, ubyte* %3036
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1284		; <ubyte*>:3037 [#uses=2]
	load ubyte* %3037		; <ubyte>:4799 [#uses=2]
	add ubyte %4799, 255		; <ubyte>:4800 [#uses=1]
	store ubyte %4800, ubyte* %3037
	seteq ubyte %4799, 1		; <bool>:2032 [#uses=1]
	br bool %2032, label %2033, label %2032

; <label>:2033		; preds = %2031, %2032
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1284		; <ubyte*>:3038 [#uses=2]
	load ubyte* %3038		; <ubyte>:4801 [#uses=1]
	add ubyte %4801, 1		; <ubyte>:4802 [#uses=1]
	store ubyte %4802, ubyte* %3038
	add uint %1279, 113		; <uint>:1288 [#uses=9]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1288		; <ubyte*>:3039 [#uses=2]
	load ubyte* %3039		; <ubyte>:4803 [#uses=2]
	add ubyte %4803, 1		; <ubyte>:4804 [#uses=1]
	store ubyte %4804, ubyte* %3039
	seteq ubyte %4803, 255		; <bool>:2033 [#uses=1]
	br bool %2033, label %2035, label %2034

; <label>:2034		; preds = %2033, %2057
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1288		; <ubyte*>:3040 [#uses=2]
	load ubyte* %3040		; <ubyte>:4805 [#uses=1]
	add ubyte %4805, 1		; <ubyte>:4806 [#uses=1]
	store ubyte %4806, ubyte* %3040
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1280		; <ubyte*>:3041 [#uses=1]
	load ubyte* %3041		; <ubyte>:4807 [#uses=1]
	seteq ubyte %4807, 0		; <bool>:2034 [#uses=1]
	br bool %2034, label %2037, label %2036

; <label>:2035		; preds = %2033, %2057
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1280		; <ubyte*>:3042 [#uses=1]
	load ubyte* %3042		; <ubyte>:4808 [#uses=1]
	seteq ubyte %4808, 0		; <bool>:2035 [#uses=1]
	br bool %2035, label %2059, label %2058

; <label>:2036		; preds = %2034, %2036
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1280		; <ubyte*>:3043 [#uses=2]
	load ubyte* %3043		; <ubyte>:4809 [#uses=1]
	add ubyte %4809, 255		; <ubyte>:4810 [#uses=1]
	store ubyte %4810, ubyte* %3043
	add uint %1279, 110		; <uint>:1289 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1289		; <ubyte*>:3044 [#uses=2]
	load ubyte* %3044		; <ubyte>:4811 [#uses=1]
	add ubyte %4811, 1		; <ubyte>:4812 [#uses=1]
	store ubyte %4812, ubyte* %3044
	add uint %1279, 114		; <uint>:1290 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1290		; <ubyte*>:3045 [#uses=2]
	load ubyte* %3045		; <ubyte>:4813 [#uses=1]
	add ubyte %4813, 1		; <ubyte>:4814 [#uses=1]
	store ubyte %4814, ubyte* %3045
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1280		; <ubyte*>:3046 [#uses=1]
	load ubyte* %3046		; <ubyte>:4815 [#uses=1]
	seteq ubyte %4815, 0		; <bool>:2036 [#uses=1]
	br bool %2036, label %2037, label %2036

; <label>:2037		; preds = %2034, %2036
	add uint %1279, 110		; <uint>:1291 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1291		; <ubyte*>:3047 [#uses=1]
	load ubyte* %3047		; <ubyte>:4816 [#uses=1]
	seteq ubyte %4816, 0		; <bool>:2037 [#uses=1]
	br bool %2037, label %2039, label %2038

; <label>:2038		; preds = %2037, %2038
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1280		; <ubyte*>:3048 [#uses=2]
	load ubyte* %3048		; <ubyte>:4817 [#uses=1]
	add ubyte %4817, 1		; <ubyte>:4818 [#uses=1]
	store ubyte %4818, ubyte* %3048
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1291		; <ubyte*>:3049 [#uses=2]
	load ubyte* %3049		; <ubyte>:4819 [#uses=2]
	add ubyte %4819, 255		; <ubyte>:4820 [#uses=1]
	store ubyte %4820, ubyte* %3049
	seteq ubyte %4819, 1		; <bool>:2038 [#uses=1]
	br bool %2038, label %2039, label %2038

; <label>:2039		; preds = %2037, %2038
	add uint %1279, 114		; <uint>:1292 [#uses=13]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1292		; <ubyte*>:3050 [#uses=1]
	load ubyte* %3050		; <ubyte>:4821 [#uses=1]
	seteq ubyte %4821, 0		; <bool>:2039 [#uses=1]
	br bool %2039, label %2041, label %2040

; <label>:2040		; preds = %2039, %2043
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1292		; <ubyte*>:3051 [#uses=1]
	load ubyte* %3051		; <ubyte>:4822 [#uses=1]
	seteq ubyte %4822, 0		; <bool>:2040 [#uses=1]
	br bool %2040, label %2043, label %2042

; <label>:2041		; preds = %2039, %2043
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1284		; <ubyte*>:3052 [#uses=1]
	load ubyte* %3052		; <ubyte>:4823 [#uses=1]
	seteq ubyte %4823, 0		; <bool>:2041 [#uses=1]
	br bool %2041, label %2045, label %2044

; <label>:2042		; preds = %2040, %2042
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1292		; <ubyte*>:3053 [#uses=2]
	load ubyte* %3053		; <ubyte>:4824 [#uses=2]
	add ubyte %4824, 255		; <ubyte>:4825 [#uses=1]
	store ubyte %4825, ubyte* %3053
	seteq ubyte %4824, 1		; <bool>:2042 [#uses=1]
	br bool %2042, label %2043, label %2042

; <label>:2043		; preds = %2040, %2042
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1288		; <ubyte*>:3054 [#uses=2]
	load ubyte* %3054		; <ubyte>:4826 [#uses=1]
	add ubyte %4826, 255		; <ubyte>:4827 [#uses=1]
	store ubyte %4827, ubyte* %3054
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1292		; <ubyte*>:3055 [#uses=1]
	load ubyte* %3055		; <ubyte>:4828 [#uses=1]
	seteq ubyte %4828, 0		; <bool>:2043 [#uses=1]
	br bool %2043, label %2041, label %2040

; <label>:2044		; preds = %2041, %2044
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1284		; <ubyte*>:3056 [#uses=2]
	load ubyte* %3056		; <ubyte>:4829 [#uses=1]
	add ubyte %4829, 255		; <ubyte>:4830 [#uses=1]
	store ubyte %4830, ubyte* %3056
	add uint %1279, 112		; <uint>:1293 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1293		; <ubyte*>:3057 [#uses=2]
	load ubyte* %3057		; <ubyte>:4831 [#uses=1]
	add ubyte %4831, 1		; <ubyte>:4832 [#uses=1]
	store ubyte %4832, ubyte* %3057
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1292		; <ubyte*>:3058 [#uses=2]
	load ubyte* %3058		; <ubyte>:4833 [#uses=1]
	add ubyte %4833, 1		; <ubyte>:4834 [#uses=1]
	store ubyte %4834, ubyte* %3058
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1284		; <ubyte*>:3059 [#uses=1]
	load ubyte* %3059		; <ubyte>:4835 [#uses=1]
	seteq ubyte %4835, 0		; <bool>:2044 [#uses=1]
	br bool %2044, label %2045, label %2044

; <label>:2045		; preds = %2041, %2044
	add uint %1279, 112		; <uint>:1294 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1294		; <ubyte*>:3060 [#uses=1]
	load ubyte* %3060		; <ubyte>:4836 [#uses=1]
	seteq ubyte %4836, 0		; <bool>:2045 [#uses=1]
	br bool %2045, label %2047, label %2046

; <label>:2046		; preds = %2045, %2046
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1284		; <ubyte*>:3061 [#uses=2]
	load ubyte* %3061		; <ubyte>:4837 [#uses=1]
	add ubyte %4837, 1		; <ubyte>:4838 [#uses=1]
	store ubyte %4838, ubyte* %3061
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1294		; <ubyte*>:3062 [#uses=2]
	load ubyte* %3062		; <ubyte>:4839 [#uses=2]
	add ubyte %4839, 255		; <ubyte>:4840 [#uses=1]
	store ubyte %4840, ubyte* %3062
	seteq ubyte %4839, 1		; <bool>:2046 [#uses=1]
	br bool %2046, label %2047, label %2046

; <label>:2047		; preds = %2045, %2046
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1292		; <ubyte*>:3063 [#uses=1]
	load ubyte* %3063		; <ubyte>:4841 [#uses=1]
	seteq ubyte %4841, 0		; <bool>:2047 [#uses=1]
	br bool %2047, label %2049, label %2048

; <label>:2048		; preds = %2047, %2051
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1292		; <ubyte*>:3064 [#uses=1]
	load ubyte* %3064		; <ubyte>:4842 [#uses=1]
	seteq ubyte %4842, 0		; <bool>:2048 [#uses=1]
	br bool %2048, label %2051, label %2050

; <label>:2049		; preds = %2047, %2051
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1292		; <ubyte*>:3065 [#uses=2]
	load ubyte* %3065		; <ubyte>:4843 [#uses=1]
	add ubyte %4843, 1		; <ubyte>:4844 [#uses=1]
	store ubyte %4844, ubyte* %3065
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1288		; <ubyte*>:3066 [#uses=1]
	load ubyte* %3066		; <ubyte>:4845 [#uses=1]
	seteq ubyte %4845, 0		; <bool>:2049 [#uses=1]
	br bool %2049, label %2053, label %2052

; <label>:2050		; preds = %2048, %2050
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1292		; <ubyte*>:3067 [#uses=2]
	load ubyte* %3067		; <ubyte>:4846 [#uses=2]
	add ubyte %4846, 255		; <ubyte>:4847 [#uses=1]
	store ubyte %4847, ubyte* %3067
	seteq ubyte %4846, 1		; <bool>:2050 [#uses=1]
	br bool %2050, label %2051, label %2050

; <label>:2051		; preds = %2048, %2050
	add uint %1279, 113		; <uint>:1295 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1295		; <ubyte*>:3068 [#uses=2]
	load ubyte* %3068		; <ubyte>:4848 [#uses=1]
	add ubyte %4848, 255		; <ubyte>:4849 [#uses=1]
	store ubyte %4849, ubyte* %3068
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1292		; <ubyte*>:3069 [#uses=1]
	load ubyte* %3069		; <ubyte>:4850 [#uses=1]
	seteq ubyte %4850, 0		; <bool>:2051 [#uses=1]
	br bool %2051, label %2049, label %2048

; <label>:2052		; preds = %2049, %2055
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1288		; <ubyte*>:3070 [#uses=1]
	load ubyte* %3070		; <ubyte>:4851 [#uses=1]
	seteq ubyte %4851, 0		; <bool>:2052 [#uses=1]
	br bool %2052, label %2055, label %2054

; <label>:2053		; preds = %2049, %2055
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1292		; <ubyte*>:3071 [#uses=1]
	load ubyte* %3071		; <ubyte>:4852 [#uses=1]
	seteq ubyte %4852, 0		; <bool>:2053 [#uses=1]
	br bool %2053, label %2057, label %2056

; <label>:2054		; preds = %2052, %2054
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1288		; <ubyte*>:3072 [#uses=2]
	load ubyte* %3072		; <ubyte>:4853 [#uses=2]
	add ubyte %4853, 255		; <ubyte>:4854 [#uses=1]
	store ubyte %4854, ubyte* %3072
	seteq ubyte %4853, 1		; <bool>:2054 [#uses=1]
	br bool %2054, label %2055, label %2054

; <label>:2055		; preds = %2052, %2054
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1292		; <ubyte*>:3073 [#uses=2]
	load ubyte* %3073		; <ubyte>:4855 [#uses=1]
	add ubyte %4855, 255		; <ubyte>:4856 [#uses=1]
	store ubyte %4856, ubyte* %3073
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1288		; <ubyte*>:3074 [#uses=1]
	load ubyte* %3074		; <ubyte>:4857 [#uses=1]
	seteq ubyte %4857, 0		; <bool>:2055 [#uses=1]
	br bool %2055, label %2053, label %2052

; <label>:2056		; preds = %2053, %2056
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1280		; <ubyte*>:3075 [#uses=2]
	load ubyte* %3075		; <ubyte>:4858 [#uses=1]
	add ubyte %4858, 255		; <ubyte>:4859 [#uses=1]
	store ubyte %4859, ubyte* %3075
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1284		; <ubyte*>:3076 [#uses=2]
	load ubyte* %3076		; <ubyte>:4860 [#uses=1]
	add ubyte %4860, 255		; <ubyte>:4861 [#uses=1]
	store ubyte %4861, ubyte* %3076
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1288		; <ubyte*>:3077 [#uses=2]
	load ubyte* %3077		; <ubyte>:4862 [#uses=1]
	add ubyte %4862, 1		; <ubyte>:4863 [#uses=1]
	store ubyte %4863, ubyte* %3077
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1292		; <ubyte*>:3078 [#uses=2]
	load ubyte* %3078		; <ubyte>:4864 [#uses=2]
	add ubyte %4864, 255		; <ubyte>:4865 [#uses=1]
	store ubyte %4865, ubyte* %3078
	seteq ubyte %4864, 1		; <bool>:2056 [#uses=1]
	br bool %2056, label %2057, label %2056

; <label>:2057		; preds = %2053, %2056
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1288		; <ubyte*>:3079 [#uses=1]
	load ubyte* %3079		; <ubyte>:4866 [#uses=1]
	seteq ubyte %4866, 0		; <bool>:2057 [#uses=1]
	br bool %2057, label %2035, label %2034

; <label>:2058		; preds = %2035, %2061
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1280		; <ubyte*>:3080 [#uses=1]
	load ubyte* %3080		; <ubyte>:4867 [#uses=1]
	seteq ubyte %4867, 0		; <bool>:2058 [#uses=1]
	br bool %2058, label %2061, label %2060

; <label>:2059		; preds = %2035, %2061
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1284		; <ubyte*>:3081 [#uses=1]
	load ubyte* %3081		; <ubyte>:4868 [#uses=1]
	seteq ubyte %4868, 0		; <bool>:2059 [#uses=1]
	br bool %2059, label %2063, label %2062

; <label>:2060		; preds = %2058, %2060
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1280		; <ubyte*>:3082 [#uses=2]
	load ubyte* %3082		; <ubyte>:4869 [#uses=2]
	add ubyte %4869, 255		; <ubyte>:4870 [#uses=1]
	store ubyte %4870, ubyte* %3082
	seteq ubyte %4869, 1		; <bool>:2060 [#uses=1]
	br bool %2060, label %2061, label %2060

; <label>:2061		; preds = %2058, %2060
	add uint %1279, 110		; <uint>:1296 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1296		; <ubyte*>:3083 [#uses=2]
	load ubyte* %3083		; <ubyte>:4871 [#uses=1]
	add ubyte %4871, 1		; <ubyte>:4872 [#uses=1]
	store ubyte %4872, ubyte* %3083
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1280		; <ubyte*>:3084 [#uses=1]
	load ubyte* %3084		; <ubyte>:4873 [#uses=1]
	seteq ubyte %4873, 0		; <bool>:2061 [#uses=1]
	br bool %2061, label %2059, label %2058

; <label>:2062		; preds = %2059, %2065
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1284		; <ubyte*>:3085 [#uses=1]
	load ubyte* %3085		; <ubyte>:4874 [#uses=1]
	seteq ubyte %4874, 0		; <bool>:2062 [#uses=1]
	br bool %2062, label %2065, label %2064

; <label>:2063		; preds = %2059, %2065
	add uint %1279, 112		; <uint>:1297 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1297		; <ubyte*>:3086 [#uses=1]
	load ubyte* %3086		; <ubyte>:4875 [#uses=1]
	seteq ubyte %4875, 0		; <bool>:2063 [#uses=1]
	br bool %2063, label %2067, label %2066

; <label>:2064		; preds = %2062, %2064
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1284		; <ubyte*>:3087 [#uses=2]
	load ubyte* %3087		; <ubyte>:4876 [#uses=2]
	add ubyte %4876, 255		; <ubyte>:4877 [#uses=1]
	store ubyte %4877, ubyte* %3087
	seteq ubyte %4876, 1		; <bool>:2064 [#uses=1]
	br bool %2064, label %2065, label %2064

; <label>:2065		; preds = %2062, %2064
	add uint %1279, 112		; <uint>:1298 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1298		; <ubyte*>:3088 [#uses=2]
	load ubyte* %3088		; <ubyte>:4878 [#uses=1]
	add ubyte %4878, 1		; <ubyte>:4879 [#uses=1]
	store ubyte %4879, ubyte* %3088
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1284		; <ubyte*>:3089 [#uses=1]
	load ubyte* %3089		; <ubyte>:4880 [#uses=1]
	seteq ubyte %4880, 0		; <bool>:2065 [#uses=1]
	br bool %2065, label %2063, label %2062

; <label>:2066		; preds = %2063, %2066
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1297		; <ubyte*>:3090 [#uses=2]
	load ubyte* %3090		; <ubyte>:4881 [#uses=2]
	add ubyte %4881, 255		; <ubyte>:4882 [#uses=1]
	store ubyte %4882, ubyte* %3090
	seteq ubyte %4881, 1		; <bool>:2066 [#uses=1]
	br bool %2066, label %2067, label %2066

; <label>:2067		; preds = %2063, %2066
	add uint %1279, 110		; <uint>:1299 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1299		; <ubyte*>:3091 [#uses=1]
	load ubyte* %3091		; <ubyte>:4883 [#uses=1]
	seteq ubyte %4883, 0		; <bool>:2067 [#uses=1]
	br bool %2067, label %2069, label %2068

; <label>:2068		; preds = %2067, %2071
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1299		; <ubyte*>:3092 [#uses=1]
	load ubyte* %3092		; <ubyte>:4884 [#uses=1]
	seteq ubyte %4884, 0		; <bool>:2068 [#uses=1]
	br bool %2068, label %2071, label %2070

; <label>:2069		; preds = %2067, %2071
	add uint %1279, 98		; <uint>:1300 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1300		; <ubyte*>:3093 [#uses=2]
	load ubyte* %3093		; <ubyte>:4885 [#uses=1]
	add ubyte %4885, 8		; <ubyte>:4886 [#uses=1]
	store ubyte %4886, ubyte* %3093
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1280		; <ubyte*>:3094 [#uses=1]
	load ubyte* %3094		; <ubyte>:4887 [#uses=1]
	seteq ubyte %4887, 0		; <bool>:2069 [#uses=1]
	br bool %2069, label %2073, label %2072

; <label>:2070		; preds = %2068, %2070
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1299		; <ubyte*>:3095 [#uses=2]
	load ubyte* %3095		; <ubyte>:4888 [#uses=2]
	add ubyte %4888, 255		; <ubyte>:4889 [#uses=1]
	store ubyte %4889, ubyte* %3095
	seteq ubyte %4888, 1		; <bool>:2070 [#uses=1]
	br bool %2070, label %2071, label %2070

; <label>:2071		; preds = %2068, %2070
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1280		; <ubyte*>:3096 [#uses=2]
	load ubyte* %3096		; <ubyte>:4890 [#uses=1]
	add ubyte %4890, 1		; <ubyte>:4891 [#uses=1]
	store ubyte %4891, ubyte* %3096
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1299		; <ubyte*>:3097 [#uses=1]
	load ubyte* %3097		; <ubyte>:4892 [#uses=1]
	seteq ubyte %4892, 0		; <bool>:2071 [#uses=1]
	br bool %2071, label %2069, label %2068

; <label>:2072		; preds = %2069, %2075
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1280		; <ubyte*>:3098 [#uses=1]
	load ubyte* %3098		; <ubyte>:4893 [#uses=1]
	seteq ubyte %4893, 0		; <bool>:2072 [#uses=1]
	br bool %2072, label %2075, label %2074

; <label>:2073		; preds = %2069, %2075
	add uint %1279, 100		; <uint>:1301 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1301		; <ubyte*>:3099 [#uses=1]
	load ubyte* %3099		; <ubyte>:4894 [#uses=1]
	seteq ubyte %4894, 0		; <bool>:2073 [#uses=1]
	br bool %2073, label %1649, label %1648

; <label>:2074		; preds = %2072, %2074
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1280		; <ubyte*>:3100 [#uses=2]
	load ubyte* %3100		; <ubyte>:4895 [#uses=2]
	add ubyte %4895, 255		; <ubyte>:4896 [#uses=1]
	store ubyte %4896, ubyte* %3100
	seteq ubyte %4895, 1		; <bool>:2074 [#uses=1]
	br bool %2074, label %2075, label %2074

; <label>:2075		; preds = %2072, %2074
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1300		; <ubyte*>:3101 [#uses=2]
	load ubyte* %3101		; <ubyte>:4897 [#uses=1]
	add ubyte %4897, 1		; <ubyte>:4898 [#uses=1]
	store ubyte %4898, ubyte* %3101
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1280		; <ubyte*>:3102 [#uses=1]
	load ubyte* %3102		; <ubyte>:4899 [#uses=1]
	seteq ubyte %4899, 0		; <bool>:2075 [#uses=1]
	br bool %2075, label %2073, label %2072

; <label>:2076		; preds = %573, %2249
	phi uint [ %387, %573 ], [ %1410, %2249 ]		; <uint>:1302 [#uses=66]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1302		; <ubyte*>:3103 [#uses=2]
	load ubyte* %3103		; <ubyte>:4900 [#uses=1]
	add ubyte %4900, 255		; <ubyte>:4901 [#uses=1]
	store ubyte %4901, ubyte* %3103
	add uint %1302, 18		; <uint>:1303 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1303		; <ubyte*>:3104 [#uses=1]
	load ubyte* %3104		; <ubyte>:4902 [#uses=1]
	seteq ubyte %4902, 0		; <bool>:2076 [#uses=1]
	br bool %2076, label %2079, label %2078

; <label>:2077		; preds = %573, %2249
	phi uint [ %387, %573 ], [ %1410, %2249 ]		; <uint>:1304 [#uses=1]
	add uint %1304, 4294967295		; <uint>:1305 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1305		; <ubyte*>:3105 [#uses=1]
	load ubyte* %3105		; <ubyte>:4903 [#uses=1]
	seteq ubyte %4903, 0		; <bool>:2077 [#uses=1]
	br bool %2077, label %571, label %570

; <label>:2078		; preds = %2076, %2078
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1303		; <ubyte*>:3106 [#uses=2]
	load ubyte* %3106		; <ubyte>:4904 [#uses=2]
	add ubyte %4904, 255		; <ubyte>:4905 [#uses=1]
	store ubyte %4905, ubyte* %3106
	seteq ubyte %4904, 1		; <bool>:2078 [#uses=1]
	br bool %2078, label %2079, label %2078

; <label>:2079		; preds = %2076, %2078
	add uint %1302, 4294967201		; <uint>:1306 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1306		; <ubyte*>:3107 [#uses=1]
	load ubyte* %3107		; <ubyte>:4906 [#uses=1]
	seteq ubyte %4906, 0		; <bool>:2079 [#uses=1]
	br bool %2079, label %2081, label %2080

; <label>:2080		; preds = %2079, %2080
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1306		; <ubyte*>:3108 [#uses=2]
	load ubyte* %3108		; <ubyte>:4907 [#uses=1]
	add ubyte %4907, 255		; <ubyte>:4908 [#uses=1]
	store ubyte %4908, ubyte* %3108
	add uint %1302, 4294967202		; <uint>:1307 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1307		; <ubyte*>:3109 [#uses=2]
	load ubyte* %3109		; <ubyte>:4909 [#uses=1]
	add ubyte %4909, 1		; <ubyte>:4910 [#uses=1]
	store ubyte %4910, ubyte* %3109
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1303		; <ubyte*>:3110 [#uses=2]
	load ubyte* %3110		; <ubyte>:4911 [#uses=1]
	add ubyte %4911, 1		; <ubyte>:4912 [#uses=1]
	store ubyte %4912, ubyte* %3110
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1306		; <ubyte*>:3111 [#uses=1]
	load ubyte* %3111		; <ubyte>:4913 [#uses=1]
	seteq ubyte %4913, 0		; <bool>:2080 [#uses=1]
	br bool %2080, label %2081, label %2080

; <label>:2081		; preds = %2079, %2080
	add uint %1302, 4294967202		; <uint>:1308 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1308		; <ubyte*>:3112 [#uses=1]
	load ubyte* %3112		; <ubyte>:4914 [#uses=1]
	seteq ubyte %4914, 0		; <bool>:2081 [#uses=1]
	br bool %2081, label %2083, label %2082

; <label>:2082		; preds = %2081, %2082
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1306		; <ubyte*>:3113 [#uses=2]
	load ubyte* %3113		; <ubyte>:4915 [#uses=1]
	add ubyte %4915, 1		; <ubyte>:4916 [#uses=1]
	store ubyte %4916, ubyte* %3113
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1308		; <ubyte*>:3114 [#uses=2]
	load ubyte* %3114		; <ubyte>:4917 [#uses=2]
	add ubyte %4917, 255		; <ubyte>:4918 [#uses=1]
	store ubyte %4918, ubyte* %3114
	seteq ubyte %4917, 1		; <bool>:2082 [#uses=1]
	br bool %2082, label %2083, label %2082

; <label>:2083		; preds = %2081, %2082
	add uint %1302, 24		; <uint>:1309 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1309		; <ubyte*>:3115 [#uses=1]
	load ubyte* %3115		; <ubyte>:4919 [#uses=1]
	seteq ubyte %4919, 0		; <bool>:2083 [#uses=1]
	br bool %2083, label %2085, label %2084

; <label>:2084		; preds = %2083, %2084
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1309		; <ubyte*>:3116 [#uses=2]
	load ubyte* %3116		; <ubyte>:4920 [#uses=2]
	add ubyte %4920, 255		; <ubyte>:4921 [#uses=1]
	store ubyte %4921, ubyte* %3116
	seteq ubyte %4920, 1		; <bool>:2084 [#uses=1]
	br bool %2084, label %2085, label %2084

; <label>:2085		; preds = %2083, %2084
	add uint %1302, 4294967207		; <uint>:1310 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1310		; <ubyte*>:3117 [#uses=1]
	load ubyte* %3117		; <ubyte>:4922 [#uses=1]
	seteq ubyte %4922, 0		; <bool>:2085 [#uses=1]
	br bool %2085, label %2087, label %2086

; <label>:2086		; preds = %2085, %2086
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1310		; <ubyte*>:3118 [#uses=2]
	load ubyte* %3118		; <ubyte>:4923 [#uses=1]
	add ubyte %4923, 255		; <ubyte>:4924 [#uses=1]
	store ubyte %4924, ubyte* %3118
	add uint %1302, 4294967208		; <uint>:1311 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1311		; <ubyte*>:3119 [#uses=2]
	load ubyte* %3119		; <ubyte>:4925 [#uses=1]
	add ubyte %4925, 1		; <ubyte>:4926 [#uses=1]
	store ubyte %4926, ubyte* %3119
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1309		; <ubyte*>:3120 [#uses=2]
	load ubyte* %3120		; <ubyte>:4927 [#uses=1]
	add ubyte %4927, 1		; <ubyte>:4928 [#uses=1]
	store ubyte %4928, ubyte* %3120
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1310		; <ubyte*>:3121 [#uses=1]
	load ubyte* %3121		; <ubyte>:4929 [#uses=1]
	seteq ubyte %4929, 0		; <bool>:2086 [#uses=1]
	br bool %2086, label %2087, label %2086

; <label>:2087		; preds = %2085, %2086
	add uint %1302, 4294967208		; <uint>:1312 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1312		; <ubyte*>:3122 [#uses=1]
	load ubyte* %3122		; <ubyte>:4930 [#uses=1]
	seteq ubyte %4930, 0		; <bool>:2087 [#uses=1]
	br bool %2087, label %2089, label %2088

; <label>:2088		; preds = %2087, %2088
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1310		; <ubyte*>:3123 [#uses=2]
	load ubyte* %3123		; <ubyte>:4931 [#uses=1]
	add ubyte %4931, 1		; <ubyte>:4932 [#uses=1]
	store ubyte %4932, ubyte* %3123
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1312		; <ubyte*>:3124 [#uses=2]
	load ubyte* %3124		; <ubyte>:4933 [#uses=2]
	add ubyte %4933, 255		; <ubyte>:4934 [#uses=1]
	store ubyte %4934, ubyte* %3124
	seteq ubyte %4933, 1		; <bool>:2088 [#uses=1]
	br bool %2088, label %2089, label %2088

; <label>:2089		; preds = %2087, %2088
	add uint %1302, 30		; <uint>:1313 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1313		; <ubyte*>:3125 [#uses=1]
	load ubyte* %3125		; <ubyte>:4935 [#uses=1]
	seteq ubyte %4935, 0		; <bool>:2089 [#uses=1]
	br bool %2089, label %2091, label %2090

; <label>:2090		; preds = %2089, %2090
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1313		; <ubyte*>:3126 [#uses=2]
	load ubyte* %3126		; <ubyte>:4936 [#uses=2]
	add ubyte %4936, 255		; <ubyte>:4937 [#uses=1]
	store ubyte %4937, ubyte* %3126
	seteq ubyte %4936, 1		; <bool>:2090 [#uses=1]
	br bool %2090, label %2091, label %2090

; <label>:2091		; preds = %2089, %2090
	add uint %1302, 4294967213		; <uint>:1314 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1314		; <ubyte*>:3127 [#uses=1]
	load ubyte* %3127		; <ubyte>:4938 [#uses=1]
	seteq ubyte %4938, 0		; <bool>:2091 [#uses=1]
	br bool %2091, label %2093, label %2092

; <label>:2092		; preds = %2091, %2092
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1314		; <ubyte*>:3128 [#uses=2]
	load ubyte* %3128		; <ubyte>:4939 [#uses=1]
	add ubyte %4939, 255		; <ubyte>:4940 [#uses=1]
	store ubyte %4940, ubyte* %3128
	add uint %1302, 4294967214		; <uint>:1315 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1315		; <ubyte*>:3129 [#uses=2]
	load ubyte* %3129		; <ubyte>:4941 [#uses=1]
	add ubyte %4941, 1		; <ubyte>:4942 [#uses=1]
	store ubyte %4942, ubyte* %3129
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1313		; <ubyte*>:3130 [#uses=2]
	load ubyte* %3130		; <ubyte>:4943 [#uses=1]
	add ubyte %4943, 1		; <ubyte>:4944 [#uses=1]
	store ubyte %4944, ubyte* %3130
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1314		; <ubyte*>:3131 [#uses=1]
	load ubyte* %3131		; <ubyte>:4945 [#uses=1]
	seteq ubyte %4945, 0		; <bool>:2092 [#uses=1]
	br bool %2092, label %2093, label %2092

; <label>:2093		; preds = %2091, %2092
	add uint %1302, 4294967214		; <uint>:1316 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1316		; <ubyte*>:3132 [#uses=1]
	load ubyte* %3132		; <ubyte>:4946 [#uses=1]
	seteq ubyte %4946, 0		; <bool>:2093 [#uses=1]
	br bool %2093, label %2095, label %2094

; <label>:2094		; preds = %2093, %2094
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1314		; <ubyte*>:3133 [#uses=2]
	load ubyte* %3133		; <ubyte>:4947 [#uses=1]
	add ubyte %4947, 1		; <ubyte>:4948 [#uses=1]
	store ubyte %4948, ubyte* %3133
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1316		; <ubyte*>:3134 [#uses=2]
	load ubyte* %3134		; <ubyte>:4949 [#uses=2]
	add ubyte %4949, 255		; <ubyte>:4950 [#uses=1]
	store ubyte %4950, ubyte* %3134
	seteq ubyte %4949, 1		; <bool>:2094 [#uses=1]
	br bool %2094, label %2095, label %2094

; <label>:2095		; preds = %2093, %2094
	add uint %1302, 36		; <uint>:1317 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1317		; <ubyte*>:3135 [#uses=1]
	load ubyte* %3135		; <ubyte>:4951 [#uses=1]
	seteq ubyte %4951, 0		; <bool>:2095 [#uses=1]
	br bool %2095, label %2097, label %2096

; <label>:2096		; preds = %2095, %2096
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1317		; <ubyte*>:3136 [#uses=2]
	load ubyte* %3136		; <ubyte>:4952 [#uses=2]
	add ubyte %4952, 255		; <ubyte>:4953 [#uses=1]
	store ubyte %4953, ubyte* %3136
	seteq ubyte %4952, 1		; <bool>:2096 [#uses=1]
	br bool %2096, label %2097, label %2096

; <label>:2097		; preds = %2095, %2096
	add uint %1302, 4294967219		; <uint>:1318 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1318		; <ubyte*>:3137 [#uses=1]
	load ubyte* %3137		; <ubyte>:4954 [#uses=1]
	seteq ubyte %4954, 0		; <bool>:2097 [#uses=1]
	br bool %2097, label %2099, label %2098

; <label>:2098		; preds = %2097, %2098
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1318		; <ubyte*>:3138 [#uses=2]
	load ubyte* %3138		; <ubyte>:4955 [#uses=1]
	add ubyte %4955, 255		; <ubyte>:4956 [#uses=1]
	store ubyte %4956, ubyte* %3138
	add uint %1302, 4294967220		; <uint>:1319 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1319		; <ubyte*>:3139 [#uses=2]
	load ubyte* %3139		; <ubyte>:4957 [#uses=1]
	add ubyte %4957, 1		; <ubyte>:4958 [#uses=1]
	store ubyte %4958, ubyte* %3139
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1317		; <ubyte*>:3140 [#uses=2]
	load ubyte* %3140		; <ubyte>:4959 [#uses=1]
	add ubyte %4959, 1		; <ubyte>:4960 [#uses=1]
	store ubyte %4960, ubyte* %3140
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1318		; <ubyte*>:3141 [#uses=1]
	load ubyte* %3141		; <ubyte>:4961 [#uses=1]
	seteq ubyte %4961, 0		; <bool>:2098 [#uses=1]
	br bool %2098, label %2099, label %2098

; <label>:2099		; preds = %2097, %2098
	add uint %1302, 4294967220		; <uint>:1320 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1320		; <ubyte*>:3142 [#uses=1]
	load ubyte* %3142		; <ubyte>:4962 [#uses=1]
	seteq ubyte %4962, 0		; <bool>:2099 [#uses=1]
	br bool %2099, label %2101, label %2100

; <label>:2100		; preds = %2099, %2100
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1318		; <ubyte*>:3143 [#uses=2]
	load ubyte* %3143		; <ubyte>:4963 [#uses=1]
	add ubyte %4963, 1		; <ubyte>:4964 [#uses=1]
	store ubyte %4964, ubyte* %3143
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1320		; <ubyte*>:3144 [#uses=2]
	load ubyte* %3144		; <ubyte>:4965 [#uses=2]
	add ubyte %4965, 255		; <ubyte>:4966 [#uses=1]
	store ubyte %4966, ubyte* %3144
	seteq ubyte %4965, 1		; <bool>:2100 [#uses=1]
	br bool %2100, label %2101, label %2100

; <label>:2101		; preds = %2099, %2100
	add uint %1302, 42		; <uint>:1321 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1321		; <ubyte*>:3145 [#uses=1]
	load ubyte* %3145		; <ubyte>:4967 [#uses=1]
	seteq ubyte %4967, 0		; <bool>:2101 [#uses=1]
	br bool %2101, label %2103, label %2102

; <label>:2102		; preds = %2101, %2102
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1321		; <ubyte*>:3146 [#uses=2]
	load ubyte* %3146		; <ubyte>:4968 [#uses=2]
	add ubyte %4968, 255		; <ubyte>:4969 [#uses=1]
	store ubyte %4969, ubyte* %3146
	seteq ubyte %4968, 1		; <bool>:2102 [#uses=1]
	br bool %2102, label %2103, label %2102

; <label>:2103		; preds = %2101, %2102
	add uint %1302, 4294967225		; <uint>:1322 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1322		; <ubyte*>:3147 [#uses=1]
	load ubyte* %3147		; <ubyte>:4970 [#uses=1]
	seteq ubyte %4970, 0		; <bool>:2103 [#uses=1]
	br bool %2103, label %2105, label %2104

; <label>:2104		; preds = %2103, %2104
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1322		; <ubyte*>:3148 [#uses=2]
	load ubyte* %3148		; <ubyte>:4971 [#uses=1]
	add ubyte %4971, 255		; <ubyte>:4972 [#uses=1]
	store ubyte %4972, ubyte* %3148
	add uint %1302, 4294967226		; <uint>:1323 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1323		; <ubyte*>:3149 [#uses=2]
	load ubyte* %3149		; <ubyte>:4973 [#uses=1]
	add ubyte %4973, 1		; <ubyte>:4974 [#uses=1]
	store ubyte %4974, ubyte* %3149
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1321		; <ubyte*>:3150 [#uses=2]
	load ubyte* %3150		; <ubyte>:4975 [#uses=1]
	add ubyte %4975, 1		; <ubyte>:4976 [#uses=1]
	store ubyte %4976, ubyte* %3150
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1322		; <ubyte*>:3151 [#uses=1]
	load ubyte* %3151		; <ubyte>:4977 [#uses=1]
	seteq ubyte %4977, 0		; <bool>:2104 [#uses=1]
	br bool %2104, label %2105, label %2104

; <label>:2105		; preds = %2103, %2104
	add uint %1302, 4294967226		; <uint>:1324 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1324		; <ubyte*>:3152 [#uses=1]
	load ubyte* %3152		; <ubyte>:4978 [#uses=1]
	seteq ubyte %4978, 0		; <bool>:2105 [#uses=1]
	br bool %2105, label %2107, label %2106

; <label>:2106		; preds = %2105, %2106
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1322		; <ubyte*>:3153 [#uses=2]
	load ubyte* %3153		; <ubyte>:4979 [#uses=1]
	add ubyte %4979, 1		; <ubyte>:4980 [#uses=1]
	store ubyte %4980, ubyte* %3153
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1324		; <ubyte*>:3154 [#uses=2]
	load ubyte* %3154		; <ubyte>:4981 [#uses=2]
	add ubyte %4981, 255		; <ubyte>:4982 [#uses=1]
	store ubyte %4982, ubyte* %3154
	seteq ubyte %4981, 1		; <bool>:2106 [#uses=1]
	br bool %2106, label %2107, label %2106

; <label>:2107		; preds = %2105, %2106
	add uint %1302, 48		; <uint>:1325 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1325		; <ubyte*>:3155 [#uses=1]
	load ubyte* %3155		; <ubyte>:4983 [#uses=1]
	seteq ubyte %4983, 0		; <bool>:2107 [#uses=1]
	br bool %2107, label %2109, label %2108

; <label>:2108		; preds = %2107, %2108
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1325		; <ubyte*>:3156 [#uses=2]
	load ubyte* %3156		; <ubyte>:4984 [#uses=2]
	add ubyte %4984, 255		; <ubyte>:4985 [#uses=1]
	store ubyte %4985, ubyte* %3156
	seteq ubyte %4984, 1		; <bool>:2108 [#uses=1]
	br bool %2108, label %2109, label %2108

; <label>:2109		; preds = %2107, %2108
	add uint %1302, 4294967231		; <uint>:1326 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1326		; <ubyte*>:3157 [#uses=1]
	load ubyte* %3157		; <ubyte>:4986 [#uses=1]
	seteq ubyte %4986, 0		; <bool>:2109 [#uses=1]
	br bool %2109, label %2111, label %2110

; <label>:2110		; preds = %2109, %2110
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1326		; <ubyte*>:3158 [#uses=2]
	load ubyte* %3158		; <ubyte>:4987 [#uses=1]
	add ubyte %4987, 255		; <ubyte>:4988 [#uses=1]
	store ubyte %4988, ubyte* %3158
	add uint %1302, 4294967232		; <uint>:1327 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1327		; <ubyte*>:3159 [#uses=2]
	load ubyte* %3159		; <ubyte>:4989 [#uses=1]
	add ubyte %4989, 1		; <ubyte>:4990 [#uses=1]
	store ubyte %4990, ubyte* %3159
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1325		; <ubyte*>:3160 [#uses=2]
	load ubyte* %3160		; <ubyte>:4991 [#uses=1]
	add ubyte %4991, 1		; <ubyte>:4992 [#uses=1]
	store ubyte %4992, ubyte* %3160
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1326		; <ubyte*>:3161 [#uses=1]
	load ubyte* %3161		; <ubyte>:4993 [#uses=1]
	seteq ubyte %4993, 0		; <bool>:2110 [#uses=1]
	br bool %2110, label %2111, label %2110

; <label>:2111		; preds = %2109, %2110
	add uint %1302, 4294967232		; <uint>:1328 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1328		; <ubyte*>:3162 [#uses=1]
	load ubyte* %3162		; <ubyte>:4994 [#uses=1]
	seteq ubyte %4994, 0		; <bool>:2111 [#uses=1]
	br bool %2111, label %2113, label %2112

; <label>:2112		; preds = %2111, %2112
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1326		; <ubyte*>:3163 [#uses=2]
	load ubyte* %3163		; <ubyte>:4995 [#uses=1]
	add ubyte %4995, 1		; <ubyte>:4996 [#uses=1]
	store ubyte %4996, ubyte* %3163
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1328		; <ubyte*>:3164 [#uses=2]
	load ubyte* %3164		; <ubyte>:4997 [#uses=2]
	add ubyte %4997, 255		; <ubyte>:4998 [#uses=1]
	store ubyte %4998, ubyte* %3164
	seteq ubyte %4997, 1		; <bool>:2112 [#uses=1]
	br bool %2112, label %2113, label %2112

; <label>:2113		; preds = %2111, %2112
	add uint %1302, 54		; <uint>:1329 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1329		; <ubyte*>:3165 [#uses=1]
	load ubyte* %3165		; <ubyte>:4999 [#uses=1]
	seteq ubyte %4999, 0		; <bool>:2113 [#uses=1]
	br bool %2113, label %2115, label %2114

; <label>:2114		; preds = %2113, %2114
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1329		; <ubyte*>:3166 [#uses=2]
	load ubyte* %3166		; <ubyte>:5000 [#uses=2]
	add ubyte %5000, 255		; <ubyte>:5001 [#uses=1]
	store ubyte %5001, ubyte* %3166
	seteq ubyte %5000, 1		; <bool>:2114 [#uses=1]
	br bool %2114, label %2115, label %2114

; <label>:2115		; preds = %2113, %2114
	add uint %1302, 4294967237		; <uint>:1330 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1330		; <ubyte*>:3167 [#uses=1]
	load ubyte* %3167		; <ubyte>:5002 [#uses=1]
	seteq ubyte %5002, 0		; <bool>:2115 [#uses=1]
	br bool %2115, label %2117, label %2116

; <label>:2116		; preds = %2115, %2116
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1330		; <ubyte*>:3168 [#uses=2]
	load ubyte* %3168		; <ubyte>:5003 [#uses=1]
	add ubyte %5003, 255		; <ubyte>:5004 [#uses=1]
	store ubyte %5004, ubyte* %3168
	add uint %1302, 4294967238		; <uint>:1331 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1331		; <ubyte*>:3169 [#uses=2]
	load ubyte* %3169		; <ubyte>:5005 [#uses=1]
	add ubyte %5005, 1		; <ubyte>:5006 [#uses=1]
	store ubyte %5006, ubyte* %3169
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1329		; <ubyte*>:3170 [#uses=2]
	load ubyte* %3170		; <ubyte>:5007 [#uses=1]
	add ubyte %5007, 1		; <ubyte>:5008 [#uses=1]
	store ubyte %5008, ubyte* %3170
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1330		; <ubyte*>:3171 [#uses=1]
	load ubyte* %3171		; <ubyte>:5009 [#uses=1]
	seteq ubyte %5009, 0		; <bool>:2116 [#uses=1]
	br bool %2116, label %2117, label %2116

; <label>:2117		; preds = %2115, %2116
	add uint %1302, 4294967238		; <uint>:1332 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1332		; <ubyte*>:3172 [#uses=1]
	load ubyte* %3172		; <ubyte>:5010 [#uses=1]
	seteq ubyte %5010, 0		; <bool>:2117 [#uses=1]
	br bool %2117, label %2119, label %2118

; <label>:2118		; preds = %2117, %2118
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1330		; <ubyte*>:3173 [#uses=2]
	load ubyte* %3173		; <ubyte>:5011 [#uses=1]
	add ubyte %5011, 1		; <ubyte>:5012 [#uses=1]
	store ubyte %5012, ubyte* %3173
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1332		; <ubyte*>:3174 [#uses=2]
	load ubyte* %3174		; <ubyte>:5013 [#uses=2]
	add ubyte %5013, 255		; <ubyte>:5014 [#uses=1]
	store ubyte %5014, ubyte* %3174
	seteq ubyte %5013, 1		; <bool>:2118 [#uses=1]
	br bool %2118, label %2119, label %2118

; <label>:2119		; preds = %2117, %2118
	add uint %1302, 60		; <uint>:1333 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1333		; <ubyte*>:3175 [#uses=1]
	load ubyte* %3175		; <ubyte>:5015 [#uses=1]
	seteq ubyte %5015, 0		; <bool>:2119 [#uses=1]
	br bool %2119, label %2121, label %2120

; <label>:2120		; preds = %2119, %2120
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1333		; <ubyte*>:3176 [#uses=2]
	load ubyte* %3176		; <ubyte>:5016 [#uses=2]
	add ubyte %5016, 255		; <ubyte>:5017 [#uses=1]
	store ubyte %5017, ubyte* %3176
	seteq ubyte %5016, 1		; <bool>:2120 [#uses=1]
	br bool %2120, label %2121, label %2120

; <label>:2121		; preds = %2119, %2120
	add uint %1302, 4294967243		; <uint>:1334 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1334		; <ubyte*>:3177 [#uses=1]
	load ubyte* %3177		; <ubyte>:5018 [#uses=1]
	seteq ubyte %5018, 0		; <bool>:2121 [#uses=1]
	br bool %2121, label %2123, label %2122

; <label>:2122		; preds = %2121, %2122
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1334		; <ubyte*>:3178 [#uses=2]
	load ubyte* %3178		; <ubyte>:5019 [#uses=1]
	add ubyte %5019, 255		; <ubyte>:5020 [#uses=1]
	store ubyte %5020, ubyte* %3178
	add uint %1302, 4294967244		; <uint>:1335 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1335		; <ubyte*>:3179 [#uses=2]
	load ubyte* %3179		; <ubyte>:5021 [#uses=1]
	add ubyte %5021, 1		; <ubyte>:5022 [#uses=1]
	store ubyte %5022, ubyte* %3179
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1333		; <ubyte*>:3180 [#uses=2]
	load ubyte* %3180		; <ubyte>:5023 [#uses=1]
	add ubyte %5023, 1		; <ubyte>:5024 [#uses=1]
	store ubyte %5024, ubyte* %3180
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1334		; <ubyte*>:3181 [#uses=1]
	load ubyte* %3181		; <ubyte>:5025 [#uses=1]
	seteq ubyte %5025, 0		; <bool>:2122 [#uses=1]
	br bool %2122, label %2123, label %2122

; <label>:2123		; preds = %2121, %2122
	add uint %1302, 4294967244		; <uint>:1336 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1336		; <ubyte*>:3182 [#uses=1]
	load ubyte* %3182		; <ubyte>:5026 [#uses=1]
	seteq ubyte %5026, 0		; <bool>:2123 [#uses=1]
	br bool %2123, label %2125, label %2124

; <label>:2124		; preds = %2123, %2124
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1334		; <ubyte*>:3183 [#uses=2]
	load ubyte* %3183		; <ubyte>:5027 [#uses=1]
	add ubyte %5027, 1		; <ubyte>:5028 [#uses=1]
	store ubyte %5028, ubyte* %3183
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1336		; <ubyte*>:3184 [#uses=2]
	load ubyte* %3184		; <ubyte>:5029 [#uses=2]
	add ubyte %5029, 255		; <ubyte>:5030 [#uses=1]
	store ubyte %5030, ubyte* %3184
	seteq ubyte %5029, 1		; <bool>:2124 [#uses=1]
	br bool %2124, label %2125, label %2124

; <label>:2125		; preds = %2123, %2124
	add uint %1302, 66		; <uint>:1337 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1337		; <ubyte*>:3185 [#uses=1]
	load ubyte* %3185		; <ubyte>:5031 [#uses=1]
	seteq ubyte %5031, 0		; <bool>:2125 [#uses=1]
	br bool %2125, label %2127, label %2126

; <label>:2126		; preds = %2125, %2126
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1337		; <ubyte*>:3186 [#uses=2]
	load ubyte* %3186		; <ubyte>:5032 [#uses=2]
	add ubyte %5032, 255		; <ubyte>:5033 [#uses=1]
	store ubyte %5033, ubyte* %3186
	seteq ubyte %5032, 1		; <bool>:2126 [#uses=1]
	br bool %2126, label %2127, label %2126

; <label>:2127		; preds = %2125, %2126
	add uint %1302, 4294967249		; <uint>:1338 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1338		; <ubyte*>:3187 [#uses=1]
	load ubyte* %3187		; <ubyte>:5034 [#uses=1]
	seteq ubyte %5034, 0		; <bool>:2127 [#uses=1]
	br bool %2127, label %2129, label %2128

; <label>:2128		; preds = %2127, %2128
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1338		; <ubyte*>:3188 [#uses=2]
	load ubyte* %3188		; <ubyte>:5035 [#uses=1]
	add ubyte %5035, 255		; <ubyte>:5036 [#uses=1]
	store ubyte %5036, ubyte* %3188
	add uint %1302, 4294967250		; <uint>:1339 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1339		; <ubyte*>:3189 [#uses=2]
	load ubyte* %3189		; <ubyte>:5037 [#uses=1]
	add ubyte %5037, 1		; <ubyte>:5038 [#uses=1]
	store ubyte %5038, ubyte* %3189
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1337		; <ubyte*>:3190 [#uses=2]
	load ubyte* %3190		; <ubyte>:5039 [#uses=1]
	add ubyte %5039, 1		; <ubyte>:5040 [#uses=1]
	store ubyte %5040, ubyte* %3190
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1338		; <ubyte*>:3191 [#uses=1]
	load ubyte* %3191		; <ubyte>:5041 [#uses=1]
	seteq ubyte %5041, 0		; <bool>:2128 [#uses=1]
	br bool %2128, label %2129, label %2128

; <label>:2129		; preds = %2127, %2128
	add uint %1302, 4294967250		; <uint>:1340 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1340		; <ubyte*>:3192 [#uses=1]
	load ubyte* %3192		; <ubyte>:5042 [#uses=1]
	seteq ubyte %5042, 0		; <bool>:2129 [#uses=1]
	br bool %2129, label %2131, label %2130

; <label>:2130		; preds = %2129, %2130
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1338		; <ubyte*>:3193 [#uses=2]
	load ubyte* %3193		; <ubyte>:5043 [#uses=1]
	add ubyte %5043, 1		; <ubyte>:5044 [#uses=1]
	store ubyte %5044, ubyte* %3193
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1340		; <ubyte*>:3194 [#uses=2]
	load ubyte* %3194		; <ubyte>:5045 [#uses=2]
	add ubyte %5045, 255		; <ubyte>:5046 [#uses=1]
	store ubyte %5046, ubyte* %3194
	seteq ubyte %5045, 1		; <bool>:2130 [#uses=1]
	br bool %2130, label %2131, label %2130

; <label>:2131		; preds = %2129, %2130
	add uint %1302, 72		; <uint>:1341 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1341		; <ubyte*>:3195 [#uses=1]
	load ubyte* %3195		; <ubyte>:5047 [#uses=1]
	seteq ubyte %5047, 0		; <bool>:2131 [#uses=1]
	br bool %2131, label %2133, label %2132

; <label>:2132		; preds = %2131, %2132
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1341		; <ubyte*>:3196 [#uses=2]
	load ubyte* %3196		; <ubyte>:5048 [#uses=2]
	add ubyte %5048, 255		; <ubyte>:5049 [#uses=1]
	store ubyte %5049, ubyte* %3196
	seteq ubyte %5048, 1		; <bool>:2132 [#uses=1]
	br bool %2132, label %2133, label %2132

; <label>:2133		; preds = %2131, %2132
	add uint %1302, 4294967255		; <uint>:1342 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1342		; <ubyte*>:3197 [#uses=1]
	load ubyte* %3197		; <ubyte>:5050 [#uses=1]
	seteq ubyte %5050, 0		; <bool>:2133 [#uses=1]
	br bool %2133, label %2135, label %2134

; <label>:2134		; preds = %2133, %2134
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1342		; <ubyte*>:3198 [#uses=2]
	load ubyte* %3198		; <ubyte>:5051 [#uses=1]
	add ubyte %5051, 255		; <ubyte>:5052 [#uses=1]
	store ubyte %5052, ubyte* %3198
	add uint %1302, 4294967256		; <uint>:1343 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1343		; <ubyte*>:3199 [#uses=2]
	load ubyte* %3199		; <ubyte>:5053 [#uses=1]
	add ubyte %5053, 1		; <ubyte>:5054 [#uses=1]
	store ubyte %5054, ubyte* %3199
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1341		; <ubyte*>:3200 [#uses=2]
	load ubyte* %3200		; <ubyte>:5055 [#uses=1]
	add ubyte %5055, 1		; <ubyte>:5056 [#uses=1]
	store ubyte %5056, ubyte* %3200
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1342		; <ubyte*>:3201 [#uses=1]
	load ubyte* %3201		; <ubyte>:5057 [#uses=1]
	seteq ubyte %5057, 0		; <bool>:2134 [#uses=1]
	br bool %2134, label %2135, label %2134

; <label>:2135		; preds = %2133, %2134
	add uint %1302, 4294967256		; <uint>:1344 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1344		; <ubyte*>:3202 [#uses=1]
	load ubyte* %3202		; <ubyte>:5058 [#uses=1]
	seteq ubyte %5058, 0		; <bool>:2135 [#uses=1]
	br bool %2135, label %2137, label %2136

; <label>:2136		; preds = %2135, %2136
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1342		; <ubyte*>:3203 [#uses=2]
	load ubyte* %3203		; <ubyte>:5059 [#uses=1]
	add ubyte %5059, 1		; <ubyte>:5060 [#uses=1]
	store ubyte %5060, ubyte* %3203
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1344		; <ubyte*>:3204 [#uses=2]
	load ubyte* %3204		; <ubyte>:5061 [#uses=2]
	add ubyte %5061, 255		; <ubyte>:5062 [#uses=1]
	store ubyte %5062, ubyte* %3204
	seteq ubyte %5061, 1		; <bool>:2136 [#uses=1]
	br bool %2136, label %2137, label %2136

; <label>:2137		; preds = %2135, %2136
	add uint %1302, 78		; <uint>:1345 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1345		; <ubyte*>:3205 [#uses=1]
	load ubyte* %3205		; <ubyte>:5063 [#uses=1]
	seteq ubyte %5063, 0		; <bool>:2137 [#uses=1]
	br bool %2137, label %2139, label %2138

; <label>:2138		; preds = %2137, %2138
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1345		; <ubyte*>:3206 [#uses=2]
	load ubyte* %3206		; <ubyte>:5064 [#uses=2]
	add ubyte %5064, 255		; <ubyte>:5065 [#uses=1]
	store ubyte %5065, ubyte* %3206
	seteq ubyte %5064, 1		; <bool>:2138 [#uses=1]
	br bool %2138, label %2139, label %2138

; <label>:2139		; preds = %2137, %2138
	add uint %1302, 4294967261		; <uint>:1346 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1346		; <ubyte*>:3207 [#uses=1]
	load ubyte* %3207		; <ubyte>:5066 [#uses=1]
	seteq ubyte %5066, 0		; <bool>:2139 [#uses=1]
	br bool %2139, label %2141, label %2140

; <label>:2140		; preds = %2139, %2140
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1346		; <ubyte*>:3208 [#uses=2]
	load ubyte* %3208		; <ubyte>:5067 [#uses=1]
	add ubyte %5067, 255		; <ubyte>:5068 [#uses=1]
	store ubyte %5068, ubyte* %3208
	add uint %1302, 4294967262		; <uint>:1347 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1347		; <ubyte*>:3209 [#uses=2]
	load ubyte* %3209		; <ubyte>:5069 [#uses=1]
	add ubyte %5069, 1		; <ubyte>:5070 [#uses=1]
	store ubyte %5070, ubyte* %3209
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1345		; <ubyte*>:3210 [#uses=2]
	load ubyte* %3210		; <ubyte>:5071 [#uses=1]
	add ubyte %5071, 1		; <ubyte>:5072 [#uses=1]
	store ubyte %5072, ubyte* %3210
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1346		; <ubyte*>:3211 [#uses=1]
	load ubyte* %3211		; <ubyte>:5073 [#uses=1]
	seteq ubyte %5073, 0		; <bool>:2140 [#uses=1]
	br bool %2140, label %2141, label %2140

; <label>:2141		; preds = %2139, %2140
	add uint %1302, 4294967262		; <uint>:1348 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1348		; <ubyte*>:3212 [#uses=1]
	load ubyte* %3212		; <ubyte>:5074 [#uses=1]
	seteq ubyte %5074, 0		; <bool>:2141 [#uses=1]
	br bool %2141, label %2143, label %2142

; <label>:2142		; preds = %2141, %2142
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1346		; <ubyte*>:3213 [#uses=2]
	load ubyte* %3213		; <ubyte>:5075 [#uses=1]
	add ubyte %5075, 1		; <ubyte>:5076 [#uses=1]
	store ubyte %5076, ubyte* %3213
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1348		; <ubyte*>:3214 [#uses=2]
	load ubyte* %3214		; <ubyte>:5077 [#uses=2]
	add ubyte %5077, 255		; <ubyte>:5078 [#uses=1]
	store ubyte %5078, ubyte* %3214
	seteq ubyte %5077, 1		; <bool>:2142 [#uses=1]
	br bool %2142, label %2143, label %2142

; <label>:2143		; preds = %2141, %2142
	add uint %1302, 84		; <uint>:1349 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1349		; <ubyte*>:3215 [#uses=1]
	load ubyte* %3215		; <ubyte>:5079 [#uses=1]
	seteq ubyte %5079, 0		; <bool>:2143 [#uses=1]
	br bool %2143, label %2145, label %2144

; <label>:2144		; preds = %2143, %2144
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1349		; <ubyte*>:3216 [#uses=2]
	load ubyte* %3216		; <ubyte>:5080 [#uses=2]
	add ubyte %5080, 255		; <ubyte>:5081 [#uses=1]
	store ubyte %5081, ubyte* %3216
	seteq ubyte %5080, 1		; <bool>:2144 [#uses=1]
	br bool %2144, label %2145, label %2144

; <label>:2145		; preds = %2143, %2144
	add uint %1302, 4294967267		; <uint>:1350 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1350		; <ubyte*>:3217 [#uses=1]
	load ubyte* %3217		; <ubyte>:5082 [#uses=1]
	seteq ubyte %5082, 0		; <bool>:2145 [#uses=1]
	br bool %2145, label %2147, label %2146

; <label>:2146		; preds = %2145, %2146
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1350		; <ubyte*>:3218 [#uses=2]
	load ubyte* %3218		; <ubyte>:5083 [#uses=1]
	add ubyte %5083, 255		; <ubyte>:5084 [#uses=1]
	store ubyte %5084, ubyte* %3218
	add uint %1302, 4294967268		; <uint>:1351 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1351		; <ubyte*>:3219 [#uses=2]
	load ubyte* %3219		; <ubyte>:5085 [#uses=1]
	add ubyte %5085, 1		; <ubyte>:5086 [#uses=1]
	store ubyte %5086, ubyte* %3219
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1349		; <ubyte*>:3220 [#uses=2]
	load ubyte* %3220		; <ubyte>:5087 [#uses=1]
	add ubyte %5087, 1		; <ubyte>:5088 [#uses=1]
	store ubyte %5088, ubyte* %3220
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1350		; <ubyte*>:3221 [#uses=1]
	load ubyte* %3221		; <ubyte>:5089 [#uses=1]
	seteq ubyte %5089, 0		; <bool>:2146 [#uses=1]
	br bool %2146, label %2147, label %2146

; <label>:2147		; preds = %2145, %2146
	add uint %1302, 4294967268		; <uint>:1352 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1352		; <ubyte*>:3222 [#uses=1]
	load ubyte* %3222		; <ubyte>:5090 [#uses=1]
	seteq ubyte %5090, 0		; <bool>:2147 [#uses=1]
	br bool %2147, label %2149, label %2148

; <label>:2148		; preds = %2147, %2148
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1350		; <ubyte*>:3223 [#uses=2]
	load ubyte* %3223		; <ubyte>:5091 [#uses=1]
	add ubyte %5091, 1		; <ubyte>:5092 [#uses=1]
	store ubyte %5092, ubyte* %3223
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1352		; <ubyte*>:3224 [#uses=2]
	load ubyte* %3224		; <ubyte>:5093 [#uses=2]
	add ubyte %5093, 255		; <ubyte>:5094 [#uses=1]
	store ubyte %5094, ubyte* %3224
	seteq ubyte %5093, 1		; <bool>:2148 [#uses=1]
	br bool %2148, label %2149, label %2148

; <label>:2149		; preds = %2147, %2148
	add uint %1302, 90		; <uint>:1353 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1353		; <ubyte*>:3225 [#uses=1]
	load ubyte* %3225		; <ubyte>:5095 [#uses=1]
	seteq ubyte %5095, 0		; <bool>:2149 [#uses=1]
	br bool %2149, label %2151, label %2150

; <label>:2150		; preds = %2149, %2150
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1353		; <ubyte*>:3226 [#uses=2]
	load ubyte* %3226		; <ubyte>:5096 [#uses=2]
	add ubyte %5096, 255		; <ubyte>:5097 [#uses=1]
	store ubyte %5097, ubyte* %3226
	seteq ubyte %5096, 1		; <bool>:2150 [#uses=1]
	br bool %2150, label %2151, label %2150

; <label>:2151		; preds = %2149, %2150
	add uint %1302, 4294967273		; <uint>:1354 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1354		; <ubyte*>:3227 [#uses=1]
	load ubyte* %3227		; <ubyte>:5098 [#uses=1]
	seteq ubyte %5098, 0		; <bool>:2151 [#uses=1]
	br bool %2151, label %2153, label %2152

; <label>:2152		; preds = %2151, %2152
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1354		; <ubyte*>:3228 [#uses=2]
	load ubyte* %3228		; <ubyte>:5099 [#uses=1]
	add ubyte %5099, 255		; <ubyte>:5100 [#uses=1]
	store ubyte %5100, ubyte* %3228
	add uint %1302, 4294967274		; <uint>:1355 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1355		; <ubyte*>:3229 [#uses=2]
	load ubyte* %3229		; <ubyte>:5101 [#uses=1]
	add ubyte %5101, 1		; <ubyte>:5102 [#uses=1]
	store ubyte %5102, ubyte* %3229
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1353		; <ubyte*>:3230 [#uses=2]
	load ubyte* %3230		; <ubyte>:5103 [#uses=1]
	add ubyte %5103, 1		; <ubyte>:5104 [#uses=1]
	store ubyte %5104, ubyte* %3230
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1354		; <ubyte*>:3231 [#uses=1]
	load ubyte* %3231		; <ubyte>:5105 [#uses=1]
	seteq ubyte %5105, 0		; <bool>:2152 [#uses=1]
	br bool %2152, label %2153, label %2152

; <label>:2153		; preds = %2151, %2152
	add uint %1302, 4294967274		; <uint>:1356 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1356		; <ubyte*>:3232 [#uses=1]
	load ubyte* %3232		; <ubyte>:5106 [#uses=1]
	seteq ubyte %5106, 0		; <bool>:2153 [#uses=1]
	br bool %2153, label %2155, label %2154

; <label>:2154		; preds = %2153, %2154
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1354		; <ubyte*>:3233 [#uses=2]
	load ubyte* %3233		; <ubyte>:5107 [#uses=1]
	add ubyte %5107, 1		; <ubyte>:5108 [#uses=1]
	store ubyte %5108, ubyte* %3233
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1356		; <ubyte*>:3234 [#uses=2]
	load ubyte* %3234		; <ubyte>:5109 [#uses=2]
	add ubyte %5109, 255		; <ubyte>:5110 [#uses=1]
	store ubyte %5110, ubyte* %3234
	seteq ubyte %5109, 1		; <bool>:2154 [#uses=1]
	br bool %2154, label %2155, label %2154

; <label>:2155		; preds = %2153, %2154
	add uint %1302, 96		; <uint>:1357 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1357		; <ubyte*>:3235 [#uses=1]
	load ubyte* %3235		; <ubyte>:5111 [#uses=1]
	seteq ubyte %5111, 0		; <bool>:2155 [#uses=1]
	br bool %2155, label %2157, label %2156

; <label>:2156		; preds = %2155, %2156
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1357		; <ubyte*>:3236 [#uses=2]
	load ubyte* %3236		; <ubyte>:5112 [#uses=2]
	add ubyte %5112, 255		; <ubyte>:5113 [#uses=1]
	store ubyte %5113, ubyte* %3236
	seteq ubyte %5112, 1		; <bool>:2156 [#uses=1]
	br bool %2156, label %2157, label %2156

; <label>:2157		; preds = %2155, %2156
	add uint %1302, 4294967279		; <uint>:1358 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1358		; <ubyte*>:3237 [#uses=1]
	load ubyte* %3237		; <ubyte>:5114 [#uses=1]
	seteq ubyte %5114, 0		; <bool>:2157 [#uses=1]
	br bool %2157, label %2159, label %2158

; <label>:2158		; preds = %2157, %2158
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1358		; <ubyte*>:3238 [#uses=2]
	load ubyte* %3238		; <ubyte>:5115 [#uses=1]
	add ubyte %5115, 255		; <ubyte>:5116 [#uses=1]
	store ubyte %5116, ubyte* %3238
	add uint %1302, 4294967280		; <uint>:1359 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1359		; <ubyte*>:3239 [#uses=2]
	load ubyte* %3239		; <ubyte>:5117 [#uses=1]
	add ubyte %5117, 1		; <ubyte>:5118 [#uses=1]
	store ubyte %5118, ubyte* %3239
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1357		; <ubyte*>:3240 [#uses=2]
	load ubyte* %3240		; <ubyte>:5119 [#uses=1]
	add ubyte %5119, 1		; <ubyte>:5120 [#uses=1]
	store ubyte %5120, ubyte* %3240
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1358		; <ubyte*>:3241 [#uses=1]
	load ubyte* %3241		; <ubyte>:5121 [#uses=1]
	seteq ubyte %5121, 0		; <bool>:2158 [#uses=1]
	br bool %2158, label %2159, label %2158

; <label>:2159		; preds = %2157, %2158
	add uint %1302, 4294967280		; <uint>:1360 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1360		; <ubyte*>:3242 [#uses=1]
	load ubyte* %3242		; <ubyte>:5122 [#uses=1]
	seteq ubyte %5122, 0		; <bool>:2159 [#uses=1]
	br bool %2159, label %2161, label %2160

; <label>:2160		; preds = %2159, %2160
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1358		; <ubyte*>:3243 [#uses=2]
	load ubyte* %3243		; <ubyte>:5123 [#uses=1]
	add ubyte %5123, 1		; <ubyte>:5124 [#uses=1]
	store ubyte %5124, ubyte* %3243
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1360		; <ubyte*>:3244 [#uses=2]
	load ubyte* %3244		; <ubyte>:5125 [#uses=2]
	add ubyte %5125, 255		; <ubyte>:5126 [#uses=1]
	store ubyte %5126, ubyte* %3244
	seteq ubyte %5125, 1		; <bool>:2160 [#uses=1]
	br bool %2160, label %2161, label %2160

; <label>:2161		; preds = %2159, %2160
	add uint %1302, 102		; <uint>:1361 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1361		; <ubyte*>:3245 [#uses=1]
	load ubyte* %3245		; <ubyte>:5127 [#uses=1]
	seteq ubyte %5127, 0		; <bool>:2161 [#uses=1]
	br bool %2161, label %2163, label %2162

; <label>:2162		; preds = %2161, %2162
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1361		; <ubyte*>:3246 [#uses=2]
	load ubyte* %3246		; <ubyte>:5128 [#uses=2]
	add ubyte %5128, 255		; <ubyte>:5129 [#uses=1]
	store ubyte %5129, ubyte* %3246
	seteq ubyte %5128, 1		; <bool>:2162 [#uses=1]
	br bool %2162, label %2163, label %2162

; <label>:2163		; preds = %2161, %2162
	add uint %1302, 4294967285		; <uint>:1362 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1362		; <ubyte*>:3247 [#uses=1]
	load ubyte* %3247		; <ubyte>:5130 [#uses=1]
	seteq ubyte %5130, 0		; <bool>:2163 [#uses=1]
	br bool %2163, label %2165, label %2164

; <label>:2164		; preds = %2163, %2164
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1362		; <ubyte*>:3248 [#uses=2]
	load ubyte* %3248		; <ubyte>:5131 [#uses=1]
	add ubyte %5131, 255		; <ubyte>:5132 [#uses=1]
	store ubyte %5132, ubyte* %3248
	add uint %1302, 4294967286		; <uint>:1363 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1363		; <ubyte*>:3249 [#uses=2]
	load ubyte* %3249		; <ubyte>:5133 [#uses=1]
	add ubyte %5133, 1		; <ubyte>:5134 [#uses=1]
	store ubyte %5134, ubyte* %3249
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1361		; <ubyte*>:3250 [#uses=2]
	load ubyte* %3250		; <ubyte>:5135 [#uses=1]
	add ubyte %5135, 1		; <ubyte>:5136 [#uses=1]
	store ubyte %5136, ubyte* %3250
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1362		; <ubyte*>:3251 [#uses=1]
	load ubyte* %3251		; <ubyte>:5137 [#uses=1]
	seteq ubyte %5137, 0		; <bool>:2164 [#uses=1]
	br bool %2164, label %2165, label %2164

; <label>:2165		; preds = %2163, %2164
	add uint %1302, 4294967286		; <uint>:1364 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1364		; <ubyte*>:3252 [#uses=1]
	load ubyte* %3252		; <ubyte>:5138 [#uses=1]
	seteq ubyte %5138, 0		; <bool>:2165 [#uses=1]
	br bool %2165, label %2167, label %2166

; <label>:2166		; preds = %2165, %2166
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1362		; <ubyte*>:3253 [#uses=2]
	load ubyte* %3253		; <ubyte>:5139 [#uses=1]
	add ubyte %5139, 1		; <ubyte>:5140 [#uses=1]
	store ubyte %5140, ubyte* %3253
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1364		; <ubyte*>:3254 [#uses=2]
	load ubyte* %3254		; <ubyte>:5141 [#uses=2]
	add ubyte %5141, 255		; <ubyte>:5142 [#uses=1]
	store ubyte %5142, ubyte* %3254
	seteq ubyte %5141, 1		; <bool>:2166 [#uses=1]
	br bool %2166, label %2167, label %2166

; <label>:2167		; preds = %2165, %2166
	add uint %1302, 106		; <uint>:1365 [#uses=5]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1365		; <ubyte*>:3255 [#uses=1]
	load ubyte* %3255		; <ubyte>:5143 [#uses=1]
	seteq ubyte %5143, 0		; <bool>:2167 [#uses=1]
	br bool %2167, label %2169, label %2168

; <label>:2168		; preds = %2167, %2168
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1365		; <ubyte*>:3256 [#uses=2]
	load ubyte* %3256		; <ubyte>:5144 [#uses=2]
	add ubyte %5144, 255		; <ubyte>:5145 [#uses=1]
	store ubyte %5145, ubyte* %3256
	seteq ubyte %5144, 1		; <bool>:2168 [#uses=1]
	br bool %2168, label %2169, label %2168

; <label>:2169		; preds = %2167, %2168
	add uint %1302, 4294967191		; <uint>:1366 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1366		; <ubyte*>:3257 [#uses=1]
	load ubyte* %3257		; <ubyte>:5146 [#uses=1]
	seteq ubyte %5146, 0		; <bool>:2169 [#uses=1]
	br bool %2169, label %2171, label %2170

; <label>:2170		; preds = %2169, %2170
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1366		; <ubyte*>:3258 [#uses=2]
	load ubyte* %3258		; <ubyte>:5147 [#uses=1]
	add ubyte %5147, 255		; <ubyte>:5148 [#uses=1]
	store ubyte %5148, ubyte* %3258
	add uint %1302, 4294967192		; <uint>:1367 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1367		; <ubyte*>:3259 [#uses=2]
	load ubyte* %3259		; <ubyte>:5149 [#uses=1]
	add ubyte %5149, 1		; <ubyte>:5150 [#uses=1]
	store ubyte %5150, ubyte* %3259
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1365		; <ubyte*>:3260 [#uses=2]
	load ubyte* %3260		; <ubyte>:5151 [#uses=1]
	add ubyte %5151, 1		; <ubyte>:5152 [#uses=1]
	store ubyte %5152, ubyte* %3260
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1366		; <ubyte*>:3261 [#uses=1]
	load ubyte* %3261		; <ubyte>:5153 [#uses=1]
	seteq ubyte %5153, 0		; <bool>:2170 [#uses=1]
	br bool %2170, label %2171, label %2170

; <label>:2171		; preds = %2169, %2170
	add uint %1302, 4294967192		; <uint>:1368 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1368		; <ubyte*>:3262 [#uses=1]
	load ubyte* %3262		; <ubyte>:5154 [#uses=1]
	seteq ubyte %5154, 0		; <bool>:2171 [#uses=1]
	br bool %2171, label %2173, label %2172

; <label>:2172		; preds = %2171, %2172
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1366		; <ubyte*>:3263 [#uses=2]
	load ubyte* %3263		; <ubyte>:5155 [#uses=1]
	add ubyte %5155, 1		; <ubyte>:5156 [#uses=1]
	store ubyte %5156, ubyte* %3263
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1368		; <ubyte*>:3264 [#uses=2]
	load ubyte* %3264		; <ubyte>:5157 [#uses=2]
	add ubyte %5157, 255		; <ubyte>:5158 [#uses=1]
	store ubyte %5158, ubyte* %3264
	seteq ubyte %5157, 1		; <bool>:2172 [#uses=1]
	br bool %2172, label %2173, label %2172

; <label>:2173		; preds = %2171, %2172
	add uint %1302, 20		; <uint>:1369 [#uses=6]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1369		; <ubyte*>:3265 [#uses=1]
	load ubyte* %3265		; <ubyte>:5159 [#uses=1]
	seteq ubyte %5159, 0		; <bool>:2173 [#uses=1]
	br bool %2173, label %2175, label %2174

; <label>:2174		; preds = %2173, %2174
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1369		; <ubyte*>:3266 [#uses=2]
	load ubyte* %3266		; <ubyte>:5160 [#uses=2]
	add ubyte %5160, 255		; <ubyte>:5161 [#uses=1]
	store ubyte %5161, ubyte* %3266
	seteq ubyte %5160, 1		; <bool>:2174 [#uses=1]
	br bool %2174, label %2175, label %2174

; <label>:2175		; preds = %2173, %2174
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1365		; <ubyte*>:3267 [#uses=1]
	load ubyte* %3267		; <ubyte>:5162 [#uses=1]
	seteq ubyte %5162, 0		; <bool>:2175 [#uses=1]
	br bool %2175, label %2177, label %2176

; <label>:2176		; preds = %2175, %2176
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1369		; <ubyte*>:3268 [#uses=2]
	load ubyte* %3268		; <ubyte>:5163 [#uses=1]
	add ubyte %5163, 1		; <ubyte>:5164 [#uses=1]
	store ubyte %5164, ubyte* %3268
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1365		; <ubyte*>:3269 [#uses=2]
	load ubyte* %3269		; <ubyte>:5165 [#uses=2]
	add ubyte %5165, 255		; <ubyte>:5166 [#uses=1]
	store ubyte %5166, ubyte* %3269
	seteq ubyte %5165, 1		; <bool>:2176 [#uses=1]
	br bool %2176, label %2177, label %2176

; <label>:2177		; preds = %2175, %2176
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1369		; <ubyte*>:3270 [#uses=1]
	load ubyte* %3270		; <ubyte>:5167 [#uses=1]
	seteq ubyte %5167, 0		; <bool>:2177 [#uses=1]
	br bool %2177, label %2179, label %2178

; <label>:2178		; preds = %2177, %2181
	phi uint [ %1369, %2177 ], [ %1374, %2181 ]		; <uint>:1370 [#uses=6]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1370		; <ubyte*>:3271 [#uses=1]
	load ubyte* %3271		; <ubyte>:5168 [#uses=1]
	seteq ubyte %5168, 0		; <bool>:2178 [#uses=1]
	br bool %2178, label %2181, label %2180

; <label>:2179		; preds = %2177, %2181
	phi uint [ %1369, %2177 ], [ %1374, %2181 ]		; <uint>:1371 [#uses=7]
	add uint %1371, 4294967292		; <uint>:1372 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1372		; <ubyte*>:3272 [#uses=1]
	load ubyte* %3272		; <ubyte>:5169 [#uses=1]
	seteq ubyte %5169, 0		; <bool>:2179 [#uses=1]
	br bool %2179, label %2183, label %2182

; <label>:2180		; preds = %2178, %2180
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1370		; <ubyte*>:3273 [#uses=2]
	load ubyte* %3273		; <ubyte>:5170 [#uses=1]
	add ubyte %5170, 255		; <ubyte>:5171 [#uses=1]
	store ubyte %5171, ubyte* %3273
	add uint %1370, 6		; <uint>:1373 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1373		; <ubyte*>:3274 [#uses=2]
	load ubyte* %3274		; <ubyte>:5172 [#uses=1]
	add ubyte %5172, 1		; <ubyte>:5173 [#uses=1]
	store ubyte %5173, ubyte* %3274
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1370		; <ubyte*>:3275 [#uses=1]
	load ubyte* %3275		; <ubyte>:5174 [#uses=1]
	seteq ubyte %5174, 0		; <bool>:2180 [#uses=1]
	br bool %2180, label %2181, label %2180

; <label>:2181		; preds = %2178, %2180
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1370		; <ubyte*>:3276 [#uses=2]
	load ubyte* %3276		; <ubyte>:5175 [#uses=1]
	add ubyte %5175, 1		; <ubyte>:5176 [#uses=1]
	store ubyte %5176, ubyte* %3276
	add uint %1370, 6		; <uint>:1374 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1374		; <ubyte*>:3277 [#uses=2]
	load ubyte* %3277		; <ubyte>:5177 [#uses=2]
	add ubyte %5177, 255		; <ubyte>:5178 [#uses=1]
	store ubyte %5178, ubyte* %3277
	seteq ubyte %5177, 1		; <bool>:2181 [#uses=1]
	br bool %2181, label %2179, label %2178

; <label>:2182		; preds = %2179, %2182
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1372		; <ubyte*>:3278 [#uses=2]
	load ubyte* %3278		; <ubyte>:5179 [#uses=2]
	add ubyte %5179, 255		; <ubyte>:5180 [#uses=1]
	store ubyte %5180, ubyte* %3278
	seteq ubyte %5179, 1		; <bool>:2182 [#uses=1]
	br bool %2182, label %2183, label %2182

; <label>:2183		; preds = %2179, %2182
	add uint %1371, 4294967294		; <uint>:1375 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1375		; <ubyte*>:3279 [#uses=1]
	load ubyte* %3279		; <ubyte>:5181 [#uses=1]
	seteq ubyte %5181, 0		; <bool>:2183 [#uses=1]
	br bool %2183, label %2185, label %2184

; <label>:2184		; preds = %2183, %2184
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1372		; <ubyte*>:3280 [#uses=2]
	load ubyte* %3280		; <ubyte>:5182 [#uses=1]
	add ubyte %5182, 1		; <ubyte>:5183 [#uses=1]
	store ubyte %5183, ubyte* %3280
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1375		; <ubyte*>:3281 [#uses=2]
	load ubyte* %3281		; <ubyte>:5184 [#uses=1]
	add ubyte %5184, 255		; <ubyte>:5185 [#uses=1]
	store ubyte %5185, ubyte* %3281
	add uint %1371, 4294967295		; <uint>:1376 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1376		; <ubyte*>:3282 [#uses=2]
	load ubyte* %3282		; <ubyte>:5186 [#uses=1]
	add ubyte %5186, 1		; <ubyte>:5187 [#uses=1]
	store ubyte %5187, ubyte* %3282
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1375		; <ubyte*>:3283 [#uses=1]
	load ubyte* %3283		; <ubyte>:5188 [#uses=1]
	seteq ubyte %5188, 0		; <bool>:2184 [#uses=1]
	br bool %2184, label %2185, label %2184

; <label>:2185		; preds = %2183, %2184
	add uint %1371, 4294967295		; <uint>:1377 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1377		; <ubyte*>:3284 [#uses=1]
	load ubyte* %3284		; <ubyte>:5189 [#uses=1]
	seteq ubyte %5189, 0		; <bool>:2185 [#uses=1]
	br bool %2185, label %2187, label %2186

; <label>:2186		; preds = %2185, %2186
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1375		; <ubyte*>:3285 [#uses=2]
	load ubyte* %3285		; <ubyte>:5190 [#uses=1]
	add ubyte %5190, 1		; <ubyte>:5191 [#uses=1]
	store ubyte %5191, ubyte* %3285
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1377		; <ubyte*>:3286 [#uses=2]
	load ubyte* %3286		; <ubyte>:5192 [#uses=2]
	add ubyte %5192, 255		; <ubyte>:5193 [#uses=1]
	store ubyte %5193, ubyte* %3286
	seteq ubyte %5192, 1		; <bool>:2186 [#uses=1]
	br bool %2186, label %2187, label %2186

; <label>:2187		; preds = %2185, %2186
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1371		; <ubyte*>:3287 [#uses=2]
	load ubyte* %3287		; <ubyte>:5194 [#uses=2]
	add ubyte %5194, 1		; <ubyte>:5195 [#uses=1]
	store ubyte %5195, ubyte* %3287
	seteq ubyte %5194, 255		; <bool>:2187 [#uses=1]
	br bool %2187, label %2189, label %2188

; <label>:2188		; preds = %2187, %2193
	phi uint [ %1371, %2187 ], [ %1383, %2193 ]		; <uint>:1378 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1378		; <ubyte*>:3288 [#uses=2]
	load ubyte* %3288		; <ubyte>:5196 [#uses=1]
	add ubyte %5196, 255		; <ubyte>:5197 [#uses=1]
	store ubyte %5197, ubyte* %3288
	add uint %1378, 4294967286		; <uint>:1379 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1379		; <ubyte*>:3289 [#uses=1]
	load ubyte* %3289		; <ubyte>:5198 [#uses=1]
	seteq ubyte %5198, 0		; <bool>:2188 [#uses=1]
	br bool %2188, label %2191, label %2190

; <label>:2189		; preds = %2187, %2193
	phi uint [ %1371, %2187 ], [ %1383, %2193 ]		; <uint>:1380 [#uses=28]
	add uint %1380, 4		; <uint>:1381 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1381		; <ubyte*>:3290 [#uses=1]
	load ubyte* %3290		; <ubyte>:5199 [#uses=1]
	seteq ubyte %5199, 0		; <bool>:2189 [#uses=1]
	br bool %2189, label %2195, label %2194

; <label>:2190		; preds = %2188, %2190
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1379		; <ubyte*>:3291 [#uses=2]
	load ubyte* %3291		; <ubyte>:5200 [#uses=2]
	add ubyte %5200, 255		; <ubyte>:5201 [#uses=1]
	store ubyte %5201, ubyte* %3291
	seteq ubyte %5200, 1		; <bool>:2190 [#uses=1]
	br bool %2190, label %2191, label %2190

; <label>:2191		; preds = %2188, %2190
	add uint %1378, 4294967292		; <uint>:1382 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1382		; <ubyte*>:3292 [#uses=1]
	load ubyte* %3292		; <ubyte>:5202 [#uses=1]
	seteq ubyte %5202, 0		; <bool>:2191 [#uses=1]
	br bool %2191, label %2193, label %2192

; <label>:2192		; preds = %2191, %2192
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1379		; <ubyte*>:3293 [#uses=2]
	load ubyte* %3293		; <ubyte>:5203 [#uses=1]
	add ubyte %5203, 1		; <ubyte>:5204 [#uses=1]
	store ubyte %5204, ubyte* %3293
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1382		; <ubyte*>:3294 [#uses=2]
	load ubyte* %3294		; <ubyte>:5205 [#uses=2]
	add ubyte %5205, 255		; <ubyte>:5206 [#uses=1]
	store ubyte %5206, ubyte* %3294
	seteq ubyte %5205, 1		; <bool>:2192 [#uses=1]
	br bool %2192, label %2193, label %2192

; <label>:2193		; preds = %2191, %2192
	add uint %1378, 4294967290		; <uint>:1383 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1383		; <ubyte*>:3295 [#uses=1]
	load ubyte* %3295		; <ubyte>:5207 [#uses=1]
	seteq ubyte %5207, 0		; <bool>:2193 [#uses=1]
	br bool %2193, label %2189, label %2188

; <label>:2194		; preds = %2189, %2194
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1381		; <ubyte*>:3296 [#uses=2]
	load ubyte* %3296		; <ubyte>:5208 [#uses=2]
	add ubyte %5208, 255		; <ubyte>:5209 [#uses=1]
	store ubyte %5209, ubyte* %3296
	seteq ubyte %5208, 1		; <bool>:2194 [#uses=1]
	br bool %2194, label %2195, label %2194

; <label>:2195		; preds = %2189, %2194
	add uint %1380, 10		; <uint>:1384 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1384		; <ubyte*>:3297 [#uses=1]
	load ubyte* %3297		; <ubyte>:5210 [#uses=1]
	seteq ubyte %5210, 0		; <bool>:2195 [#uses=1]
	br bool %2195, label %2197, label %2196

; <label>:2196		; preds = %2195, %2196
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1384		; <ubyte*>:3298 [#uses=2]
	load ubyte* %3298		; <ubyte>:5211 [#uses=2]
	add ubyte %5211, 255		; <ubyte>:5212 [#uses=1]
	store ubyte %5212, ubyte* %3298
	seteq ubyte %5211, 1		; <bool>:2196 [#uses=1]
	br bool %2196, label %2197, label %2196

; <label>:2197		; preds = %2195, %2196
	add uint %1380, 16		; <uint>:1385 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1385		; <ubyte*>:3299 [#uses=1]
	load ubyte* %3299		; <ubyte>:5213 [#uses=1]
	seteq ubyte %5213, 0		; <bool>:2197 [#uses=1]
	br bool %2197, label %2199, label %2198

; <label>:2198		; preds = %2197, %2198
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1385		; <ubyte*>:3300 [#uses=2]
	load ubyte* %3300		; <ubyte>:5214 [#uses=2]
	add ubyte %5214, 255		; <ubyte>:5215 [#uses=1]
	store ubyte %5215, ubyte* %3300
	seteq ubyte %5214, 1		; <bool>:2198 [#uses=1]
	br bool %2198, label %2199, label %2198

; <label>:2199		; preds = %2197, %2198
	add uint %1380, 22		; <uint>:1386 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1386		; <ubyte*>:3301 [#uses=1]
	load ubyte* %3301		; <ubyte>:5216 [#uses=1]
	seteq ubyte %5216, 0		; <bool>:2199 [#uses=1]
	br bool %2199, label %2201, label %2200

; <label>:2200		; preds = %2199, %2200
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1386		; <ubyte*>:3302 [#uses=2]
	load ubyte* %3302		; <ubyte>:5217 [#uses=2]
	add ubyte %5217, 255		; <ubyte>:5218 [#uses=1]
	store ubyte %5218, ubyte* %3302
	seteq ubyte %5217, 1		; <bool>:2200 [#uses=1]
	br bool %2200, label %2201, label %2200

; <label>:2201		; preds = %2199, %2200
	add uint %1380, 28		; <uint>:1387 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1387		; <ubyte*>:3303 [#uses=1]
	load ubyte* %3303		; <ubyte>:5219 [#uses=1]
	seteq ubyte %5219, 0		; <bool>:2201 [#uses=1]
	br bool %2201, label %2203, label %2202

; <label>:2202		; preds = %2201, %2202
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1387		; <ubyte*>:3304 [#uses=2]
	load ubyte* %3304		; <ubyte>:5220 [#uses=2]
	add ubyte %5220, 255		; <ubyte>:5221 [#uses=1]
	store ubyte %5221, ubyte* %3304
	seteq ubyte %5220, 1		; <bool>:2202 [#uses=1]
	br bool %2202, label %2203, label %2202

; <label>:2203		; preds = %2201, %2202
	add uint %1380, 34		; <uint>:1388 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1388		; <ubyte*>:3305 [#uses=1]
	load ubyte* %3305		; <ubyte>:5222 [#uses=1]
	seteq ubyte %5222, 0		; <bool>:2203 [#uses=1]
	br bool %2203, label %2205, label %2204

; <label>:2204		; preds = %2203, %2204
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1388		; <ubyte*>:3306 [#uses=2]
	load ubyte* %3306		; <ubyte>:5223 [#uses=2]
	add ubyte %5223, 255		; <ubyte>:5224 [#uses=1]
	store ubyte %5224, ubyte* %3306
	seteq ubyte %5223, 1		; <bool>:2204 [#uses=1]
	br bool %2204, label %2205, label %2204

; <label>:2205		; preds = %2203, %2204
	add uint %1380, 40		; <uint>:1389 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1389		; <ubyte*>:3307 [#uses=1]
	load ubyte* %3307		; <ubyte>:5225 [#uses=1]
	seteq ubyte %5225, 0		; <bool>:2205 [#uses=1]
	br bool %2205, label %2207, label %2206

; <label>:2206		; preds = %2205, %2206
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1389		; <ubyte*>:3308 [#uses=2]
	load ubyte* %3308		; <ubyte>:5226 [#uses=2]
	add ubyte %5226, 255		; <ubyte>:5227 [#uses=1]
	store ubyte %5227, ubyte* %3308
	seteq ubyte %5226, 1		; <bool>:2206 [#uses=1]
	br bool %2206, label %2207, label %2206

; <label>:2207		; preds = %2205, %2206
	add uint %1380, 46		; <uint>:1390 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1390		; <ubyte*>:3309 [#uses=1]
	load ubyte* %3309		; <ubyte>:5228 [#uses=1]
	seteq ubyte %5228, 0		; <bool>:2207 [#uses=1]
	br bool %2207, label %2209, label %2208

; <label>:2208		; preds = %2207, %2208
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1390		; <ubyte*>:3310 [#uses=2]
	load ubyte* %3310		; <ubyte>:5229 [#uses=2]
	add ubyte %5229, 255		; <ubyte>:5230 [#uses=1]
	store ubyte %5230, ubyte* %3310
	seteq ubyte %5229, 1		; <bool>:2208 [#uses=1]
	br bool %2208, label %2209, label %2208

; <label>:2209		; preds = %2207, %2208
	add uint %1380, 52		; <uint>:1391 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1391		; <ubyte*>:3311 [#uses=1]
	load ubyte* %3311		; <ubyte>:5231 [#uses=1]
	seteq ubyte %5231, 0		; <bool>:2209 [#uses=1]
	br bool %2209, label %2211, label %2210

; <label>:2210		; preds = %2209, %2210
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1391		; <ubyte*>:3312 [#uses=2]
	load ubyte* %3312		; <ubyte>:5232 [#uses=2]
	add ubyte %5232, 255		; <ubyte>:5233 [#uses=1]
	store ubyte %5233, ubyte* %3312
	seteq ubyte %5232, 1		; <bool>:2210 [#uses=1]
	br bool %2210, label %2211, label %2210

; <label>:2211		; preds = %2209, %2210
	add uint %1380, 58		; <uint>:1392 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1392		; <ubyte*>:3313 [#uses=1]
	load ubyte* %3313		; <ubyte>:5234 [#uses=1]
	seteq ubyte %5234, 0		; <bool>:2211 [#uses=1]
	br bool %2211, label %2213, label %2212

; <label>:2212		; preds = %2211, %2212
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1392		; <ubyte*>:3314 [#uses=2]
	load ubyte* %3314		; <ubyte>:5235 [#uses=2]
	add ubyte %5235, 255		; <ubyte>:5236 [#uses=1]
	store ubyte %5236, ubyte* %3314
	seteq ubyte %5235, 1		; <bool>:2212 [#uses=1]
	br bool %2212, label %2213, label %2212

; <label>:2213		; preds = %2211, %2212
	add uint %1380, 64		; <uint>:1393 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1393		; <ubyte*>:3315 [#uses=1]
	load ubyte* %3315		; <ubyte>:5237 [#uses=1]
	seteq ubyte %5237, 0		; <bool>:2213 [#uses=1]
	br bool %2213, label %2215, label %2214

; <label>:2214		; preds = %2213, %2214
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1393		; <ubyte*>:3316 [#uses=2]
	load ubyte* %3316		; <ubyte>:5238 [#uses=2]
	add ubyte %5238, 255		; <ubyte>:5239 [#uses=1]
	store ubyte %5239, ubyte* %3316
	seteq ubyte %5238, 1		; <bool>:2214 [#uses=1]
	br bool %2214, label %2215, label %2214

; <label>:2215		; preds = %2213, %2214
	add uint %1380, 70		; <uint>:1394 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1394		; <ubyte*>:3317 [#uses=1]
	load ubyte* %3317		; <ubyte>:5240 [#uses=1]
	seteq ubyte %5240, 0		; <bool>:2215 [#uses=1]
	br bool %2215, label %2217, label %2216

; <label>:2216		; preds = %2215, %2216
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1394		; <ubyte*>:3318 [#uses=2]
	load ubyte* %3318		; <ubyte>:5241 [#uses=2]
	add ubyte %5241, 255		; <ubyte>:5242 [#uses=1]
	store ubyte %5242, ubyte* %3318
	seteq ubyte %5241, 1		; <bool>:2216 [#uses=1]
	br bool %2216, label %2217, label %2216

; <label>:2217		; preds = %2215, %2216
	add uint %1380, 76		; <uint>:1395 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1395		; <ubyte*>:3319 [#uses=1]
	load ubyte* %3319		; <ubyte>:5243 [#uses=1]
	seteq ubyte %5243, 0		; <bool>:2217 [#uses=1]
	br bool %2217, label %2219, label %2218

; <label>:2218		; preds = %2217, %2218
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1395		; <ubyte*>:3320 [#uses=2]
	load ubyte* %3320		; <ubyte>:5244 [#uses=2]
	add ubyte %5244, 255		; <ubyte>:5245 [#uses=1]
	store ubyte %5245, ubyte* %3320
	seteq ubyte %5244, 1		; <bool>:2218 [#uses=1]
	br bool %2218, label %2219, label %2218

; <label>:2219		; preds = %2217, %2218
	add uint %1380, 82		; <uint>:1396 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1396		; <ubyte*>:3321 [#uses=1]
	load ubyte* %3321		; <ubyte>:5246 [#uses=1]
	seteq ubyte %5246, 0		; <bool>:2219 [#uses=1]
	br bool %2219, label %2221, label %2220

; <label>:2220		; preds = %2219, %2220
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1396		; <ubyte*>:3322 [#uses=2]
	load ubyte* %3322		; <ubyte>:5247 [#uses=2]
	add ubyte %5247, 255		; <ubyte>:5248 [#uses=1]
	store ubyte %5248, ubyte* %3322
	seteq ubyte %5247, 1		; <bool>:2220 [#uses=1]
	br bool %2220, label %2221, label %2220

; <label>:2221		; preds = %2219, %2220
	add uint %1380, 88		; <uint>:1397 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1397		; <ubyte*>:3323 [#uses=1]
	load ubyte* %3323		; <ubyte>:5249 [#uses=1]
	seteq ubyte %5249, 0		; <bool>:2221 [#uses=1]
	br bool %2221, label %2223, label %2222

; <label>:2222		; preds = %2221, %2222
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1397		; <ubyte*>:3324 [#uses=2]
	load ubyte* %3324		; <ubyte>:5250 [#uses=2]
	add ubyte %5250, 255		; <ubyte>:5251 [#uses=1]
	store ubyte %5251, ubyte* %3324
	seteq ubyte %5250, 1		; <bool>:2222 [#uses=1]
	br bool %2222, label %2223, label %2222

; <label>:2223		; preds = %2221, %2222
	add uint %1380, 4294967284		; <uint>:1398 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1398		; <ubyte*>:3325 [#uses=1]
	load ubyte* %3325		; <ubyte>:5252 [#uses=1]
	seteq ubyte %5252, 0		; <bool>:2223 [#uses=1]
	br bool %2223, label %2225, label %2224

; <label>:2224		; preds = %2223, %2224
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1398		; <ubyte*>:3326 [#uses=2]
	load ubyte* %3326		; <ubyte>:5253 [#uses=2]
	add ubyte %5253, 255		; <ubyte>:5254 [#uses=1]
	store ubyte %5254, ubyte* %3326
	seteq ubyte %5253, 1		; <bool>:2224 [#uses=1]
	br bool %2224, label %2225, label %2224

; <label>:2225		; preds = %2223, %2224
	add uint %1380, 4294967292		; <uint>:1399 [#uses=13]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1399		; <ubyte*>:3327 [#uses=1]
	load ubyte* %3327		; <ubyte>:5255 [#uses=1]
	seteq ubyte %5255, 0		; <bool>:2225 [#uses=1]
	br bool %2225, label %2227, label %2226

; <label>:2226		; preds = %2225, %2226
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1398		; <ubyte*>:3328 [#uses=2]
	load ubyte* %3328		; <ubyte>:5256 [#uses=1]
	add ubyte %5256, 1		; <ubyte>:5257 [#uses=1]
	store ubyte %5257, ubyte* %3328
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1399		; <ubyte*>:3329 [#uses=2]
	load ubyte* %3329		; <ubyte>:5258 [#uses=2]
	add ubyte %5258, 255		; <ubyte>:5259 [#uses=1]
	store ubyte %5259, ubyte* %3329
	seteq ubyte %5258, 1		; <bool>:2226 [#uses=1]
	br bool %2226, label %2227, label %2226

; <label>:2227		; preds = %2225, %2226
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1399		; <ubyte*>:3330 [#uses=1]
	load ubyte* %3330		; <ubyte>:5260 [#uses=1]
	seteq ubyte %5260, 0		; <bool>:2227 [#uses=1]
	br bool %2227, label %2229, label %2228

; <label>:2228		; preds = %2227, %2228
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1399		; <ubyte*>:3331 [#uses=2]
	load ubyte* %3331		; <ubyte>:5261 [#uses=2]
	add ubyte %5261, 255		; <ubyte>:5262 [#uses=1]
	store ubyte %5262, ubyte* %3331
	seteq ubyte %5261, 1		; <bool>:2228 [#uses=1]
	br bool %2228, label %2229, label %2228

; <label>:2229		; preds = %2227, %2228
	add uint %1380, 4294967177		; <uint>:1400 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1400		; <ubyte*>:3332 [#uses=1]
	load ubyte* %3332		; <ubyte>:5263 [#uses=1]
	seteq ubyte %5263, 0		; <bool>:2229 [#uses=1]
	br bool %2229, label %2231, label %2230

; <label>:2230		; preds = %2229, %2230
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1400		; <ubyte*>:3333 [#uses=2]
	load ubyte* %3333		; <ubyte>:5264 [#uses=1]
	add ubyte %5264, 255		; <ubyte>:5265 [#uses=1]
	store ubyte %5265, ubyte* %3333
	add uint %1380, 4294967178		; <uint>:1401 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1401		; <ubyte*>:3334 [#uses=2]
	load ubyte* %3334		; <ubyte>:5266 [#uses=1]
	add ubyte %5266, 1		; <ubyte>:5267 [#uses=1]
	store ubyte %5267, ubyte* %3334
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1399		; <ubyte*>:3335 [#uses=2]
	load ubyte* %3335		; <ubyte>:5268 [#uses=1]
	add ubyte %5268, 1		; <ubyte>:5269 [#uses=1]
	store ubyte %5269, ubyte* %3335
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1400		; <ubyte*>:3336 [#uses=1]
	load ubyte* %3336		; <ubyte>:5270 [#uses=1]
	seteq ubyte %5270, 0		; <bool>:2230 [#uses=1]
	br bool %2230, label %2231, label %2230

; <label>:2231		; preds = %2229, %2230
	add uint %1380, 4294967178		; <uint>:1402 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1402		; <ubyte*>:3337 [#uses=1]
	load ubyte* %3337		; <ubyte>:5271 [#uses=1]
	seteq ubyte %5271, 0		; <bool>:2231 [#uses=1]
	br bool %2231, label %2233, label %2232

; <label>:2232		; preds = %2231, %2232
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1400		; <ubyte*>:3338 [#uses=2]
	load ubyte* %3338		; <ubyte>:5272 [#uses=1]
	add ubyte %5272, 1		; <ubyte>:5273 [#uses=1]
	store ubyte %5273, ubyte* %3338
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1402		; <ubyte*>:3339 [#uses=2]
	load ubyte* %3339		; <ubyte>:5274 [#uses=2]
	add ubyte %5274, 255		; <ubyte>:5275 [#uses=1]
	store ubyte %5275, ubyte* %3339
	seteq ubyte %5274, 1		; <bool>:2232 [#uses=1]
	br bool %2232, label %2233, label %2232

; <label>:2233		; preds = %2231, %2232
	add uint %1380, 4294967294		; <uint>:1403 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1403		; <ubyte*>:3340 [#uses=2]
	load ubyte* %3340		; <ubyte>:5276 [#uses=2]
	add ubyte %5276, 1		; <ubyte>:5277 [#uses=1]
	store ubyte %5277, ubyte* %3340
	seteq ubyte %5276, 255		; <bool>:2233 [#uses=1]
	br bool %2233, label %2235, label %2234

; <label>:2234		; preds = %2233, %2234
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1399		; <ubyte*>:3341 [#uses=2]
	load ubyte* %3341		; <ubyte>:5278 [#uses=1]
	add ubyte %5278, 1		; <ubyte>:5279 [#uses=1]
	store ubyte %5279, ubyte* %3341
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1403		; <ubyte*>:3342 [#uses=2]
	load ubyte* %3342		; <ubyte>:5280 [#uses=2]
	add ubyte %5280, 255		; <ubyte>:5281 [#uses=1]
	store ubyte %5281, ubyte* %3342
	seteq ubyte %5280, 1		; <bool>:2234 [#uses=1]
	br bool %2234, label %2235, label %2234

; <label>:2235		; preds = %2233, %2234
	add uint %1380, 4294967286		; <uint>:1404 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1404		; <ubyte*>:3343 [#uses=1]
	load ubyte* %3343		; <ubyte>:5282 [#uses=1]
	seteq ubyte %5282, 0		; <bool>:2235 [#uses=1]
	br bool %2235, label %2237, label %2236

; <label>:2236		; preds = %2235, %2236
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1404		; <ubyte*>:3344 [#uses=2]
	load ubyte* %3344		; <ubyte>:5283 [#uses=2]
	add ubyte %5283, 255		; <ubyte>:5284 [#uses=1]
	store ubyte %5284, ubyte* %3344
	seteq ubyte %5283, 1		; <bool>:2236 [#uses=1]
	br bool %2236, label %2237, label %2236

; <label>:2237		; preds = %2235, %2236
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1399		; <ubyte*>:3345 [#uses=1]
	load ubyte* %3345		; <ubyte>:5285 [#uses=1]
	seteq ubyte %5285, 0		; <bool>:2237 [#uses=1]
	br bool %2237, label %2239, label %2238

; <label>:2238		; preds = %2237, %2238
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1404		; <ubyte*>:3346 [#uses=2]
	load ubyte* %3346		; <ubyte>:5286 [#uses=1]
	add ubyte %5286, 1		; <ubyte>:5287 [#uses=1]
	store ubyte %5287, ubyte* %3346
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1399		; <ubyte*>:3347 [#uses=2]
	load ubyte* %3347		; <ubyte>:5288 [#uses=2]
	add ubyte %5288, 255		; <ubyte>:5289 [#uses=1]
	store ubyte %5289, ubyte* %3347
	seteq ubyte %5288, 1		; <bool>:2238 [#uses=1]
	br bool %2238, label %2239, label %2238

; <label>:2239		; preds = %2237, %2238
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1399		; <ubyte*>:3348 [#uses=1]
	load ubyte* %3348		; <ubyte>:5290 [#uses=1]
	seteq ubyte %5290, 0		; <bool>:2239 [#uses=1]
	br bool %2239, label %2241, label %2240

; <label>:2240		; preds = %2239, %2240
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1399		; <ubyte*>:3349 [#uses=2]
	load ubyte* %3349		; <ubyte>:5291 [#uses=2]
	add ubyte %5291, 255		; <ubyte>:5292 [#uses=1]
	store ubyte %5292, ubyte* %3349
	seteq ubyte %5291, 1		; <bool>:2240 [#uses=1]
	br bool %2240, label %2241, label %2240

; <label>:2241		; preds = %2239, %2240
	add uint %1380, 4294967175		; <uint>:1405 [#uses=4]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1405		; <ubyte*>:3350 [#uses=1]
	load ubyte* %3350		; <ubyte>:5293 [#uses=1]
	seteq ubyte %5293, 0		; <bool>:2241 [#uses=1]
	br bool %2241, label %2243, label %2242

; <label>:2242		; preds = %2241, %2242
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1405		; <ubyte*>:3351 [#uses=2]
	load ubyte* %3351		; <ubyte>:5294 [#uses=1]
	add ubyte %5294, 255		; <ubyte>:5295 [#uses=1]
	store ubyte %5295, ubyte* %3351
	add uint %1380, 4294967176		; <uint>:1406 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1406		; <ubyte*>:3352 [#uses=2]
	load ubyte* %3352		; <ubyte>:5296 [#uses=1]
	add ubyte %5296, 1		; <ubyte>:5297 [#uses=1]
	store ubyte %5297, ubyte* %3352
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1399		; <ubyte*>:3353 [#uses=2]
	load ubyte* %3353		; <ubyte>:5298 [#uses=1]
	add ubyte %5298, 1		; <ubyte>:5299 [#uses=1]
	store ubyte %5299, ubyte* %3353
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1405		; <ubyte*>:3354 [#uses=1]
	load ubyte* %3354		; <ubyte>:5300 [#uses=1]
	seteq ubyte %5300, 0		; <bool>:2242 [#uses=1]
	br bool %2242, label %2243, label %2242

; <label>:2243		; preds = %2241, %2242
	add uint %1380, 4294967176		; <uint>:1407 [#uses=2]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1407		; <ubyte*>:3355 [#uses=1]
	load ubyte* %3355		; <ubyte>:5301 [#uses=1]
	seteq ubyte %5301, 0		; <bool>:2243 [#uses=1]
	br bool %2243, label %2245, label %2244

; <label>:2244		; preds = %2243, %2244
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1405		; <ubyte*>:3356 [#uses=2]
	load ubyte* %3356		; <ubyte>:5302 [#uses=1]
	add ubyte %5302, 1		; <ubyte>:5303 [#uses=1]
	store ubyte %5303, ubyte* %3356
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1407		; <ubyte*>:3357 [#uses=2]
	load ubyte* %3357		; <ubyte>:5304 [#uses=2]
	add ubyte %5304, 255		; <ubyte>:5305 [#uses=1]
	store ubyte %5305, ubyte* %3357
	seteq ubyte %5304, 1		; <bool>:2244 [#uses=1]
	br bool %2244, label %2245, label %2244

; <label>:2245		; preds = %2243, %2244
	add uint %1380, 4294967288		; <uint>:1408 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1408		; <ubyte*>:3358 [#uses=1]
	load ubyte* %3358		; <ubyte>:5306 [#uses=1]
	seteq ubyte %5306, 0		; <bool>:2245 [#uses=1]
	br bool %2245, label %2247, label %2246

; <label>:2246		; preds = %2245, %2246
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1408		; <ubyte*>:3359 [#uses=2]
	load ubyte* %3359		; <ubyte>:5307 [#uses=2]
	add ubyte %5307, 255		; <ubyte>:5308 [#uses=1]
	store ubyte %5308, ubyte* %3359
	seteq ubyte %5307, 1		; <bool>:2246 [#uses=1]
	br bool %2246, label %2247, label %2246

; <label>:2247		; preds = %2245, %2246
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1399		; <ubyte*>:3360 [#uses=1]
	load ubyte* %3360		; <ubyte>:5309 [#uses=1]
	seteq ubyte %5309, 0		; <bool>:2247 [#uses=1]
	br bool %2247, label %2249, label %2248

; <label>:2248		; preds = %2247, %2248
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1408		; <ubyte*>:3361 [#uses=2]
	load ubyte* %3361		; <ubyte>:5310 [#uses=1]
	add ubyte %5310, 1		; <ubyte>:5311 [#uses=1]
	store ubyte %5311, ubyte* %3361
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1399		; <ubyte*>:3362 [#uses=2]
	load ubyte* %3362		; <ubyte>:5312 [#uses=2]
	add ubyte %5312, 255		; <ubyte>:5313 [#uses=1]
	store ubyte %5313, ubyte* %3362
	seteq ubyte %5312, 1		; <bool>:2248 [#uses=1]
	br bool %2248, label %2249, label %2248

; <label>:2249		; preds = %2247, %2248
	add uint %1380, 4294967281		; <uint>:1409 [#uses=1]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1409		; <ubyte*>:3363 [#uses=2]
	load ubyte* %3363		; <ubyte>:5314 [#uses=1]
	add ubyte %5314, 3		; <ubyte>:5315 [#uses=1]
	store ubyte %5315, ubyte* %3363
	add uint %1380, 4294967283		; <uint>:1410 [#uses=3]
	getelementptr [262144 x ubyte]* %bfarray, int 0, uint %1410		; <ubyte*>:3364 [#uses=1]
	load ubyte* %3364		; <ubyte>:5316 [#uses=1]
	seteq ubyte %5316, 0		; <bool>:2249 [#uses=1]
	br bool %2249, label %2077, label %2076
}

