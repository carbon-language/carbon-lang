; RUN: as < %s | opt -adce -simplifycfg | dis	

%FILE = type { int, ubyte*, ubyte*, ubyte, ubyte, uint, uint, uint }
	%spec_fd_t = type { int, int, int, ubyte* }
%__iob = uninitialized global [20 x %FILE]		; <[20 x %FILE]*> [#uses=1]
%dbglvl = global int 4		; <int*> [#uses=3]
%spec_fd = uninitialized global [3 x %spec_fd_t]		; <[3 x %spec_fd_t]*> [#uses=4]
%.LC9 = internal global [34 x sbyte] c"spec_read: fd=%d, > MAX_SPEC_FD!\0A\00"		; <[34 x sbyte]*> [#uses=1]
%.LC10 = internal global [4 x sbyte] c"EOF\00"		; <[4 x sbyte]*> [#uses=1]
%.LC11 = internal global [4 x sbyte] c"%d\0A\00"		; <[4 x sbyte]*> [#uses=1]
%.LC12 = internal global [17 x sbyte] c"spec_getc: %d = \00"		; <[17 x sbyte]*> [#uses=1]

implementation   ; Functions:

declare int "fprintf"(%FILE*, sbyte*, ...)

declare void "exit"(int)

declare int "remove"(sbyte*)

declare int "fputc"(int, %FILE*)

declare uint "fwrite"(sbyte*, uint, uint, %FILE*)

declare void "perror"(sbyte*)

int "spec_getc"(int %fd) {
; <label>:0					;[#uses=0]
	%reg109 = load int* %dbglvl		; <int> [#uses=1]
	%cond266 = setle int %reg109, 4		; <bool> [#uses=1]
	br bool %cond266, label %bb3, label %bb2

bb2:					;[#uses=1]
	%cast273 = getelementptr [17 x sbyte]* %.LC12, long 0, long 0		; <sbyte*> [#uses=0]
	br label %bb3

bb3:					;[#uses=2]
	%cond267 = setle int %fd, 3		; <bool> [#uses=1]
;	br bool %cond267, label %bb5, label %bb4
	br label %bb5

bb4:					;[#uses=2]
	%reg111 = getelementptr [20 x %FILE]* %__iob, long 0, long 1, ubyte 3		; <ubyte*> [#uses=1]
	%cast274 = getelementptr [34 x sbyte]* %.LC9, long 0, long 0		; <sbyte*> [#uses=0]
	%cast282 = cast ubyte* %reg111 to %FILE*		; <%FILE*> [#uses=0]
	call void %exit( int 1 )
	br label %UnifiedExitNode

bb5:					;[#uses=1]
	%reg107-idxcast1 = cast int %fd to long		; <long> [#uses=2]
	%reg107-idxcast2 = cast int %fd to long		; <long> [#uses=1]
	%reg1311 = getelementptr [3 x %spec_fd_t]* %spec_fd, long 0, long %reg107-idxcast2		; <%spec_fd_t*> [#uses=1]
	%idx1 = getelementptr [3 x %spec_fd_t]* %spec_fd, long 0, long %reg107-idxcast1, ubyte 2		; <int> [#uses=3]
	%reg1321 = load int* %idx1
	%idx2 = getelementptr %spec_fd_t* %reg1311, long 0, ubyte 1		; <int> [#uses=1]
	%reg1331 = load int* %idx2
	%cond270 = setlt int %reg1321, %reg1331		; <bool> [#uses=1]
	br bool %cond270, label %bb9, label %bb6

bb6:					;[#uses=1]
	%reg134 = load int* %dbglvl		; <int> [#uses=1]
	%cond271 = setle int %reg134, 4		; <bool> [#uses=1]
	br bool %cond271, label %bb8, label %bb7

bb7:					;[#uses=1]
	%cast277 = getelementptr [4 x sbyte]* %.LC10, long 0, long 0		; <sbyte*> [#uses=0]
	br label %bb8

bb8:					;[#uses=3]
	br label %UnifiedExitNode

bb9:					;[#uses=1]
	%reg107-idxcast3 = cast int %fd to long		; <uint> [#uses=1]
	%idx3 = getelementptr [3 x %spec_fd_t]* %spec_fd, long 0, long %reg107-idxcast3, ubyte 3		; <ubyte*> [#uses=1]
	%reg1601 = load ubyte** %idx3
	%reg132-idxcast1 = cast int %reg1321 to long		; <long> [#uses=1]
	%idx4 = getelementptr ubyte* %reg1601, long %reg132-idxcast1		; <ubyte> [#uses=2]
	%reg1621 = load ubyte* %idx4
	%cast108 = cast ubyte %reg1621 to long		; <long> [#uses=0]
	%reg157 = add int %reg1321, 1		; <int> [#uses=1]
	%idx5 = getelementptr [3 x %spec_fd_t]* %spec_fd, long 0, long %reg107-idxcast1, ubyte 2
	store int %reg157, int* %idx5
	%reg163 = load int* %dbglvl		; <int> [#uses=1]
	%cond272 = setle int %reg163, 4		; <bool> [#uses=1]
	br bool %cond272, label %bb11, label %bb10

bb10:					;[#uses=1]
	%cast279 = getelementptr [4 x sbyte]* %.LC11, long 0, long 0		; <sbyte*> [#uses=0]
	br label %bb11

bb11:					;[#uses=3]
	%cast291 = cast ubyte %reg1621 to int		; <int> [#uses=1]
	br label %UnifiedExitNode

UnifiedExitNode:					;[#uses=3]
	%UnifiedRetVal = phi int [ 42, %bb4 ], [ -1, %bb8 ], [ %cast291, %bb11 ]		; <int> [#uses=1]
	ret int %UnifiedRetVal
}

declare int "puts"(sbyte*)

declare int "printf"(sbyte*, ...)
