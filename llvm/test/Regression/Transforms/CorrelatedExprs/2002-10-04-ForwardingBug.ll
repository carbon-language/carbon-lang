; Testcase distilled from bisort where we tried to perform branch target 
; forwarding where it was not safe.
; RUN: llvm-as < %s | opt -cee
;
	%HANDLE = type { int, %HANDLE*, %HANDLE* }
	%struct.node = type { int, %HANDLE*, %HANDLE* }
%.LC0 = internal global [11 x sbyte] c"%d @ 0x%x\0A\00"		; <[11 x sbyte]*> [#uses=1]

implementation   ; Functions:

void %InOrder(%HANDLE* %h) {
bb0:		; No predecessors!
	br label %bb2

bb2:		; preds = %bb3, %bb0
	%reg113 = phi %HANDLE* [ %reg109, %bb3 ], [ %h, %bb0 ]		; <%HANDLE*> [#uses=4]
	%cond217 = seteq %HANDLE* %reg113, null		; <bool> [#uses=1]
	br bool %cond217, label %bb4, label %bb3

bb3:		; preds = %bb2
	%reg221 = getelementptr %HANDLE* %reg113, long 0, ubyte 1		; <%HANDLE**> [#uses=1]
	%reg108 = load %HANDLE** %reg221		; <%HANDLE*> [#uses=1]
	%reg226 = getelementptr %HANDLE* %reg113, long 0, ubyte 2		; <%HANDLE**> [#uses=1]
	%reg109 = load %HANDLE** %reg226		; <%HANDLE*> [#uses=1]
	call void %InOrder( %HANDLE* %reg108 )
	%cast231 = getelementptr %HANDLE* %reg113, long 0, ubyte 0		; <int*> [#uses=1]
	%reg111 = load int* %cast231		; <int> [#uses=1]
	%reg233 = call int (sbyte*, ...)* %printf( sbyte* getelementptr ([11 x sbyte]* %.LC0, long 0, long 0), int %reg111, uint 0 )		; <int> [#uses=0]
	br label %bb2

bb4:		; preds = %bb2
	ret void
}

declare int %printf(sbyte*, ...)
