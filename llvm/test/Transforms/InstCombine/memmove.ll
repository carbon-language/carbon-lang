; This test makes sure that memmove instructions are properly eliminated.
;
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:    not grep {call void @llvm.memmove}

@S = internal constant [33 x i8] c"panic: restorelist inconsistency\00"		; <[33 x i8]*> [#uses=1]

declare void @llvm.memmove.i32(i8*, i8*, i32, i32)

define void @test1(i8* %A, i8* %B, i32 %N) {
	call void @llvm.memmove.i32( i8* %A, i8* %B, i32 0, i32 1 )
	ret void
}

define void @test2(i8* %A, i32 %N) {
        ;; dest can't alias source since we can't write to source!
	call void @llvm.memmove.i32( i8* %A, i8* getelementptr ([33 x i8]* @S, i32 0, i32 0), i32 %N, i32 1 )
	ret void
}
