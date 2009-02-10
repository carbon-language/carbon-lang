; RUN: llvm-as < %s | llc -mtriple=arm-apple-darwin | grep ldrb
; RUN: llvm-as < %s | llc -mtriple=arm-apple-darwin | grep ldrh
; This used to look for ldmia. But it's no longer lucky enough to
; have the load / store instructions lined up just right after
; scheduler change for pr3457. We'll look for a robust solution
; later.

	%struct.x = type { i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 }
@src = external global %struct.x
@dst = external global %struct.x

define i32 @t() {
entry:
	call void @llvm.memcpy.i32( i8* getelementptr (%struct.x* @dst, i32 0, i32 0), i8* getelementptr (%struct.x* @src, i32 0, i32 0), i32 11, i32 8 )
	ret i32 0
}

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32)
