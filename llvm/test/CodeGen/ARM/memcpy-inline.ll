; RUN: llc < %s -mtriple=thumbv7-apple-darwin -regalloc=linearscan -disable-post-ra | FileCheck %s

; The ARM magic hinting works best with linear scan.
; CHECK: ldrd
; CHECK: strd
; CHECK: ldrb

%struct.x = type { i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 }
@src = external global %struct.x
@dst = external global %struct.x

define i32 @t() {
entry:
	call void @llvm.memcpy.i32( i8* getelementptr (%struct.x* @dst, i32 0, i32 0), i8* getelementptr (%struct.x* @src, i32 0, i32 0), i32 11, i32 8 )
	ret i32 0
}

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32)
