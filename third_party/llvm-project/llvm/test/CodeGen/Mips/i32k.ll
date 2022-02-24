; RUN: llc -mtriple=mipsel-linux-gnu -march=mipsel -mattr=mips16 -relocation-model=pic -mips16-constant-islands=false -O3 < %s | FileCheck %s -check-prefix=16

@.str = private unnamed_addr constant [4 x i8] c"%i\0A\00", align 1

define i32 @main() nounwind {
entry:
  %call = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i32 0, i32 0), i32 1075344593) nounwind
; 16:	lw	${{[0-9]+}}, 1f
; 16:	b	2f
; 16:	.align	2
; 16: 1: 	.word	1075344593
; 16: 2:

  %call1 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i32 0, i32 0), i32 -1075344593) nounwind

; 16:	lw	${{[0-9]+}}, 1f
; 16:	b	2f
; 16:	.align	2
; 16: 1: 	.word	-1075344593
; 16: 2:
  ret i32 0
}

declare i32 @printf(i8* nocapture, ...) nounwind
