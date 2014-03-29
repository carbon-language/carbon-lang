; RUN: llc < %s -mtriple=arm64-apple-ios7.0 | FileCheck %s
; RUN: llc < %s -mtriple=arm64-linux-gnu | FileCheck %s --check-prefix=CHECK-LINUX

; CHECK-LABEL: main:
; CHECK:	stp	fp, lr, [sp, #-16]!
; CHECK-NEXT:	mov	fp, sp
; CHECK-NEXT:	sub	sp, sp, #16
; CHECK-NEXT:	stur	wzr, [fp, #-4]
; CHECK:	adrp	x0, L_.str@PAGE
; CHECK:	add	x0, x0, L_.str@PAGEOFF
; CHECK-NEXT:	bl	_puts
; CHECK-NEXT:	mov	sp, fp
; CHECK-NEXT:	ldp	fp, lr, [sp], #16
; CHECK-NEXT:	ret

; CHECK-LINUX-LABEL: main:
; CHECK-LINUX:	stp	fp, lr, [sp, #-16]!
; CHECK-LINUX-NEXT:	mov	fp, sp
; CHECK-LINUX-NEXT:	sub	sp, sp, #16
; CHECK-LINUX-NEXT:	stur	wzr, [fp, #-4]
; CHECK-LINUX:	adrp	x0, .L.str
; CHECK-LINUX:	add	x0, x0, :lo12:.L.str
; CHECK-LINUX-NEXT:	bl	puts
; CHECK-LINUX-NEXT:	mov	sp, fp
; CHECK-LINUX-NEXT:	ldp	fp, lr, [sp], #16
; CHECK-LINUX-NEXT:	ret

@.str = private unnamed_addr constant [7 x i8] c"hello\0A\00"

define i32 @main() nounwind ssp {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  %call = call i32 @puts(i8* getelementptr inbounds ([7 x i8]* @.str, i32 0, i32 0))
  ret i32 %call
}

declare i32 @puts(i8*)
