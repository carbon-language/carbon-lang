; RUN: llvm-as < %s | llc -march=xcore > %t1.s
; RUN: grep "bl __misaligned_load" %t1.s | count 1

; Byte aligned load. Expands to call to __misaligned_load.
define i32 @align1(i32* %p) nounwind {
entry:
	%0 = load i32* %p, align 1		; <i32> [#uses=1]
	ret i32 %0
}
