; RUN: llc < %s -march=xcore > %t1.s
; RUN: grep "bl memmove" %t1.s | count 1
; RUN: grep "ldc r., 8" %t1.s | count 1

; Unaligned load / store pair. Should be combined into a memmove
; of size 8
define void @f(i64* %dst, i64* %src) nounwind {
entry:
	%0 = load i64* %src, align 1
	store i64 %0, i64* %dst, align 1
	ret void
}
