; RUN: llc < %s -mtriple=thumbv7-apple-ios -mcpu=cortex-a8 -regalloc=fast | FileCheck %s -check-prefix=A8
; RUN: llc < %s -mtriple=thumbv7-apple-ios -mcpu=cortex-m3 -regalloc=fast | FileCheck %s -check-prefix=M3
; rdar://6949835

; Magic ARM pair hints works best with linearscan / fast.

; Cortex-M3 errata 602117: LDRD with base in list may result in incorrect base
; register when interrupted or faulted.

@b = external global i64*

define i64 @t(i64 %a) nounwind readonly {
entry:
; A8: t:
; A8:   ldrd r2, r3, [r2]

; M3: t:
; M3-NOT: ldrd
; M3: ldm.w r2, {r2, r3}

	%0 = load i64** @b, align 4
	%1 = load i64* %0, align 4
	%2 = mul i64 %1, %a
	ret i64 %2
}
