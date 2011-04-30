; RUN: llc < %s -mtriple=armv6-apple-darwin -regalloc=linearscan | FileCheck %s -check-prefix=V6
; RUN: llc < %s -mtriple=armv5-apple-darwin -regalloc=linearscan | FileCheck %s -check-prefix=V5
; RUN: llc < %s -mtriple=armv6-eabi -regalloc=linearscan | FileCheck %s -check-prefix=EABI
; rdar://r6949835

; Magic ARM pair hints works best with linearscan.

@b = external global i64*

define i64 @t(i64 %a) nounwind readonly {
entry:
;V6:   ldrd r2, r3, [r2]

;V5:   ldr r{{[0-9]+}}, [r2]
;V5:   ldr r{{[0-9]+}}, [r2, #4]

;EABI: ldr r{{[0-9]+}}, [r2]
;EABI: ldr r{{[0-9]+}}, [r2, #4]

	%0 = load i64** @b, align 4
	%1 = load i64* %0, align 4
	%2 = mul i64 %1, %a
	ret i64 %2
}
