; RUN: llc -mtriple="powerpc-unknown-linux-gnu" < %s | FileCheck %s --check-prefix=PPC32
; RUN: llc -mtriple="powerpc64-unknown-linux-gnu" < %s | FileCheck %s --check-prefix=PPC64
; PR15332

define void @regalloc() nounwind {
entry:
	%0 = add i32 1, 2
	ret void
}
; PPC32-LABEL: regalloc:
; PPC32-NOT: stwu 1, -{{[0-9]+}}(1)
; PPC32: blr

; PPC64-LABEL: regalloc:
; PPC64-NOT: stdu 1, -{{[0-9]+}}(1)
; PPC64: blr

define void @smallstack() nounwind {
entry:
	%0 = alloca i8, i32 4
	ret void
}
; PPC32-LABEL: smallstack:
; PPC32: stwu 1, -16(1)

; PPC64-LABEL: smallstack:
; PPC64-NOT: stdu 1, -{{[0-9]+}}(1)
; PPC64: blr

define void @bigstack() nounwind {
entry:
	%0 = alloca i8, i32 230
	ret void
}
; PPC32-LABEL: bigstack:
; PPC32: stwu 1, -240(1)

; PPC64-LABEL: bigstack:
; PPC64: stdu 1, -352(1)
