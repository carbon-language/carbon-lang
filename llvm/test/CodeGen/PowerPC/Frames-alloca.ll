; RUN: llvm-as < %s | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 | FileCheck %s -check-prefix=PPC32
; RUN: llvm-as < %s | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 | FileCheck %s -check-prefix=PPC64
; RUN: llvm-as < %s | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | FileCheck %s -check-prefix=PPC32-NOFP
; RUN: llvm-as < %s | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | FileCheck %s -check-prefix=PPC64-NOFP
; RUN: llvm-as < %s | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -enable-ppc32-regscavenger | FileCheck %s -check-prefix=PPC32
; RUN: llvm-as < %s | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -enable-ppc32-regscavenger | FileCheck %s -check-prefix=PPC32-RS
; RUN: llvm-as < %s | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -disable-fp-elim -enable-ppc32-regscavenger | FileCheck %s -check-prefix=PPC32-RS-NOFP

; CHECK-PPC32: stw r31, 20(r1)
; CHECK-PPC32: lwz r1, 0(r1)
; CHECK-PPC32: lwz r31, 20(r1)
; CHECK-PPC32-NOFP: stw r31, 20(r1)
; CHECK-PPC32-NOFP: lwz r1, 0(r1)
; CHECK-PPC32-NOFP: lwz r31, 20(r1)
; CHECK-PPC32-RS: stwu r1, -80(r1)
; CHECK-PPC32-RS-NOFP: stwu r1, -80(r1)

; CHECK-PPC64: std r31, 40(r1)
; CHECK-PPC64: stdu r1, -112(r1)
; CHECK-PPC64: ld r1, 0(r1)
; CHECK-PPC64: ld r31, 40(r1)
; CHECK-PPC64-NOFP: std r31, 40(r1)
; CHECK-PPC64-NOFP: stdu r1, -112(r1)
; CHECK-PPC64-NOFP: ld r1, 0(r1)
; CHECK-PPC64-NOFP: ld r31, 40(r1)

define i32* @f1(i32 %n) {
	%tmp = alloca i32, i32 %n		; <i32*> [#uses=1]
	ret i32* %tmp
}
