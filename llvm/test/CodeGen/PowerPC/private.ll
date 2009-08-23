; Test to make sure that the 'private' is used correctly.
;
; RUN: llvm-as < %s | llc -mtriple=powerpc-unknown-linux-gnu -march=ppc32 | FileCheck %s -check-prefix=LINUX
; RUN: llvm-as < %s | llc -mtriple=powerpc-apple-darwin -march=ppc32 | FileCheck %s -check-prefix=DARWIN

define private void @foo() nounwind {
        ret void
; LINUX: .Lfoo:

; DARWIN: L_foo:
}

define i32 @bar() nounwind {
        call void @foo()
	%1 = load i32* @baz, align 4
        ret i32 %1
; LINUX: bar:
; LINUX: bl .Lfoo
; LINUX: lis 3, .Lbaz@ha
; LINUX: lwz 3, .Lbaz@l(3)

; DARWIN: _bar:
; DARWIN: bl L_foo
; DARWIN: lis r2, ha16(L_baz)
; DARWIN: lwz r3, lo16(L_baz)(r2)
}


; LINUX: .Lbaz:
; DARWIN: L_baz:
@baz = private global i32 4

