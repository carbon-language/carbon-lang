; Test to make sure that the 'private' is used correctly.
;
; RUN: llc < %s -mtriple=powerpc-unknown-linux-gnu | \
; RUN: FileCheck --check-prefix=LINUX %s
;
; RUN: llc < %s -mtriple=powerpc-apple-darwin | \
; RUN: FileCheck --check-prefix=OSX %s

; LINUX: .Lfoo:
; OSX: L_foo:
define private void @foo() nounwind {
        ret void
}

define i32 @bar() nounwind {
; LINUX: bl{{.*}}.Lfoo
; OSX: bl{{.*}}L_foo
        call void @foo()

; LINUX: lis{{.*}}.Lbaz
; OSX:  lis{{.*}}L_baz
	%1 = load i32* @baz, align 4
        ret i32 %1
}

; LINUX: .Lbaz:
; OSX: L_baz:
@baz = private global i32 4
