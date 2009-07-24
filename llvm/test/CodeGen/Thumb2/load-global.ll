; RUN: llvm-as < %s | \
; RUN:   llc -mtriple=thumbv7-apple-darwin -relocation-model=static | FileCheck %s -check-prefix=STATIC
; RUN: llvm-as < %s | \
; RUN:   llc -mtriple=thumbv7-apple-darwin -relocation-model=dynamic-no-pic | FileCheck %s -check-prefix=DYNAMIC
; RUN: llvm-as < %s | \
; RUN:   llc -mtriple=thumbv7-apple-darwin -relocation-model=pic | FileCheck %s -check-prefix=PIC
; RUN: llvm-as < %s | \
; RUN:   llc -mtriple=thumbv7-linux-gnueabi -relocation-model=pic | FileCheck %s -check-prefix=LINUX

@G = external global i32

define i32 @test1() {
; STATIC: _test1:
; STATIC: .long _G

; DYNAMIC: _test1:
; DYNAMIC: .long L_G$non_lazy_ptr

; PIC: _test1
; PIC: add r0, r0, pc
; PIC: .long L_G$non_lazy_ptr-(LPC0+4)

; LINUX: test1
; LINUX: .long G(GOT)
	%tmp = load i32* @G
	ret i32 %tmp
}
