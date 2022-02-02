; RUN: llc < %s -mtriple=i386-apple-darwin -relocation-model=static | FileCheck -check-prefix X86STA %s
; RUN: llc < %s -mtriple=i386-apple-darwin -relocation-model=pic | FileCheck -check-prefix X86PIC %s
; RUN: llc < %s -mtriple=i386-pc-linux -relocation-model=dynamic-no-pic | FileCheck -check-prefix X86DYN %s
; RUN: llc < %s -mtriple=i386-pc-win32 -relocation-model=static | FileCheck -check-prefix X86WINSTA %s

; Call to immediate is not safe on x86-64 unless we *know* that the
; call will be within 32-bits pcrel from the dest immediate.

; RUN: llc < %s -mtriple=x86_64-- | FileCheck -check-prefix X64 %s

; PR3666
; PR3773
; rdar://6904453

define i32 @main() nounwind {
entry:
	%0 = call i32 inttoptr (i32 12345678 to i32 (i32)*)(i32 0) nounwind		; <i32> [#uses=1]
	ret i32 %0
}

; X86STA: {{call.*12345678}}
; X86PIC-NOT: {{call.*12345678}}
; X86DYN: {{call.*12345678}}
; X86WINSTA: {{call.*[*]%eax}}
; X64: {{call.*[*]%rax}}
