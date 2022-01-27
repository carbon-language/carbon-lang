; RUN: llc < %s -emulated-tls -mcpu=generic -mtriple=i386-linux-gnu -relocation-model=pic \
; RUN:   | FileCheck -check-prefix=X86 %s
; RUN: llc < %s -emulated-tls -mcpu=generic -mtriple=x86_64-linux-gnu -relocation-model=pic \
; RUN:   | FileCheck -check-prefix=X64 %s
; RUN: llc < %s -emulated-tls -mcpu=generic -mtriple=i386-linux-android -relocation-model=pic \
; RUN:   | FileCheck -check-prefix=X86 %s
; RUN: llc < %s -emulated-tls -mcpu=generic -mtriple=x86_64-linux-android -relocation-model=pic \
; RUN:   | FileCheck -check-prefix=X64 %s

; RUN: llc < %s -mcpu=generic -mtriple=i386-linux-gnu -relocation-model=pic \
; RUN:   | FileCheck -check-prefix=NoEMU %s
; RUN: llc < %s -mcpu=generic -mtriple=x86_64-linux-gnu -relocation-model=pic \
; RUN:   | FileCheck -check-prefix=NoEMU %s
; RUN: llc < %s -mcpu=generic -mtriple=i386-linux-android -relocation-model=pic \
; RUN:   | FileCheck -check-prefix=X86 %s
; RUN: llc < %s -mcpu=generic -mtriple=x86_64-linux-android -relocation-model=pic \
; RUN:   | FileCheck -check-prefix=X64 %s

; NoEMU-NOT: __emutls

; Use my_emutls_get_address like __emutls_get_address.
@my_emutls_v_xyz = external global i8*, align 4
declare i8* @my_emutls_get_address(i8*)

define dso_local i32 @my_get_xyz() {
; X86-LABEL: my_get_xyz:
; X86:      movl my_emutls_v_xyz@GOT(%ebx), %eax
; X86-NEXT: movl %eax, (%esp)
; X86-NEXT: calll my_emutls_get_address@PLT
; X86-NEXT: movl (%eax), %eax
; X86-NEXT: addl $8, %esp
; X86-NEXT: .cfi_def_cfa_offset 8
; X86-NEXT: popl %ebx
; X86-NEXT: .cfi_def_cfa_offset 4
; X86-NEXT: retl
; X64-LABEL: my_get_xyz:
; X64:      movq my_emutls_v_xyz@GOTPCREL(%rip), %rdi
; X64-NEXT: callq my_emutls_get_address@PLT
; X64-NEXT: movl (%rax), %eax
; X64-NEXT: popq %rcx
; X64-NEXT: .cfi_def_cfa_offset 8
; X64-NEXT: retq

entry:
  %call = call i8* @my_emutls_get_address(i8* bitcast (i8** @my_emutls_v_xyz to i8*))
  %0 = bitcast i8* %call to i32*
  %1 = load i32, i32* %0, align 4
  ret i32 %1
}

@i = dso_local thread_local global i32 15
@i2 = external thread_local global i32

define dso_local i32 @f1() {
; X86-LABEL: f1:
; X86:      leal __emutls_v.i@GOTOFF(%ebx), %eax
; X86-NEXT: movl %eax, (%esp)
; X86-NEXT: calll __emutls_get_address@PLT
; X86-NEXT: movl (%eax), %eax
; X86-NEXT: addl $8, %esp
; X86-NEXT: .cfi_def_cfa_offset 8
; X86-NEXT: popl %ebx
; X86-NEXT: .cfi_def_cfa_offset 4
; X86-NEXT: retl
; X64-LABEL: f1:
; X64:      leaq __emutls_v.i(%rip), %rdi
; X64-NEXT: callq __emutls_get_address@PLT
; X64-NEXT: movl (%rax), %eax
; X64-NEXT: popq %rcx
; X64-NEXT: .cfi_def_cfa_offset 8
; X64-NEXT: retq

entry:
  %tmp1 = load i32, i32* @i
  ret i32 %tmp1
}

define dso_local i32* @f2() {
; X86-LABEL: f2:
; X86:      leal __emutls_v.i@GOTOFF(%ebx), %eax
; X86-NEXT: movl %eax, (%esp)
; X86-NEXT: calll __emutls_get_address@PLT
; X64-LABEL: f2:
; X64:      leaq __emutls_v.i(%rip), %rdi
; X64-NEXT: callq __emutls_get_address@PLT

entry:
  ret i32* @i
}

define dso_local i32 @f3() {
; X86-LABEL: f3:
; X86:      movl __emutls_v.i2@GOT(%ebx), %eax
; X86-NEXT: movl %eax, (%esp)
; X86-NEXT: calll __emutls_get_address@PLT
; X64-LABEL: f3:
; X64:      movq __emutls_v.i2@GOTPCREL(%rip), %rdi
; X64-NEXT: callq __emutls_get_address@PLT

entry:
  %tmp1 = load i32, i32* @i2
  ret i32 %tmp1
}

define dso_local i32* @f4() {
; X86-LABEL: f4:
; X86:      movl __emutls_v.i2@GOT(%ebx), %eax
; X86-NEXT: movl %eax, (%esp)
; X86-NEXT: calll __emutls_get_address@PLT
; X64-LABEL: f4:
; X64:      movq __emutls_v.i2@GOTPCREL(%rip), %rdi
; X64-NEXT: callq __emutls_get_address@PLT

entry:
  ret i32* @i2
}

;;;;; 32-bit targets

; X86:      .data
; X86-LABEL: __emutls_v.i:
; X86-NEXT: .long 4
; X86-NEXT: .long 4
; X86-NEXT: .long 0
; X86-NEXT: .long __emutls_t.i

; X86:      .section .rodata,
; X86-LABEL: __emutls_t.i:
; X86-NEXT: .long 15

; X86-NOT:   __emutls_v.i2
; X86-NOT:   __emutls_t.i2

;;;;; 64-bit targets

; X64:      .data
; X64-LABEL: __emutls_v.i:
; X64-NEXT: .quad 4
; X64-NEXT: .quad 4
; X64-NEXT: .quad 0
; X64-NEXT: .quad __emutls_t.i

; X64:      .section .rodata,
; X64-LABEL: __emutls_t.i:
; X64-NEXT: .long 15

; X64-NOT:   __emutls_v.i2
; X64-NOT:   __emutls_t.i2


!llvm.module.flags = !{!0, !1}
!0 = !{i32 1, !"PIC Level", i32 1}
!1 = !{i32 1, !"PIE Level", i32 1}
