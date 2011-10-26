; RUN: llc < %s -mtriple=i686-linux -segmented-stacks -verify-machineinstrs | FileCheck %s -check-prefix=X32
; RUN: llc < %s -mtriple=x86_64-linux  -segmented-stacks -verify-machineinstrs | FileCheck %s -check-prefix=X64

; Just to prevent the alloca from being optimized away
declare void @dummy_use(i32*, i32)

define i32 @test_basic(i32 %l) {
        %mem = alloca i32, i32 %l
        call void @dummy_use (i32* %mem, i32 %l)
        %terminate = icmp eq i32 %l, 0
        br i1 %terminate, label %true, label %false

true:
        ret i32 0

false:
        %newlen = sub i32 %l, 1
        %retvalue = call i32 @test_basic(i32 %newlen)
        ret i32 %retvalue

; X32:      test_basic:

; X32:      leal -12(%esp), %ecx
; X32-NEXT: cmpl %gs:48, %ecx

; X32:      subl $8, %esp
; X32-NEXT: pushl $4
; X32-NEXT: pushl $12
; X32-NEXT: calll __morestack
; X32-NEXT: addl $8, %esp
; X32-NEXT: ret 

; X32:      movl %esp, %eax
; X32-NEXT: subl %ecx, %eax
; X32-NEXT: cmpl %eax, %gs:48

; X32:      movl %eax, %esp

; X32:      subl $12, %esp
; X32-NEXT: pushl %ecx
; X32-NEXT: calll __morestack_allocate_stack_space
; X32-NEXT: addl $16, %esp

; X64:      test_basic:

; X64:      leaq -24(%rsp), %r11
; X64-NEXT: cmpq %fs:112, %r11

; X64:      movabsq $24, %r10
; X64-NEXT: movabsq $0, %r11
; X64-NEXT: callq __morestack
; X64-NEXT: ret

; X64:      movq %rsp, %rdi
; X64-NEXT: subq %rax, %rdi
; X64-NEXT: cmpq %rdi, %fs:112

; X64:      movq %rdi, %rsp

; X64:      movq %rax, %rdi
; X64-NEXT: callq __morestack_allocate_stack_space
; X64-NEXT: movq %rax, %rdi

}

define i32 @test_nested(i32 * nest %closure, i32 %other) {
       %addend = load i32 * %closure
       %result = add i32 %other, %addend
       ret i32 %result

; X32:      leal (%esp), %edx
; X32-NEXT: cmpl %gs:48, %edx


; X32:      subl $8, %esp
; X32-NEXT: pushl $4
; X32-NEXT: pushl $0
; X32-NEXT: calll __morestack
; X32-NEXT: addl $8, %esp
; X32-NEXT: ret

; X64:      leaq (%rsp), %r11
; X64-NEXT: cmpq %fs:112, %r11

; X64:      movq %r10, %rax
; X64-NEXT: movabsq $0, %r10
; X64-NEXT: movabsq $0, %r11
; X64-NEXT: callq __morestack
; X64-NEXT: ret
; X64-NEXT: movq %rax, %r10

}
