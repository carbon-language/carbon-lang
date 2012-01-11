; RUN: llc < %s -mtriple=i686-linux -segmented-stacks -verify-machineinstrs | FileCheck %s -check-prefix=X32
; RUN: llc < %s -mtriple=x86_64-linux  -segmented-stacks -verify-machineinstrs | FileCheck %s -check-prefix=X64

; We used to crash with filetype=obj
; RUN: llc < %s -mtriple=i686-linux -segmented-stacks -filetype=obj
; RUN: llc < %s -mtriple=x86_64-linux -segmented-stacks -filetype=obj

; Just to prevent the alloca from being optimized away
declare void @dummy_use(i32*, i32)

define void @test_basic() {
        %mem = alloca i32, i32 10
        call void @dummy_use (i32* %mem, i32 10)
	ret void

; X32:      test_basic:

; X32:      cmpl %gs:48, %esp
; X32-NEXT: ja      .LBB0_2

; X32:      pushl $0
; X32-NEXT: pushl $60
; X32-NEXT: calll __morestack
; X32-NEXT: ret 

; X64:      test_basic:

; X64:      cmpq %fs:112, %rsp
; X64-NEXT: ja      .LBB0_2

; X64:      movabsq $40, %r10
; X64-NEXT: movabsq $0, %r11
; X64-NEXT: callq __morestack
; X64-NEXT: ret

}

define i32 @test_nested(i32 * nest %closure, i32 %other) {
       %addend = load i32 * %closure
       %result = add i32 %other, %addend
       ret i32 %result

; X32:      cmpl %gs:48, %esp
; X32-NEXT: ja      .LBB1_2

; X32:      pushl $4
; X32-NEXT: pushl $0
; X32-NEXT: calll __morestack
; X32-NEXT: ret

; X64:      cmpq %fs:112, %rsp
; X64-NEXT: ja      .LBB1_2

; X64:      movq %r10, %rax
; X64-NEXT: movabsq $0, %r10
; X64-NEXT: movabsq $0, %r11
; X64-NEXT: callq __morestack
; X64-NEXT: ret
; X64-NEXT: movq %rax, %r10

}

define void @test_large() {
        %mem = alloca i32, i32 10000
        call void @dummy_use (i32* %mem, i32 0)
        ret void

; X32:      leal -40012(%esp), %ecx
; X32-NEXT: cmpl %gs:48, %ecx
; X32-NEXT: ja      .LBB2_2

; X32:      pushl $0
; X32-NEXT: pushl $40012
; X32-NEXT: calll __morestack
; X32-NEXT: ret

; X64:      leaq -40008(%rsp), %r11
; X64-NEXT: cmpq %fs:112, %r11
; X64-NEXT: ja      .LBB2_2

; X64:      movabsq $40008, %r10
; X64-NEXT: movabsq $0, %r11
; X64-NEXT: callq __morestack
; X64-NEXT: ret

}

define fastcc void @test_fastcc() {
        %mem = alloca i32, i32 10
        call void @dummy_use (i32* %mem, i32 10)
        ret void

; X32:      test_fastcc:

; X32:      cmpl %gs:48, %esp
; X32-NEXT: ja      .LBB3_2

; X32:      pushl $0
; X32-NEXT: pushl $60
; X32-NEXT: calll __morestack
; X32-NEXT: ret

; X64:      test_fastcc:

; X64:      cmpq %fs:112, %rsp
; X64-NEXT: ja      .LBB3_2

; X64:      movabsq $40, %r10
; X64-NEXT: movabsq $0, %r11
; X64-NEXT: callq __morestack
; X64-NEXT: ret

}

define fastcc void @test_fastcc_large() {
        %mem = alloca i32, i32 10000
        call void @dummy_use (i32* %mem, i32 0)
        ret void

; X32:      test_fastcc_large:

; X32:      leal -40012(%esp), %eax
; X32-NEXT: cmpl %gs:48, %eax
; X32-NEXT: ja      .LBB4_2

; X32:      pushl $0
; X32-NEXT: pushl $40012
; X32-NEXT: calll __morestack
; X32-NEXT: ret

; X64:      test_fastcc_large:

; X64:      leaq -40008(%rsp), %r11
; X64-NEXT: cmpq %fs:112, %r11
; X64-NEXT: ja      .LBB4_2

; X64:      movabsq $40008, %r10
; X64-NEXT: movabsq $0, %r11
; X64-NEXT: callq __morestack
; X64-NEXT: ret
}
