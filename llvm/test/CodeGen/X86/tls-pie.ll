; RUN: llc < %s -march=x86 -mtriple=i386-linux-gnu -relocation-model=pic -enable-pie \
; RUN:   | FileCheck -check-prefix=X32 %s
; RUN: llc < %s -march=x86-64 -mtriple=x86_64-linux-gnu -relocation-model=pic -enable-pie \
; RUN:   | FileCheck -check-prefix=X64 %s

@i = thread_local global i32 15
@i2 = external thread_local global i32

define i32 @f1() {
; X32: f1:
; X32:      movl %gs:i@NTPOFF, %eax
; X32-NEXT: ret
; X64: f1:
; X64:      movl %fs:i@TPOFF, %eax
; X64-NEXT: ret

entry:
	%tmp1 = load i32* @i
	ret i32 %tmp1
}

define i32* @f2() {
; X32: f2:
; X32:      movl %gs:0, %eax
; X32-NEXT: leal i@NTPOFF(%eax), %eax
; X32-NEXT: ret
; X64: f2:
; X64:      movq %fs:0, %rax
; X64-NEXT: leaq i@TPOFF(%rax), %rax
; X64-NEXT: ret

entry:
	ret i32* @i
}

define i32 @f3() {
; X32: f3:
; X32:      movl i2@INDNTPOFF, %eax
; X32-NEXT: movl %gs:(%eax), %eax
; X32-NEXT: ret
; X64: f3:
; X64:      movq i2@GOTTPOFF(%rip), %rax
; X64-NEXT: movl %fs:(%rax), %eax
; X64-NEXT: ret

entry:
	%tmp1 = load i32* @i2
	ret i32 %tmp1
}

define i32* @f4() {
; X32: f4:
; X32:      movl %gs:0, %eax
; X32-NEXT: addl i2@INDNTPOFF, %eax
; X32-NEXT: ret
; X64: f4:
; X64:      movq %fs:0, %rax
; X64-NEXT: addq i2@GOTTPOFF(%rip), %rax
; X64-NEXT: ret

entry:
	ret i32* @i2
}
