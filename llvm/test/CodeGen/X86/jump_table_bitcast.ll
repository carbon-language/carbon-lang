; RUN: llc <%s -jump-table-type=single | FileCheck %s
target triple = "x86_64-unknown-linux-gnu"
define i32 @f() unnamed_addr jumptable {
  ret i32 0
}

define i32 @g(i8* %a) unnamed_addr jumptable {
  ret i32 0
}

define void @h(void ()* %func) unnamed_addr jumptable {
  ret void
}

define i32 @main() {
  %g = alloca i32 (...)*, align 8
  store i32 (...)* bitcast (i32 ()* @f to i32 (...)*), i32 (...)** %g, align 8
; CHECK: movq    $__llvm_jump_instr_table_0_[[ENTRY:1|2|3]], (%rsp)
; CHECK: movl    $__llvm_jump_instr_table_0_[[ENTRY]], %ecx
  %1 = load i32 (...)** %g, align 8
  %call = call i32 (...)* %1()
  call void (void ()*)* @h(void ()* bitcast (void (void ()*)* @h to void ()*))
; CHECK: movl    $__llvm_jump_instr_table_0_{{1|2|3}}, %edi
; CHECK: callq   h

  %a = call i32 (i32*)* bitcast (i32 (i8*)* @g to i32(i32*)*)(i32* null)
; CHECK: callq g
  ret i32 %a
}

; CHECK:         .globl  __llvm_jump_instr_table_0_1
; CHECK:         .align  8, 0x90
; CHECK:         .type   __llvm_jump_instr_table_0_1,@function
; CHECK: __llvm_jump_instr_table_0_1:
; CHECK:         jmp     {{f|g|h}}@PLT
; CHECK:         .globl  __llvm_jump_instr_table_0_2
; CHECK:         .align  8, 0x90
; CHECK:         .type   __llvm_jump_instr_table_0_2,@function
; CHECK: __llvm_jump_instr_table_0_2:
; CHECK:         jmp     {{f|g|h}}@PLT
; CHECK:         .globl  __llvm_jump_instr_table_0_3
; CHECK:         .align  8, 0x90
; CHECK:         .type   __llvm_jump_instr_table_0_3,@function
; CHECK: __llvm_jump_instr_table_0_3:
; CHECK:         jmp     {{f|g|h}}@PLT

