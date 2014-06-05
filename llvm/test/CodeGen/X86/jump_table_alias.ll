; RUN: llc <%s -jump-table-type=single | FileCheck %s
target triple = "x86_64-unknown-linux-gnu"
define i32 @f() unnamed_addr jumptable {
entry:
  ret i32 0
}

@i = alias internal i32 ()* @f
@j = alias i32 ()* @f

define i32 @main(i32 %argc, i8** %argv) {
  %temp = alloca i32 ()*, align 8
  store i32 ()* @i, i32()** %temp, align 8
; CHECK: movq    $__llvm_jump_instr_table_0_1
  %1 = load i32 ()** %temp, align 8
; CHECK: movl    $__llvm_jump_instr_table_0_1
  %2 = call i32 ()* %1()
  %3 = call i32 ()* @i()
; CHECK: callq   i
  %4 = call i32 ()* @j()
; CHECK: callq   j
  ret i32 %3
}

; There should only be one table, even though there are two GlobalAliases,
; because they both alias the same value.

; CHECK:         .globl  __llvm_jump_instr_table_0_1
; CHECK:         .align  8, 0x90
; CHECK:         .type   __llvm_jump_instr_table_0_1,@function
; CHECK: __llvm_jump_instr_table_0_1:
; CHECK:         jmp     f@PLT

