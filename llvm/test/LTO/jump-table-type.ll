; RUN: llvm-as <%s >%t1
; RUN: llvm-lto -o %t2 %t1 -jump-table-type=arity
; RUN: llvm-nm %t2 | FileCheck %s

; CHECK: T __llvm_jump_instr_table_0_1
; CHECK: T __llvm_jump_instr_table_1_1

target triple = "x86_64-unknown-linux-gnu"

define i32 @g(i32 %a) unnamed_addr jumptable {
  ret i32 %a
}

define i32 @f() unnamed_addr jumptable {
  ret i32 0
}

define i32 @main() {
  ret i32 0
}

@llvm.used = appending global [2 x i8*]  [i8* bitcast (i32(i32)* @g to i8*),
                                          i8* bitcast (i32()* @f to i8*)]
