; RUN: llc <%s -jump-table-type=single | FileCheck --check-prefix=SINGLE %s
; RUN: llc <%s -jump-table-type=arity | FileCheck --check-prefix=ARITY %s
; RUN: llc <%s -jump-table-type=simplified | FileCheck --check-prefix=SIMPL %s
; RUN: llc <%s -jump-table-type=full | FileCheck --check-prefix=FULL %s

target triple = "x86_64-unknown-linux-gnu"

%struct.fun_struct = type { i32 (...)* }

define void @indirect_fun() unnamed_addr jumptable {
  ret void
}

define void @indirect_fun_match() unnamed_addr jumptable {
  ret void
}

define i32 @indirect_fun_i32() unnamed_addr jumptable {
  ret i32 0
}

define i32 @indirect_fun_i32_1(i32 %a) unnamed_addr jumptable {
  ret i32 %a
}

define i32 @indirect_fun_i32_2(i32 %a, i32 %b) unnamed_addr jumptable {
  ret i32 %a
}

define i32* @indirect_fun_i32S_2(i32* %a, i32 %b) unnamed_addr jumptable {
  ret i32* %a
}

define void @indirect_fun_struct(%struct.fun_struct %fs) unnamed_addr jumptable {
  ret void
}

define void @indirect_fun_fun(i32 (...)* %fun, i32 %a) unnamed_addr jumptable {
  ret void
}

define i32 @indirect_fun_fun_ret(i32 (...)* %fun, i32 %a) unnamed_addr jumptable {
  ret i32 %a
}

define void @indirect_fun_array([19 x i8] %a) unnamed_addr jumptable {
  ret void
}

define void @indirect_fun_vec(<3 x i32> %a) unnamed_addr jumptable {
  ret void
}

define void @indirect_fun_vec_2(<4 x float> %a) unnamed_addr jumptable {
  ret void
}

define i32 @m(void ()* %fun) {
  call void ()* %fun()
  ret i32 0
}

define void ()* @get_fun() {
  ret void ()* @indirect_fun
; SINGLE: movl    $__llvm_jump_instr_table_0_
; ARITY: movl    $__llvm_jump_instr_table_
; SIMPL: movl    $__llvm_jump_instr_table_
; FULL: movl    $__llvm_jump_instr_table_
}

define i32 @main(i32 %argc, i8** %argv) {
  %f = call void ()* ()* @get_fun()
  %a = call i32 @m(void ()* %f)
  ret i32 %a
}

; SINGLE-DAG:         .globl  __llvm_jump_instr_table_0_1
; SINGLE-DAG:         .align  8, 0x90
; SINGLE-DAG:         .type   __llvm_jump_instr_table_0_1,@function
; SINGLE-DAG: __llvm_jump_instr_table_0_1:
; SINGLE-DAG:         jmp     indirect_fun_array@PLT
; SINGLE-DAG:         .globl  __llvm_jump_instr_table_0_2
; SINGLE-DAG:         .align  8, 0x90
; SINGLE-DAG:         .type   __llvm_jump_instr_table_0_2,@function
; SINGLE-DAG: __llvm_jump_instr_table_0_2:
; SINGLE-DAG:         jmp     indirect_fun_i32_2@PLT
; SINGLE-DAG:         .globl  __llvm_jump_instr_table_0_3
; SINGLE-DAG:         .align  8, 0x90
; SINGLE-DAG:         .type   __llvm_jump_instr_table_0_3,@function
; SINGLE-DAG: __llvm_jump_instr_table_0_3:
; SINGLE-DAG:         jmp     indirect_fun_vec_2@PLT
; SINGLE-DAG:         .globl  __llvm_jump_instr_table_0_4
; SINGLE-DAG:         .align  8, 0x90
; SINGLE-DAG:         .type   __llvm_jump_instr_table_0_4,@function
; SINGLE-DAG: __llvm_jump_instr_table_0_4:
; SINGLE-DAG:         jmp     indirect_fun_i32S_2@PLT
; SINGLE-DAG:         .globl  __llvm_jump_instr_table_0_5
; SINGLE-DAG:         .align  8, 0x90
; SINGLE-DAG:         .type   __llvm_jump_instr_table_0_5,@function
; SINGLE-DAG: __llvm_jump_instr_table_0_5:
; SINGLE-DAG:         jmp     indirect_fun_struct@PLT
; SINGLE-DAG:         .globl  __llvm_jump_instr_table_0_6
; SINGLE-DAG:         .align  8, 0x90
; SINGLE-DAG:         .type   __llvm_jump_instr_table_0_6,@function
; SINGLE-DAG: __llvm_jump_instr_table_0_6:
; SINGLE-DAG:         jmp     indirect_fun_i32_1@PLT
; SINGLE-DAG:         .globl  __llvm_jump_instr_table_0_7
; SINGLE-DAG:         .align  8, 0x90
; SINGLE-DAG:         .type   __llvm_jump_instr_table_0_7,@function
; SINGLE-DAG: __llvm_jump_instr_table_0_7:
; SINGLE-DAG:         jmp     indirect_fun_i32@PLT
; SINGLE-DAG:         .globl  __llvm_jump_instr_table_0_8
; SINGLE-DAG:         .align  8, 0x90
; SINGLE-DAG:         .type   __llvm_jump_instr_table_0_8,@function
; SINGLE-DAG: __llvm_jump_instr_table_0_8:
; SINGLE-DAG:         jmp     indirect_fun_fun@PLT
; SINGLE-DAG:         .globl  __llvm_jump_instr_table_0_9
; SINGLE-DAG:         .align  8, 0x90
; SINGLE-DAG:         .type   __llvm_jump_instr_table_0_9,@function
; SINGLE-DAG: __llvm_jump_instr_table_0_9:
; SINGLE-DAG:         jmp     indirect_fun_fun_ret@PLT
; SINGLE-DAG:         .globl  __llvm_jump_instr_table_0_10
; SINGLE-DAG:         .align  8, 0x90
; SINGLE-DAG:         .type   __llvm_jump_instr_table_0_10,@function
; SINGLE-DAG: __llvm_jump_instr_table_0_10:
; SINGLE-DAG:         jmp     indirect_fun@PLT
; SINGLE-DAG:         .globl  __llvm_jump_instr_table_0_11
; SINGLE-DAG:         .align  8, 0x90
; SINGLE-DAG:         .type   __llvm_jump_instr_table_0_11,@function
; SINGLE-DAG: __llvm_jump_instr_table_0_11:
; SINGLE-DAG:         jmp     indirect_fun_match@PLT
; SINGLE-DAG:         .globl  __llvm_jump_instr_table_0_12
; SINGLE-DAG:         .align  8, 0x90
; SINGLE-DAG:         .type   __llvm_jump_instr_table_0_12,@function
; SINGLE-DAG: __llvm_jump_instr_table_0_12:
; SINGLE-DAG:         jmp     indirect_fun_vec@PLT
; SINGLE-DAG:         .align  8, 0x90
; SINGLE-DAG:         ud2
; SINGLE-DAG:         .align  8, 0x90
; SINGLE-DAG:         ud2
; SINGLE-DAG:         .align  8, 0x90
; SINGLE-DAG:         ud2
; SINGLE-DAG:         .align  8, 0x90
; SINGLE-DAG:         ud2


; ARITY-DAG:         .globl  __llvm_jump_instr_table_2_1
; ARITY-DAG:         .align  8, 0x90
; ARITY-DAG:         .type   __llvm_jump_instr_table_2_1,@function
; ARITY-DAG: __llvm_jump_instr_table_2_1:
; ARITY-DAG:         jmp     indirect_fun{{.*}}@PLT
; ARITY-DAG:         .align  8, 0x90
; ARITY-DAG:         ud2
; ARITY-DAG:         .globl  __llvm_jump_instr_table_0_1
; ARITY-DAG:         .align  8, 0x90
; ARITY-DAG:         .type   __llvm_jump_instr_table_0_1,@function
; ARITY-DAG: __llvm_jump_instr_table_0_1:
; ARITY-DAG:         jmp     indirect_fun{{.*}}@PLT
; ARITY-DAG:         .globl  __llvm_jump_instr_table_1_1
; ARITY-DAG:         .align  8, 0x90
; ARITY-DAG:         .type   __llvm_jump_instr_table_1_1,@function
; ARITY-DAG: __llvm_jump_instr_table_1_1:
; ARITY-DAG:         jmp     indirect_fun{{.*}}@PLT

; SIMPL-DAG:         .globl  __llvm_jump_instr_table_2_1
; SIMPL-DAG:         .align  8, 0x90
; SIMPL-DAG:         .type   __llvm_jump_instr_table_2_1,@function
; SIMPL-DAG: __llvm_jump_instr_table_2_1:
; SIMPL-DAG:         jmp     indirect_fun{{.*}}@PLT
; SIMPL-DAG:         .align  8, 0x90
; SIMPL-DAG:         ud2
; SIMPL-DAG:         .globl  __llvm_jump_instr_table_0_1
; SIMPL-DAG:         .align  8, 0x90
; SIMPL-DAG:         .type   __llvm_jump_instr_table_0_1,@function
; SIMPL-DAG: __llvm_jump_instr_table_0_1:
; SIMPL-DAG:         jmp     indirect_fun{{.*}}@PLT
; SIMPL-DAG:         .globl  __llvm_jump_instr_table_1_1
; SIMPL-DAG:         .align  8, 0x90
; SIMPL-DAG:         .type   __llvm_jump_instr_table_1_1,@function
; SIMPL-DAG: __llvm_jump_instr_table_1_1:
; SIMPL-DAG:         jmp     indirect_fun{{.*}}@PLT
; SIMPL-DAG:         .globl  __llvm_jump_instr_table_3_1
; SIMPL-DAG:         .align  8, 0x90
; SIMPL-DAG:         .type   __llvm_jump_instr_table_3_1,@function
; SIMPL-DAG: __llvm_jump_instr_table_3_1:
; SIMPL-DAG:         jmp     indirect_fun{{.*}}@PLT
; SIMPL-DAG:         .globl  __llvm_jump_instr_table_4_1
; SIMPL-DAG:         .align  8, 0x90
; SIMPL-DAG:         .type   __llvm_jump_instr_table_4_1,@function
; SIMPL-DAG: __llvm_jump_instr_table_4_1:
; SIMPL-DAG:         jmp     indirect_fun{{.*}}@PLT


; FULL-DAG:        .globl  __llvm_jump_instr_table_10_1
; FULL-DAG:        .align  8, 0x90
; FULL-DAG:        .type   __llvm_jump_instr_table_10_1,@function
; FULL-DAG:__llvm_jump_instr_table_10_1:
; FULL-DAG:        jmp     indirect_fun_i32_1@PLT
; FULL-DAG:        .align  8, 0x90
; FULL-DAG:        ud2
; FULL-DAG:        .globl  __llvm_jump_instr_table_9_1
; FULL-DAG:        .align  8, 0x90
; FULL-DAG:        .type   __llvm_jump_instr_table_9_1,@function
; FULL-DAG:__llvm_jump_instr_table_9_1:
; FULL-DAG:        jmp     indirect_fun_i32_2@PLT
; FULL-DAG:        .align  8, 0x90
; FULL-DAG:        ud2
; FULL-DAG:        .globl  __llvm_jump_instr_table_7_1
; FULL-DAG:        .align  8, 0x90
; FULL-DAG:        .type   __llvm_jump_instr_table_7_1,@function
; FULL-DAG:__llvm_jump_instr_table_7_1:
; FULL-DAG:        jmp     indirect_fun_i32S_2@PLT
; FULL-DAG:        .align  8, 0x90
; FULL-DAG:        ud2
; FULL-DAG:        .globl  __llvm_jump_instr_table_3_1
; FULL-DAG:        .align  8, 0x90
; FULL-DAG:        .type   __llvm_jump_instr_table_3_1,@function
; FULL-DAG:__llvm_jump_instr_table_3_1:
; FULL-DAG:        jmp     indirect_fun_vec_2@PLT
; FULL-DAG:        .align  8, 0x90
; FULL-DAG:        ud2
; FULL-DAG:        .globl  __llvm_jump_instr_table_2_1
; FULL-DAG:        .align  8, 0x90
; FULL-DAG:        .type   __llvm_jump_instr_table_2_1,@function
; FULL-DAG:__llvm_jump_instr_table_2_1:
; FULL-DAG:        jmp     indirect_fun@PLT
; FULL-DAG:        .align  8, 0x90
; FULL-DAG:        ud2
; FULL-DAG:        .align  8, 0x90
; FULL-DAG:        ud2
; FULL-DAG:        .globl  __llvm_jump_instr_table_8_1
; FULL-DAG:        .align  8, 0x90
; FULL-DAG:        .type   __llvm_jump_instr_table_8_1,@function
; FULL-DAG:__llvm_jump_instr_table_8_1:
; FULL-DAG:        jmp     indirect_fun_i32@PLT
; FULL-DAG:        .align  8, 0x90
; FULL-DAG:        ud2
; FULL-DAG:        .globl  __llvm_jump_instr_table_1_1
; FULL-DAG:        .align  8, 0x90
; FULL-DAG:        .type   __llvm_jump_instr_table_1_1,@function
; FULL-DAG:__llvm_jump_instr_table_1_1:
; FULL-DAG:        jmp     indirect_fun_array@PLT
; FULL-DAG:        .align  8, 0x90
; FULL-DAG:        ud2
; FULL-DAG:        .globl  __llvm_jump_instr_table_0_1
; FULL-DAG:        .align  8, 0x90
; FULL-DAG:        .type   __llvm_jump_instr_table_0_1,@function
; FULL-DAG:__llvm_jump_instr_table_0_1:
; FULL-DAG:        jmp     indirect_fun_vec@PLT
; FULL-DAG:        .align  8, 0x90
; FULL-DAG:        ud2
; FULL-DAG:        .globl  __llvm_jump_instr_table_6_1
; FULL-DAG:        .align  8, 0x90
; FULL-DAG:        .type   __llvm_jump_instr_table_6_1,@function
; FULL-DAG:__llvm_jump_instr_table_6_1:
; FULL-DAG:        jmp     indirect_fun_struct@PLT
; FULL-DAG:        .align  8, 0x90
; FULL-DAG:        ud2
; FULL-DAG:        .globl  __llvm_jump_instr_table_5_1
; FULL-DAG:        .align  8, 0x90
; FULL-DAG:        .type   __llvm_jump_instr_table_5_1,@function
; FULL-DAG:__llvm_jump_instr_table_5_1:
; FULL-DAG:        jmp     indirect_fun_fun@PLT
; FULL-DAG:        .align  8, 0x90
; FULL-DAG:        ud2
; FULL-DAG:        .globl  __llvm_jump_instr_table_4_1
; FULL-DAG:        .align  8, 0x90
; FULL-DAG:        .type   __llvm_jump_instr_table_4_1,@function
; FULL-DAG:__llvm_jump_instr_table_4_1:
; FULL-DAG:        jmp     indirect_fun_fun_ret@PLT
; FULL-DAG:        .align  8, 0x90
; FULL-DAG:        ud2
