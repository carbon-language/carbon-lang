; RUN: llvm-as %s -o %t.o
; RUN: ld -plugin %llvmshlibdir/LLVMgold.so -m elf_x86_64 \
; RUN:    --plugin-opt=-jump-table-type=arity \
; RUN:    --plugin-opt=-mattr=+aes \
; RUN:    -shared %t.o -o %t2.o
; RUN: llvm-nm %t2.o | FileCheck %s

; CHECK: T __llvm_jump_instr_table_0_1
; CHECK: T __llvm_jump_instr_table_1_1

target triple = "x86_64-unknown-linux-gnu"
define i32 @g(i32 %a) unnamed_addr jumptable {
  ret i32 %a
}

define i32 @f() unnamed_addr jumptable {
  ret i32 0
}

define <2 x i64> @test_avx(<2 x i64> %a0, <2 x i64> %a1) {
  %res = call <2 x i64> @llvm.x86.aesni.aesenc(<2 x i64> %a0, <2 x i64> %a1)
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.aesni.aesenc(<2 x i64>, <2 x i64>) nounwind readnone
