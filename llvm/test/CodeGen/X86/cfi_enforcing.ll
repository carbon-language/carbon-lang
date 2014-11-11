; RUN: llc -mtriple=i386-unknown-linux-gnu -fcfi -cfi-enforcing <%s | FileCheck --check-prefix=X86 %s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -fcfi -cfi-enforcing <%s | FileCheck --check-prefix=X86-64 %s

define void @indirect_fun() unnamed_addr jumptable {
  ret void
}

define i32 @m(void ()* %fun) {
  call void ()* %fun()
; CHECK: subl
; X86-64: andq    $8,
; X86-64: leaq    __llvm_jump_instr_table_0_1({{%[a-z0-9]+}}), [[REG:%[a-z0-9]+]]
; X86-64-NOT: callq __llvm_cfi_pointer_warning
; X86-64: callq   *[[REG]]
; X86: andl    $8,
; X86: leal    __llvm_jump_instr_table_0_1({{%[a-z0-9]+}}), [[REG:%[a-z0-9]+]]
; X86-NOT: calll __llvm_cfi_pointer_warning
; X86: calll   *[[REG]]
  ret i32 0
}

define void ()* @get_fun() {
  ret void ()* @indirect_fun
}

define i32 @main(i32 %argc, i8** %argv) {
  %f = call void ()* ()* @get_fun()
  %a = call i32 @m(void ()* %f)
  ret i32 %a
}

; CHECK: .align 8
; CHECK: __llvm_jump_instr_table_0_1:
; CHECK: jmp indirect_fun@PLT
