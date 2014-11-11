; RUN: llc -fcfi -cfi-func-name=cfi_new_failure <%s | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"
define void @indirect_fun() unnamed_addr jumptable {
  ret void
}

define i32 @m(void ()* %fun) {
; CHECK-LABEL: @m
  call void ()* %fun()
; CHECK: callq cfi_new_failure
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
