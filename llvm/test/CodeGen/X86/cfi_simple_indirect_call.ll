; RUN: llc -fcfi -cfi-type=sub <%s | FileCheck --check-prefix=SUB %s
; RUN: llc -fcfi -cfi-type=add <%s | FileCheck --check-prefix=ADD %s
; RUN: llc -fcfi -cfi-type=ror <%s | FileCheck --check-prefix=ROR %s

target triple = "x86_64-unknown-linux-gnu"

define void @indirect_fun() unnamed_addr jumptable {
  ret void
}

define i32 @m(void ()* %fun) {
  call void ()* %fun()
; SUB: subl    
; SUB: andq    $8
; SUB-LABEL: leaq    __llvm_jump_instr_table_0_1
; SUB-LABEL: callq   __llvm_cfi_pointer_warning

; ROR: subq
; ROR: rolq    $61
; ROR: testq
; ROR-LABEL: callq   __llvm_cfi_pointer_warning

; ADD: andq    $8
; ADD-LABEL: leaq    __llvm_jump_instr_table_0_1
; ADD: cmpq
; ADD-LABEL: callq   __llvm_cfi_pointer_warning
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
; SUB: .text
; SUB: .align 8
; SUB-LABEL: .type __llvm_jump_instr_table_0_1,@function
; SUB-LABEL:__llvm_jump_instr_table_0_1:
; SUB-LABEL: jmp indirect_fun@PLT
