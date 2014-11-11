; RUN: llc -filetype=obj <%s -jump-table-type=single -o %t1
; RUN: llvm-objdump -triple=x86_64-unknown-linux-gnu -d %t1 | FileCheck %s
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
  %1 = load i32 (...)** %g, align 8
  %call = call i32 (...)* %1()
  call void (void ()*)* @h(void ()* bitcast (void (void ()*)* @h to void ()*))
  %a = call i32 (i32*)* bitcast (i32 (i8*)* @g to i32(i32*)*)(i32* null)
  ret i32 %a
}

; Make sure that the padding from getJumpInstrTableEntryBound is right.
; CHECK: __llvm_jump_instr_table_0_1:
; CHECK-NEXT: e9 00 00 00 00                                  jmp     0
; CHECK-NEXT: 0f 1f 00                                        nopl    (%rax)
