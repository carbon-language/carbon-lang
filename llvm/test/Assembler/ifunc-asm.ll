; RUN: llvm-as < %s | llvm-dis | FileCheck %s --check-prefix=CHECK-LLVM
; RUN: llvm-as < %s -o - | llc -filetype=asm | FileCheck %s --check-prefix=CHECK-ASM

target triple = "x86_64-unknown-linux-gnu"

@foo = ifunc i32 (i32), i64 ()* @foo_ifunc
; CHECK-LLVM: @foo = ifunc i32 (i32), i64 ()* @foo_ifunc
; CHECK-ASM: .type   foo,@gnu_indirect_function

define internal i64 @foo_ifunc() {
entry:
  ret i64 0
}
; CHECK-LLVM: define internal i64 @foo_ifunc()
