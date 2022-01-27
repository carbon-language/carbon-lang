; RUN: llvm-as -o %t.bc %s
; RUN: rm -f %t.0.5.precodegen.bc %t.1.5.precodegen.bc %t.lto.o %t.lto.o1
; RUN: env LD_PRELOAD=%llvmshlibdir/LLVMgold%shlibext %gold -plugin %llvmshlibdir/LLVMgold%shlibext -u foo -u bar -plugin-opt lto-partitions=2 -plugin-opt save-temps -m elf_x86_64 -o %t %t.bc
; RUN: llvm-dis %t.0.5.precodegen.bc -o - | FileCheck --check-prefix=CHECK-BC0 %s
; RUN: llvm-dis %t.1.5.precodegen.bc -o - | FileCheck --check-prefix=CHECK-BC1 %s
; RUN: llvm-nm %t.lto.o | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-nm %t.lto.o1 | FileCheck --check-prefix=CHECK1 %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-BC0: define dso_local void @foo
; CHECK-BC0: declare dso_local void @bar
; CHECK0-NOT: bar
; CHECK0: T foo
; CHECK0-NOT: bar
define void @foo() mustprogress {
  call void @bar()
  ret void
}

; CHECK-BC1: declare dso_local void @foo
; CHECK-BC1: define dso_local void @bar
; CHECK1-NOT: foo
; CHECK1: T bar
; CHECK1-NOT: foo
define void @bar() mustprogress {
  call void @foo()
  ret void
}
