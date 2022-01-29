; RUN: llvm-as %s -o %t.o
; RUN: llvm-as %p/Inputs/type-merge2.ll -o %t2.o
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    -m elf_x86_64 \
; RUN:    --plugin-opt=save-temps \
; RUN:    -shared %t.o %t2.o -o %t3.o
; RUN: llvm-dis %t3.o.0.2.internalize.bc -o - | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%zed = type { i8 }
define void @foo()  {
  call void @bar(%zed* null)
  ret void
}
declare void @bar(%zed*)

; CHECK:      %zed = type { i8 }
; CHECK-NEXT: %zed.0 = type { i16 }

; CHECK:      define void @foo() {
; CHECK-NEXT:   call void bitcast (void (%zed.0*)* @bar to void (%zed*)*)(%zed* null)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK:      define void @bar(%zed.0* %this) {
; CHECK-NEXT:   store %zed.0* %this, %zed.0** null
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
