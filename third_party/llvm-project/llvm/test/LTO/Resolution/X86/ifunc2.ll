; RUN: llvm-as -o %t1.o %s
; RUN: llvm-as -o %t2.o %S/Inputs/ifunc2.ll
; RUN: llvm-lto2 run %t1.o %t2.o -r %t1.o,foo,p -r %t1.o,foo_resolver, -r %t2.o,foo_resolver,p -save-temps -o %t3.o
; RUN: llvm-dis -o - %t3.o.0.0.preopt.bc | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @foo = ifunc i32 (), i32 ()* @foo_resolver.2
@foo = ifunc i32 (), i32 ()* @foo_resolver

; CHECK: define internal i32 @foo_resolver.2() {
; CHECK-NEXT: ret i32 1
define weak i32 @foo_resolver() {
  ret i32 1
}

; CHECK: define i32 @foo_resolver() {
; CHECK-NEXT: ret i32 2
