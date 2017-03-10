; RUN: opt -module-hash -module-summary %s -o %t.bc
; RUN: opt -module-hash -module-summary %S/Inputs/cache-typeid-resolutions-import.ll -o %t-import.bc

; RUN: llvm-as -o %t1.bc %S/Inputs/cache-typeid-resolutions1.ll

; Two resolutions for typeid1: Unsat, Single
; where both t and t-import are sensitive to typeid1's resolution
; so 4 distinct objects in total.
; RUN: rm -rf %t.cache
; RUN: llvm-lto2 -o %t.o %t.bc %t-import.bc -cache-dir %t.cache -r=%t.bc,f1,plx -r=%t-import.bc,importf1,plx -r=%t-import.bc,f1,lx
; RUN: llvm-lto2 -o %t.o %t.bc %t-import.bc %t1.bc -cache-dir %t.cache -r=%t.bc,f1,plx -r=%t-import.bc,importf1,plx -r=%t-import.bc,f1,lx -r=%t1.bc,vt1,plx
; RUN: ls %t.cache | count 4

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i1 @f1(i8* %p) {
  %x = call i1 @llvm.type.test(i8* %p, metadata !"typeid1")
  ret i1 %x
}

declare i1 @llvm.type.test(i8*, metadata)
