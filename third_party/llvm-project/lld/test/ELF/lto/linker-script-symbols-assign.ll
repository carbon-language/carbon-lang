; REQUIRES: x86
; RUN: llvm-as %s -o %t.o

; RUN: rm -f %t2.*
; RUN: echo "foo = 1;" > %t.script
; RUN: ld.lld %t.o -o %t2 --script %t.script -save-temps
;; Combined module is not empty, but it will be empty after optimization.
;; Ensure lld still emits empty combined obj in this case.
; RUN: llvm-nm %t2.lto.o | count 0

; RUN: llvm-readobj --symbols %t2 | FileCheck %s --check-prefix=VAL
; VAL:       Symbol {
; VAL:        Name: foo
; VAL-NEXT:   Value: 0x1
; VAL-NEXT:   Size:
; VAL-NEXT:   Binding: Global
; VAL-NEXT:   Type: None
; VAL-NEXT:   Other:
; VAL-NEXT:   Section: Absolute
; VAL-NEXT: }

; RUN: echo "zed = 1;" > %t2.script
; RUN: ld.lld %t.o -o %t3 --script %t2.script
; RUN: llvm-readobj --symbols %t3 | FileCheck %s --check-prefix=ABS
; ABS:      Symbol {
; ABS:        Name: zed
; ABS-NEXT:   Value: 0x1
; ABS-NEXT:   Size: 0
; ABS-NEXT:   Binding: Global
; ABS-NEXT:   Type: None
; ABS-NEXT:   Other: 0
; ABS-NEXT:   Section: Absolute
; ABS-NEXT: }

target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

@foo = global i32 0
@bar = global i32 0
