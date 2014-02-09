// RUN: llvm-mc -triple x86_64-apple-darwin10 %s | FileCheck %s

.macro .make_macro
$0 $1
$2 $3
$4
.endmacro

.make_macro .macro,.mybyte,.byte,$0,.endmacro

.data
// CHECK: .byte 10
.mybyte 10

// PR18599
.macro macro_a

.macro macro_b
.byte 10
.macro macro_c
.endm

macro_c
.purgem macro_c
.endm

macro_b
.endm

macro_a
macro_b
// CHECK: .byte 10
// CHECK: .byte 10
