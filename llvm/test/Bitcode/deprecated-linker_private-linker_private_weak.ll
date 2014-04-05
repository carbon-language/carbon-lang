; RUN: llvm-as -o - %s | llvm-dis | FileCheck %s
; RUN: llvm-as -o /dev/null %s 2>&1 | FileCheck %s -check-prefix CHECK-WARNINGS

@.linker_private = linker_private unnamed_addr constant [15 x i8] c"linker_private\00", align 64
@.linker_private_weak = linker_private_weak unnamed_addr constant [20 x i8] c"linker_private_weak\00", align 64

; CHECK: @.linker_private = private unnamed_addr constant [15 x i8] c"linker_private\00", align 64
; CHECK: @.linker_private_weak = private unnamed_addr constant [20 x i8] c"linker_private_weak\00", align 64

; CHECK-WARNINGS: warning: '.linker_private' is deprecated, treating as PrivateLinkage
; CHECK-WARNINGS: @.linker_private = linker_private unnamed_addr constant [15 x i8] c"linker_private\00", align 64
; CHECK-WARNINGS:                    ^

; CHECK-WARNINGS: warning: '.linker_private_weak' is deprecated, treating as PrivateLinkage
; CHECK-WARNINGS: @.linker_private_weak = linker_private_weak unnamed_addr constant [20 x i8] c"linker_private_weak\00", align 64
; CHECK-WARNINGS:                         ^

