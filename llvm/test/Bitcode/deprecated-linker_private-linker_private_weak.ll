; RUN: llvm-as -o - %s | llvm-dis | FileCheck %s

@.linker_private = linker_private unnamed_addr constant [15 x i8] c"linker_private\00", align 64
@.linker_private_weak = linker_private_weak unnamed_addr constant [20 x i8] c"linker_private_weak\00", align 64

; CHECK: @.linker_private = private unnamed_addr constant [15 x i8] c"linker_private\00", align 64
; CHECK: @.linker_private_weak = private unnamed_addr constant [20 x i8] c"linker_private_weak\00", align 64

