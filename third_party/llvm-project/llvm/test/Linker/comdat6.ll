; RUN: llvm-link %s %p/Inputs/comdat5.ll -S -o - | FileCheck %s
; RUN: llvm-link %p/Inputs/comdat5.ll %s -S -o - | FileCheck %s
target datalayout = "e-m:w-p:32:32-i64:64-f80:32-n8:16:32-S32"

$foo = comdat largest
@foo = linkonce_odr unnamed_addr constant [1 x i8*] [i8* bitcast (void ()* @bar to i8*)], comdat($foo)

; CHECK: @foo = alias i8*, getelementptr inbounds ([2 x i8*], [2 x i8*]* @some_name, i32 0, i32 1)

declare void @bar() unnamed_addr
