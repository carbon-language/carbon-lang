; RUN: llvm-link %s %p/Inputs/comdat5.ll -S -o - 2>&1 | FileCheck %s
; RUN: llvm-link %p/Inputs/comdat5.ll %s -S -o - 2>&1 | FileCheck %s
target datalayout = "e-m:w-p:32:32-i64:64-f80:32-n8:16:32-S32"
target triple = "i686-pc-windows-msvc"

%struct.S = type { i32 (...)** }

$"\01??_7S@@6B@" = comdat largest
@"\01??_7S@@6B@" = linkonce_odr unnamed_addr constant [1 x i8*] [i8* bitcast (void (%struct.S*, i32)* @"\01??_GS@@UAEPAXI@Z" to i8*)], comdat $"\01??_7S@@6B@"

; CHECK: @"\01??_7S@@6B@" = alias getelementptr inbounds ([2 x i8*]* @some_name, i32 0, i32 1)

declare x86_thiscallcc void @"\01??_GS@@UAEPAXI@Z"(%struct.S*, i32) unnamed_addr
