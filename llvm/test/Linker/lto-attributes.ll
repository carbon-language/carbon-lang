; RUN: llvm-link -S %s -o - | FileCheck %s

; CHECK: @foo = private externally_initialized global i8* null
@foo = private externally_initialized global i8* null
; CHECK: @array = appending global [7 x i8] c"abcdefg", align 1
@array = appending global [7 x i8] c"abcdefg", align 1

