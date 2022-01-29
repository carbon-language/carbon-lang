; RUN: llvm-link -S %s -o - | FileCheck %s

; CHECK-DAG: @foo = private externally_initialized global i8* null
@foo = private externally_initialized global i8* null

@useFoo = global i8** @foo

; CHECK-DAG: @array = appending global [7 x i8] c"abcdefg", align 1
@array = appending global [7 x i8] c"abcdefg", align 1

