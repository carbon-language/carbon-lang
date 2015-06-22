; RUN: llvm-link %s /dev/null -S -o - | FileCheck %s

$c = comdat largest

; CHECK: @c = global i32 0, comdat
@c = global i32 0, comdat
