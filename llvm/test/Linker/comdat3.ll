; RUN: not llvm-link %s %p/Inputs/comdat2.ll -S -o - 2>&1 | FileCheck %s

$foo = comdat largest
@foo = global i32 43, comdat $foo
; CHECK: Linking COMDATs named 'foo': can't do size dependent selection without DataLayout!
