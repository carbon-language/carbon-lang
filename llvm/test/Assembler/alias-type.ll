; RUN: not llvm-as %s 2>&1 | FileCheck %s

@foo = global i32 42
@bar = alias i32 @foo

CHECK: error: An alias must have pointer type
