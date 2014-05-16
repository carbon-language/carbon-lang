; RUN: not llvm-as %s 2>&1 | FileCheck %s

@foo = global i32 42
@bar = alias internal addrspace(1) i32* @foo

CHECK: error: A type is required if addrspace is given
