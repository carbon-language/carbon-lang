; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

@llvm.global_ctors = appending global [1 x { i32, void()*, i8 } ] [
  { i32, void()*, i8 } { i32 65535, void ()* null, i8 0 }
]
; CHECK: wrong type for intrinsic global variable

@llvm.global_dtors = appending global [1 x { i32, void()*, i8*, i8 } ] [
  { i32, void()*, i8*, i8 } { i32 65535, void ()* null, i8* null, i8 0}
]
; CHECK: wrong type for intrinsic global variable
