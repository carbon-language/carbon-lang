; RUN: llvm-dis < %s.bc | FileCheck %s
; CHECK-NOT: {i32 @llvm\\.loadl.pd} 
; CHECK: shufflevector
