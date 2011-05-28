; RUN: llvm-dis < %s.bc | FileCheck %s
; CHECK-NOT: {i32 @llvm\\.shuf.pd}
; CHECK: shufflevector
