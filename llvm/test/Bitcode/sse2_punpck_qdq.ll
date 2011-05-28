; RUN: llvm-dis < %s.bc | FileCheck %s
; CHECK-NOT: {i32 @llvm\\.punpckh.qdq}
; CHECK-NOT: {i32 @llvm\\.punpckl.qdq}
; CHECK: shufflevector
