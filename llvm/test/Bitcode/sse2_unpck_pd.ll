; RUN: llvm-dis < %s.bc | FileCheck %s
; CHECK-NOT: {i32 @llvm\\.unpckh.pd}
; CHECK-NOT: {i32 @llvm\\.unpckl.pd}
; CHECK: shufflevector
