; RUN: llvm-dis < %s.bc | FileCheck %s 
; CHECK-NOT: {@llvm\\.palign}
