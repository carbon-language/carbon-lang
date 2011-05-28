; This isn't really an assembly file. It just runs test on bitcode to ensure
; it is auto-upgraded.
; RUN: llvm-dis < %s.bc | FileCheck %s 
; CHECK-NOT: {i32 @\\.llvm\\.eh}
